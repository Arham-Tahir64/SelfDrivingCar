from __future__ import annotations

import time
from dataclasses import dataclass

from autonomy_demo.eval.metrics import LatencyAccumulator
from autonomy_demo.interfaces.contracts import (
    BehaviorPlanner,
    Controller,
    EvaluationHarness,
    MappingModule,
    MotionPlanner,
    PerceptionModule,
    PredictionModule,
    ReplayWriter,
    RuntimeContext,
    SimulationBackend,
    VisualizationSink,
)
from autonomy_demo.interfaces.enums import TopicName
from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import ReplayFrame, ScenarioConfig
from autonomy_demo.perception.bev_projection import BEVDrivableProjector

_logger = get_logger(__name__)

_TOP_LEVEL_LATENCY_KEYS = (
    "perception",
    "localization",
    "mapping",
    "prediction",
    "planning",
    "control",
)

_AUXILIARY_LATENCY_KEYS = (
    "segformer_drivable",
    "learned_lanes",
)


def _time_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass(slots=True)
class PipelineRuntime:
    context: RuntimeContext
    simulation: SimulationBackend
    sensors: any
    perception: PerceptionModule
    localization: any
    mapping: MappingModule
    prediction: PredictionModule
    behavior_planner: BehaviorPlanner
    motion_planner: MotionPlanner
    controller: Controller
    replay_writer: ReplayWriter | None
    visualization: VisualizationSink | None
    evaluation: EvaluationHarness

    def run(self, scenario: ScenarioConfig, max_ticks: int) -> None:
        latency = LatencyAccumulator()
        bev_projector = BEVDrivableProjector()
        stable_traffic_light_anchors = []
        try:
            self.simulation.bootstrap(scenario)
            if hasattr(self.simulation, "get_traffic_light_anchors"):
                stable_traffic_light_anchors = list(self.simulation.get_traffic_light_anchors())
            self.simulation.attach_sensors()
            self.sensors.setup()
            self.sensors.warmup(self.simulation)
            if hasattr(self.localization, "prepare"):
                self.localization.prepare(self.simulation, scenario)
            if hasattr(self.mapping, "prepare"):
                self.mapping.prepare(self.simulation, scenario)
            if hasattr(self.prediction, "prepare"):
                self.prediction.prepare(self.simulation, scenario)
            if hasattr(self.behavior_planner, "prepare"):
                self.behavior_planner.prepare(self.simulation, scenario)
            if hasattr(self.motion_planner, "prepare_route"):
                self.motion_planner.prepare_route(self.simulation, scenario)
            if hasattr(self.evaluation, "set_route_plan"):
                self.evaluation.set_route_plan(getattr(self.motion_planner, "route_plan", None))
            if self.visualization:
                self.visualization.attach(self.context.event_bus)
            self.context.event_bus.publish(
                TopicName.SCENARIO_INFO.value,
                {
                    "scenario_id": scenario.scenario_id,
                    "name": scenario.name,
                    "map_name": scenario.map_name,
                    "max_duration_s": scenario.max_duration_s,
                },
            )
            lane_graph_provider = getattr(self.mapping, "lane_graph_provider", None)
            lane_graph = getattr(lane_graph_provider, "lane_graph", None)
            if lane_graph is not None:
                prior_map = bev_projector.build_prior_map(
                    lane_graph=lane_graph,
                    map_name=scenario.map_name,
                    route_plan=getattr(self.motion_planner, "route_plan", None),
                    stable_traffic_lights=stable_traffic_light_anchors,
                )
                self.context.event_bus.publish(
                    TopicName.VISUALIZATION_PRIOR_MAP.value,
                    prior_map,
                )
            for tick_id in range(max_ticks):
                self.simulation.tick(tick_id)
                sim_time_s = tick_id / 20.0
                snapshot = getattr(self.simulation, "current_snapshot", None)
                if snapshot is not None and hasattr(snapshot, "timestamp"):
                    sim_time_s = float(snapshot.timestamp.elapsed_seconds)
                bundle = self.sensors.capture(tick_id, sim_time_s)
                self.context.event_bus.publish(TopicName.SENSOR_CAMERA_FRONT.value, bundle.front_camera)
                self.context.event_bus.publish(TopicName.SENSOR_LIDAR.value, bundle.lidar)

                # --- Timed pipeline stages ---
                t0 = _time_ms()
                detections, lanes, drivable_space, traffic_lights, _cones = self.perception.run(bundle)
                t1 = _time_ms()
                latency.record("perception", t1 - t0)
                latency.record("segformer_drivable", float(bundle.metadata.get("drivable_inference_ms", 0.0)))
                latency.record("learned_lanes", float(bundle.metadata.get("lane_inference_ms", 0.0)))

                bundle.metadata["debug_perception_detections"] = detections
                if self.visualization and hasattr(self.visualization, "update_bundle"):
                    self.visualization.update_bundle(bundle)

                t0 = _time_ms()
                ego_pose = self.localization.run(bundle)
                t1 = _time_ms()
                latency.record("localization", t1 - t0)

                t0 = _time_ms()
                local_map = self.mapping.run(detections, lanes, drivable_space, [], traffic_lights, ego_pose)
                t1 = _time_ms()
                latency.record("mapping", t1 - t0)

                bev_drivable_grid = None
                road_corridor = None
                world_layer = None
                try:
                    road_corridor = bev_projector.build_route_corridor(
                        local_map,
                        ego_pose,
                        route_plan=getattr(self.motion_planner, "route_plan", None),
                    )
                    world_layer = bev_projector.build_world_layer(
                        local_map,
                        ego_pose,
                        route_plan=getattr(self.motion_planner, "route_plan", None),
                        route_lane_ids={strip["lane_id"] for strip in road_corridor.get("strips", [])},
                        stable_traffic_lights=stable_traffic_light_anchors,
                        live_traffic_lights=traffic_lights,
                    )
                    if drivable_space is not None and ego_pose is not None:
                        world_points_xy, confidences = bev_projector.project_world_points(
                            drivable_space,
                            ego_pose,
                            camera_calibration=(
                                bundle.metadata.get("camera_calibration", {}) or {}
                            ).get("front_camera"),
                        )
                        bev_projector.update_world_history(
                            world_points_xy,
                            confidences,
                            sim_time_s=float(sim_time_s),
                            corridor_polygons_xy=list(road_corridor.get("polygons_xy", [])),
                        )
                    if ego_pose is not None:
                        bev_drivable_grid = bev_projector.render_local_crop(
                            ego_pose,
                            sim_time_s=float(sim_time_s),
                        )
                except (ValueError, TypeError, KeyError) as exc:
                    _logger.debug("BEV projection skipped for tick %s: %s", tick_id, exc)
                except Exception:
                    _logger.warning("BEV projection failed for tick %s", tick_id, exc_info=True)

                t0 = _time_ms()
                predictions = self.prediction.run(local_map)
                t1 = _time_ms()
                latency.record("prediction", t1 - t0)

                if hasattr(self.behavior_planner, "set_context"):
                    self.behavior_planner.set_context(local_map, predictions)

                t0 = _time_ms()
                behavior_state = self.behavior_planner.run(local_map, ego_pose)
                trajectory = self.motion_planner.run(local_map, ego_pose, predictions, behavior_state)
                t1 = _time_ms()
                latency.record("planning", t1 - t0)

                if hasattr(self.controller, "set_context"):
                    self.controller.set_context(local_map, predictions)

                t0 = _time_ms()
                command = self.controller.run(trajectory, ego_pose)
                t1 = _time_ms()
                latency.record("control", t1 - t0)

                self.context.event_bus.publish(TopicName.PERCEPTION_DETECTIONS.value, detections)
                self.context.event_bus.publish(TopicName.PERCEPTION_LANES.value, lanes)
                self.context.event_bus.publish(TopicName.PERCEPTION_DRIVABLE_SPACE.value, drivable_space)
                seg_map_data = bundle.metadata.get("semantic_seg_map")
                if seg_map_data:
                    front_seg = seg_map_data.get(bundle.front_camera.sensor_id)
                    if front_seg is not None:
                        self.context.event_bus.publish(TopicName.PERCEPTION_SEMANTIC_SEG.value, front_seg)
                self.context.event_bus.publish(TopicName.PERCEPTION_TRAFFIC_LIGHTS.value, traffic_lights)
                if "perception_summary" in bundle.metadata:
                    self.context.event_bus.publish(
                        TopicName.PERCEPTION_STATUS.value,
                        bundle.metadata["perception_summary"],
                    )
                self.context.event_bus.publish(TopicName.LOCALIZATION_EGO_POSE.value, ego_pose)
                self.context.event_bus.publish(TopicName.MAP_LOCAL_MAP.value, local_map)
                self.context.event_bus.publish(TopicName.PREDICTION_AGENTS.value, predictions)
                self.context.event_bus.publish(TopicName.PLANNING_EGO_TRAJECTORY.value, trajectory)
                # Publish planner candidates for dashboard visualization
                candidates = getattr(self.motion_planner, "last_candidates", None)
                if candidates:
                    self.context.event_bus.publish(TopicName.PLANNING_CANDIDATES.value, candidates)
                self.context.event_bus.publish(TopicName.CONTROL_VEHICLE_COMMAND.value, command)
                if bev_drivable_grid is not None:
                    self.context.event_bus.publish(TopicName.VISUALIZATION_BEV_DRIVABLE.value, bev_drivable_grid)
                if road_corridor is not None:
                    self.context.event_bus.publish(
                        TopicName.VISUALIZATION_ROAD_CORRIDOR.value,
                        {"strips": list(road_corridor.get("strips", []))},
                    )
                if world_layer is not None:
                    self.context.event_bus.publish(
                        TopicName.VISUALIZATION_WORLD_LAYER.value,
                        world_layer,
                    )

                # Publish per-tick latency for the dashboard
                tick_latency = latency.latest()
                tick_latency["total"] = sum(float(tick_latency.get(key, 0.0)) for key in _TOP_LEVEL_LATENCY_KEYS)
                auxiliary_total = sum(float(tick_latency.get(key, 0.0)) for key in _AUXILIARY_LATENCY_KEYS)
                tick_latency["perception_aux_total"] = auxiliary_total
                self.context.event_bus.publish(TopicName.PIPELINE_LATENCY.value, tick_latency)

                self.context.event_bus.publish(
                    TopicName.TICK_COMPLETE.value,
                    {"tick_id": tick_id, "sim_time_s": sim_time_s},
                )

                self.simulation.apply_control(command)
                snapshot = self.context.event_bus.snapshot()
                self.evaluation.update(tick_id, snapshot)
                if self.replay_writer and self.context.record_replay:
                    self.replay_writer.record(
                        ReplayFrame(tick_id=tick_id, sim_time_s=sim_time_s, topic_payloads=snapshot)
                    )
        finally:
            # Feed accumulated latency into evaluation harness
            if hasattr(self.evaluation, "set_latency"):
                self.evaluation.set_latency(latency)
            self.simulation.shutdown()
            if self.visualization:
                self.visualization.flush()
