from __future__ import annotations

from dataclasses import dataclass

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
from autonomy_demo.interfaces.types import ReplayFrame, ScenarioConfig


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
        try:
            self.simulation.bootstrap(scenario)
            self.simulation.attach_sensors()
            self.sensors.setup()
            self.sensors.warmup(self.simulation)
            if self.visualization:
                self.visualization.attach(self.context.event_bus)
            for tick_id in range(max_ticks):
                self.simulation.tick(tick_id)
                sim_time_s = tick_id / 20.0
                snapshot = getattr(self.simulation, "current_snapshot", None)
                if snapshot is not None and hasattr(snapshot, "timestamp"):
                    sim_time_s = float(snapshot.timestamp.elapsed_seconds)
                bundle = self.sensors.capture(tick_id, sim_time_s)
                self.context.event_bus.publish(TopicName.SENSOR_CAMERA_FRONT.value, bundle.front_camera)
                self.context.event_bus.publish(TopicName.SENSOR_LIDAR.value, bundle.lidar)
                detections, lanes, drivable_space, traffic_lights, cones = self.perception.run(bundle)
                ego_pose = self.localization.run(bundle)
                local_map = self.mapping.run(detections, lanes, drivable_space, cones, traffic_lights, ego_pose)
                predictions = self.prediction.run(local_map)
                behavior_state = self.behavior_planner.run(local_map, ego_pose)
                trajectory = self.motion_planner.run(local_map, ego_pose, predictions, behavior_state)
                command = self.controller.run(trajectory, ego_pose)

                self.context.event_bus.publish(TopicName.PERCEPTION_DETECTIONS.value, detections)
                self.context.event_bus.publish(TopicName.PERCEPTION_LANES.value, lanes)
                self.context.event_bus.publish(TopicName.PERCEPTION_DRIVABLE_SPACE.value, drivable_space)
                self.context.event_bus.publish(TopicName.PERCEPTION_TRAFFIC_LIGHTS.value, traffic_lights)
                self.context.event_bus.publish(TopicName.PERCEPTION_CONES.value, cones)
                self.context.event_bus.publish(TopicName.LOCALIZATION_EGO_POSE.value, ego_pose)
                self.context.event_bus.publish(TopicName.MAP_LOCAL_MAP.value, local_map)
                self.context.event_bus.publish(TopicName.PREDICTION_AGENTS.value, predictions)
                self.context.event_bus.publish(TopicName.PLANNING_EGO_TRAJECTORY.value, trajectory)
                self.context.event_bus.publish(TopicName.CONTROL_VEHICLE_COMMAND.value, command)

                self.simulation.apply_control(command)
                snapshot = self.context.event_bus.snapshot()
                self.evaluation.update(tick_id, snapshot)
                if self.replay_writer and self.context.record_replay:
                    self.replay_writer.record(
                        ReplayFrame(tick_id=tick_id, sim_time_s=sim_time_s, topic_payloads=snapshot)
                    )
        finally:
            self.simulation.shutdown()
            if self.visualization:
                self.visualization.flush()
