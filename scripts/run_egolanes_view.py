from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.common.logging import configure_logging
from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.control.controller import RouteFollowerController, StubController
from autonomy_demo.localization.module import build_localization_module
from autonomy_demo.mapping.lane_graph import LaneGraphProvider
from autonomy_demo.mapping.module import build_mapping_module
from autonomy_demo.orchestration.config_loader import load_runtime_config, load_sensor_config
from autonomy_demo.orchestration.scenario_loader import load_scenario_config
from autonomy_demo.perception.module import build_perception_module
from autonomy_demo.planning.behavior_fsm import RuleBasedBehaviorPlanner, StubBehaviorPlanner
from autonomy_demo.planning.motion_planner import FrenetMotionPlanner, StubMotionPlanner
from autonomy_demo.prediction.module import build_prediction_module
from autonomy_demo.sensors.carla_sensor_suite import CarlaSensorSuite
from autonomy_demo.sensors.sensor_manager import SensorManager
from autonomy_demo.sim.backends import CarlaSimulationBackend, StubSimulationBackend
from autonomy_demo.visualization.pygame_egolanes_view import PygameEgoLanesViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scenario and display EgoLanes in a pygame window.")
    parser.add_argument("--config", required=True, help="Path to scenario JSON.")
    parser.add_argument("--app-config", default="configs/app.low_perf.yaml")
    parser.add_argument("--sensor-config", default="configs/sensors.low_perf.yaml")
    parser.add_argument("--backend", choices=["stub", "carla"], default=None)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--perception-mode", choices=["stub", "camera_v1", "lidar_v1", "fused_v1"], default=None)
    parser.add_argument("--perception-device", default=None)
    parser.add_argument("--perception-model-variant", default=None)
    return parser.parse_args()


def _build_runtime_components(runtime_config, sensor_config: dict):
    if runtime_config.backend == "carla":
        backend = CarlaSimulationBackend(runtime_config)
        sensors = CarlaSensorSuite(sensor_config, backend)
        return backend, sensors
    backend = StubSimulationBackend(runtime_config)
    sensors = SensorManager(sensor_config)
    return backend, sensors


def _build_planning_stack(runtime_config):
    lane_graph_provider = LaneGraphProvider()
    localization = build_localization_module(runtime_config, lane_graph_provider)
    mapping = build_mapping_module(runtime_config, lane_graph_provider)
    prediction = build_prediction_module(runtime_config)
    if runtime_config.backend == "carla":
        tuning = getattr(runtime_config, "tuning", {}) or {}
        behavior_tuning = tuning.get("behavior", {})
        planning_tuning = tuning.get("planning", {})
        control_tuning = tuning.get("control", {})
        behavior_planner = (
            RuleBasedBehaviorPlanner(
                **{
                    key: value
                    for key, value in behavior_tuning.items()
                    if key in RuleBasedBehaviorPlanner.__init__.__code__.co_varnames
                }
            )
            if behavior_tuning
            else RuleBasedBehaviorPlanner()
        )
        planning_kwargs = (
            {
                key: value
                for key, value in planning_tuning.items()
                if key in FrenetMotionPlanner.__init__.__code__.co_varnames
            }
            if planning_tuning
            else {}
        )
        planning_kwargs["lane_graph_provider"] = lane_graph_provider
        motion_planner = FrenetMotionPlanner(**planning_kwargs)
        controller = (
            RouteFollowerController(
                **{
                    key: value
                    for key, value in control_tuning.items()
                    if key in RouteFollowerController.__init__.__code__.co_varnames
                }
            )
            if control_tuning
            else RouteFollowerController()
        )
    else:
        behavior_planner = StubBehaviorPlanner()
        motion_planner = StubMotionPlanner()
        controller = StubController()
    return lane_graph_provider, localization, mapping, prediction, behavior_planner, motion_planner, controller


def _prepare_modules(
    *,
    simulation,
    scenario,
    localization,
    mapping,
    prediction,
    behavior_planner,
    motion_planner,
) -> None:
    if hasattr(localization, "prepare"):
        localization.prepare(simulation, scenario)
    if hasattr(mapping, "prepare"):
        mapping.prepare(simulation, scenario)
    if hasattr(prediction, "prepare"):
        prediction.prepare(simulation, scenario)
    if hasattr(behavior_planner, "prepare"):
        behavior_planner.prepare(simulation, scenario)
    if hasattr(motion_planner, "prepare_route"):
        motion_planner.prepare_route(simulation, scenario)


def _egolanes_runtime_info(perception) -> list[str]:
    extractor = getattr(perception, "egolanes_extractor", None)
    if extractor is None and hasattr(perception, "camera_stack"):
        extractor = getattr(perception.camera_stack, "egolanes_extractor", None)
    if extractor is None:
        return ["egolanes extractor: unavailable"]

    providers = []
    session = getattr(extractor, "_session", None)
    if session is not None and hasattr(session, "get_providers"):
        providers = list(session.get_providers())
    provider_text = ",".join(providers) if providers else "unloaded"
    lines = [
        f"provider: {provider_text}",
        f"lane ms: {float(getattr(extractor, 'last_inference_ms', 0.0)):.1f}",
    ]
    if getattr(extractor, "load_error", None):
        lines.append(f"load error: {extractor.load_error}")
    return lines


def _get_egolanes_extractor(perception):
    extractor = getattr(perception, "egolanes_extractor", None)
    if extractor is None and hasattr(perception, "camera_stack"):
        extractor = getattr(perception.camera_stack, "egolanes_extractor", None)
    return extractor


def main() -> int:
    args = parse_args()
    runtime_config = load_runtime_config(Path(args.app_config))
    if args.backend:
        runtime_config.backend = args.backend
    if args.max_ticks is not None:
        runtime_config.max_ticks = args.max_ticks
    if args.perception_mode:
        runtime_config.perception_mode = args.perception_mode
    if args.perception_device:
        runtime_config.perception_device = args.perception_device
    if args.perception_model_variant:
        runtime_config.perception_model_variant = args.perception_model_variant

    configure_logging(Path("configs/logging.yaml"), runtime_config.log_level)
    sensor_config = load_sensor_config(Path(args.sensor_config))
    scenario = load_scenario_config(Path(args.config), Path("scenarios/schema/scenario.schema.json"))
    output_dir = ensure_directory(runtime_config.output_dir / f"{scenario.scenario_id}_egolanes_view")

    simulation, sensors = _build_runtime_components(runtime_config, sensor_config)
    perception = build_perception_module(runtime_config)
    _, localization, mapping, prediction, behavior_planner, motion_planner, controller = _build_planning_stack(runtime_config)
    viewer = PygameEgoLanesViewer(output_dir=output_dir)
    egolanes_extractor = _get_egolanes_extractor(perception)

    max_ticks = runtime_config.max_ticks
    if max_ticks <= 0:
        max_ticks = max(1, math.ceil(scenario.max_duration_s * runtime_config.tick_hz))

    try:
        simulation.bootstrap(scenario)
        simulation.attach_sensors()
        sensors.setup()
        sensors.warmup(simulation)
        if egolanes_extractor is not None:
            try:
                egolanes_extractor._ensure_loaded()
            except Exception:
                pass
        _prepare_modules(
            simulation=simulation,
            scenario=scenario,
            localization=localization,
            mapping=mapping,
            prediction=prediction,
            behavior_planner=behavior_planner,
            motion_planner=motion_planner,
        )

        for tick_id in range(max_ticks):
            simulation.tick(tick_id)
            sim_time_s = tick_id / float(max(runtime_config.tick_hz, 1))
            snapshot = getattr(simulation, "current_snapshot", None)
            if snapshot is not None and hasattr(snapshot, "timestamp"):
                sim_time_s = float(snapshot.timestamp.elapsed_seconds)

            bundle = sensors.capture(tick_id, sim_time_s)
            detections, lanes, drivable_space, traffic_lights, _cones = perception.run(bundle)
            lane_source = str(bundle.metadata.get("lane_source", "unknown"))
            status_lines = []
            status_lines.extend(_egolanes_runtime_info(perception))
            status_lines.append(f"lane source: {lane_source}")
            status_lines.append(f"perception: {bundle.metadata.get('perception_status', 'n/a')}")
            if "lane_fallback_reason" in bundle.metadata:
                status_lines.append(f"fallback: {bundle.metadata['lane_fallback_reason']}")
            if "perception_error" in bundle.metadata:
                status_lines.append(f"error: {bundle.metadata['perception_error']}")
            if "ego_lane_id" in bundle.metadata:
                status_lines.append(f"ego lane: {bundle.metadata['ego_lane_id']}")

            if not viewer.render(
                bundle=bundle,
                lanes=lanes,
                lane_source=lane_source,
                status_lines=status_lines,
            ):
                break

            ego_pose = localization.run(bundle)
            local_map = mapping.run(detections, lanes, drivable_space, [], traffic_lights, ego_pose)
            predictions = prediction.run(local_map)

            if hasattr(behavior_planner, "set_context"):
                behavior_planner.set_context(local_map, predictions)
            behavior_state = behavior_planner.run(local_map, ego_pose)
            trajectory = motion_planner.run(local_map, ego_pose, predictions, behavior_state)

            if hasattr(controller, "set_context"):
                controller.set_context(local_map, predictions)
            command = controller.run(trajectory, ego_pose)
            simulation.apply_control(command)
    finally:
        simulation.shutdown()
        viewer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
