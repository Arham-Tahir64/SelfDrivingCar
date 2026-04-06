from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.control.controller import RouteFollowerController, StubController
from autonomy_demo.eval.harness import LiveEvaluationHarness, SimpleEvaluationHarness
from autonomy_demo.interfaces.contracts import RuntimeContext
from autonomy_demo.localization.module import build_localization_module
from autonomy_demo.mapping.lane_graph import LaneGraphProvider
from autonomy_demo.mapping.module import build_mapping_module
from autonomy_demo.orchestration.event_bus import InProcessEventBus
from autonomy_demo.orchestration.pipeline_runtime import PipelineRuntime
from autonomy_demo.perception.module import build_perception_module
from autonomy_demo.planning.behavior_fsm import RuleBasedBehaviorPlanner, StubBehaviorPlanner
from autonomy_demo.planning.motion_planner import FrenetMotionPlanner, StubMotionPlanner
from autonomy_demo.prediction.module import build_prediction_module
from autonomy_demo.replay.writer import Hdf5OrJsonReplayWriter
from autonomy_demo.sensors.carla_sensor_suite import CarlaSensorSuite
from autonomy_demo.sensors.sensor_manager import SensorManager
from autonomy_demo.visualization.composite import CompositeVisualizationSink
from autonomy_demo.sim.backends import CarlaSimulationBackend, StubSimulationBackend
from autonomy_demo.visualization.service import NullVisualizationService
from autonomy_demo.visualization.pygame_lidar_view import PygameLidarVisualizationService




@dataclass(slots=True)
class ScenarioRunResult:
    replay_path: Path | None
    evaluation_path: Path
    metadata_path: Path


class ScenarioRunner:
    def __init__(self, runtime_config, sensor_config: dict, output_dir: Path) -> None:
        self.runtime_config = runtime_config
        self.sensor_config = sensor_config
        self.output_dir = ensure_directory(output_dir)
        self.logger = get_logger(__name__)

    def _build_runtime_components(self):
        if self.runtime_config.backend == "carla":
            backend = CarlaSimulationBackend(self.runtime_config)
            sensors = CarlaSensorSuite(self.sensor_config, backend)
            return backend, sensors
        backend = StubSimulationBackend(self.runtime_config)
        sensors = SensorManager(self.sensor_config)
        return backend, sensors

    def _serialize_metadata(self, value):
        if is_dataclass(value):
            return {key: self._serialize_metadata(val) for key, val in asdict(value).items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._serialize_metadata(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_metadata(item) for item in value]
        return value

    def _write_run_metadata(
        self,
        *,
        scenario,
        max_ticks: int,
        replay_path: Path | None,
        evaluation_path: Path,
        motion_planner,
        evaluation,
    ) -> Path:
        route_plan = getattr(motion_planner, "route_plan", None)
        metadata = {
            "scenario_id": scenario.scenario_id,
            "backend": self.runtime_config.backend,
            "map_name": scenario.map_name,
            "tick_hz": self.runtime_config.tick_hz,
            "planned_max_ticks": max_ticks,
            "executed_ticks": int(getattr(evaluation, "tick_count", 0)),
            "route_plan": self._serialize_metadata(route_plan) if route_plan is not None else None,
            "artifacts": {
                "replay": str(replay_path) if replay_path else None,
                "evaluation": str(evaluation_path),
            },
        }
        metadata_path = self.output_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata_path

    def run(
        self,
        scenario,
        visualize: bool,
        record: bool,
        *,
        lidar_view: bool = False,
    ) -> ScenarioRunResult:
        backend, sensors = self._build_runtime_components()
        max_ticks = self.runtime_config.max_ticks
        if max_ticks <= 0:
            max_ticks = max(1, math.ceil(scenario.max_duration_s * self.runtime_config.tick_hz))
        self.logger.info(
            "Running scenario %s for %s ticks at %s Hz",
            scenario.scenario_id,
            max_ticks,
            self.runtime_config.tick_hz,
        )
        bus = InProcessEventBus()
        context = RuntimeContext(
            event_bus=bus,
            record_replay=record,
            enable_visualization=visualize,
            output_dir=self.output_dir,
            latency_budget_ms=self.runtime_config.latency_budget_ms,
        )
        replay_writer = Hdf5OrJsonReplayWriter(self.output_dir) if record else None
        visualization_sinks = []
        if visualize:
            from autonomy_demo.visualization.websocket_bridge import WebSocketBridge
            from autonomy_demo.visualization.server import start_server_thread

            bridge = WebSocketBridge()
            start_server_thread(
                bridge,
                host=self.runtime_config.ws_host,
                port=self.runtime_config.ws_port,
            )
            visualization_sinks.append(bridge)
            visualization_sinks.append(NullVisualizationService(enabled=True, output_dir=self.output_dir))
        if lidar_view:
            visualization_sinks.append(PygameLidarVisualizationService(output_dir=self.output_dir))

        if not visualization_sinks:
            visualization = NullVisualizationService(enabled=False, output_dir=self.output_dir)
        elif len(visualization_sinks) == 1:
            visualization = visualization_sinks[0]
        else:
            visualization = CompositeVisualizationSink(visualization_sinks)
        perception = build_perception_module(self.runtime_config)
        lane_graph_provider = LaneGraphProvider()
        localization = build_localization_module(self.runtime_config, lane_graph_provider)
        mapping = build_mapping_module(self.runtime_config, lane_graph_provider)
        prediction = build_prediction_module(self.runtime_config)
        if self.runtime_config.backend == "carla":
            tuning = getattr(self.runtime_config, "tuning", {}) or {}
            behavior_tuning = tuning.get("behavior", {})
            planning_tuning = tuning.get("planning", {})
            control_tuning = tuning.get("control", {})
            behavior_planner = RuleBasedBehaviorPlanner(**{
                k: v for k, v in behavior_tuning.items()
                if k in RuleBasedBehaviorPlanner.__init__.__code__.co_varnames
            }) if behavior_tuning else RuleBasedBehaviorPlanner()
            planning_kwargs = {
                k: v for k, v in planning_tuning.items()
                if k in FrenetMotionPlanner.__init__.__code__.co_varnames
            } if planning_tuning else {}
            planning_kwargs["lane_graph_provider"] = lane_graph_provider
            motion_planner = FrenetMotionPlanner(**planning_kwargs)
            controller = RouteFollowerController(**{
                k: v for k, v in control_tuning.items()
                if k in RouteFollowerController.__init__.__code__.co_varnames
            }) if control_tuning else RouteFollowerController()
            evaluation = LiveEvaluationHarness(
                scenario,
                tick_hz=self.runtime_config.tick_hz,
                backend=backend,
            )
        else:
            behavior_planner = StubBehaviorPlanner()
            motion_planner = StubMotionPlanner()
            controller = StubController()
            evaluation = SimpleEvaluationHarness(scenario)
        pipeline = PipelineRuntime(
            context=context,
            simulation=backend,
            sensors=sensors,
            perception=perception,
            localization=localization,
            mapping=mapping,
            prediction=prediction,
            behavior_planner=behavior_planner,
            motion_planner=motion_planner,
            controller=controller,
            replay_writer=replay_writer,
            visualization=visualization,
            evaluation=evaluation,
        )
        pipeline.run(scenario=scenario, max_ticks=max_ticks)
        replay_path = replay_writer.finalize() if replay_writer else None
        evaluation_path = evaluation.write_summary(self.output_dir)
        metadata_path = self._write_run_metadata(
            scenario=scenario,
            max_ticks=max_ticks,
            replay_path=replay_path,
            evaluation_path=evaluation_path,
            motion_planner=motion_planner,
            evaluation=evaluation,
        )
        return ScenarioRunResult(
            replay_path=replay_path,
            evaluation_path=evaluation_path,
            metadata_path=metadata_path,
        )
