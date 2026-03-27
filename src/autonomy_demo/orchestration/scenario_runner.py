from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.control.controller import StubController
from autonomy_demo.eval.harness import SimpleEvaluationHarness
from autonomy_demo.interfaces.contracts import RuntimeContext
from autonomy_demo.localization.module import StubLocalizationModule
from autonomy_demo.mapping.module import StubMappingModule
from autonomy_demo.orchestration.event_bus import InProcessEventBus
from autonomy_demo.orchestration.pipeline_runtime import PipelineRuntime
from autonomy_demo.perception.module import StubPerceptionModule
from autonomy_demo.planning.behavior_fsm import StubBehaviorPlanner
from autonomy_demo.planning.motion_planner import StubMotionPlanner
from autonomy_demo.prediction.module import StubPredictionModule
from autonomy_demo.replay.writer import Hdf5OrJsonReplayWriter
from autonomy_demo.sensors.carla_sensor_suite import CarlaSensorSuite
from autonomy_demo.sensors.sensor_manager import SensorManager
from autonomy_demo.sim.backends import CarlaSimulationBackend, StubSimulationBackend
from autonomy_demo.visualization.service import NullVisualizationService


@dataclass(slots=True)
class ScenarioRunResult:
    replay_path: Path | None
    evaluation_path: Path


class ScenarioRunner:
    def __init__(self, runtime_config, sensor_config: dict, output_dir: Path) -> None:
        self.runtime_config = runtime_config
        self.sensor_config = sensor_config
        self.output_dir = ensure_directory(output_dir)

    def _build_runtime_components(self):
        if self.runtime_config.backend == "carla":
            backend = CarlaSimulationBackend(self.runtime_config)
            sensors = CarlaSensorSuite(self.sensor_config, backend)
            return backend, sensors
        backend = StubSimulationBackend(self.runtime_config)
        sensors = SensorManager(self.sensor_config)
        return backend, sensors

    def run(self, scenario, visualize: bool, record: bool) -> ScenarioRunResult:
        backend, sensors = self._build_runtime_components()
        bus = InProcessEventBus()
        context = RuntimeContext(
            event_bus=bus,
            record_replay=record,
            enable_visualization=visualize,
            output_dir=self.output_dir,
            latency_budget_ms=self.runtime_config.latency_budget_ms,
        )
        replay_writer = Hdf5OrJsonReplayWriter(self.output_dir) if record else None
        visualization = NullVisualizationService(enabled=visualize)
        evaluation = SimpleEvaluationHarness(scenario)
        pipeline = PipelineRuntime(
            context=context,
            simulation=backend,
            sensors=sensors,
            perception=StubPerceptionModule(),
            localization=StubLocalizationModule(),
            mapping=StubMappingModule(),
            prediction=StubPredictionModule(),
            behavior_planner=StubBehaviorPlanner(),
            motion_planner=StubMotionPlanner(),
            controller=StubController(),
            replay_writer=replay_writer,
            visualization=visualization,
            evaluation=evaluation,
        )
        pipeline.run(scenario=scenario, max_ticks=self.runtime_config.max_ticks)
        replay_path = replay_writer.finalize() if replay_writer else None
        evaluation_path = evaluation.write_summary(self.output_dir)
        return ScenarioRunResult(replay_path=replay_path, evaluation_path=evaluation_path)
