from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from autonomy_demo.interfaces.types import (
    AgentPrediction,
    ConeDetection,
    ControlCommand,
    DrivableSpaceMask,
    EgoPose,
    EgoTrajectory,
    EvaluationSummary,
    LaneLine,
    LocalMap,
    ObjectDetection,
    ReplayFrame,
    ScenarioConfig,
    SensorFrameBundle,
    TrafficLightDetection,
)


class Subscriber(Protocol):
    def __call__(self, topic: str, payload: Any) -> None:
        ...


class EventBus(Protocol):
    def publish(self, topic: str, payload: Any) -> None:
        ...

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...


class SimulationBackend(Protocol):
    def bootstrap(self, scenario: ScenarioConfig) -> None:
        ...

    def attach_sensors(self) -> None:
        ...

    def tick(self, tick_id: int) -> None:
        ...

    def apply_control(self, command: ControlCommand) -> None:
        ...

    def shutdown(self) -> None:
        ...


class SensorSuite(Protocol):
    def setup(self) -> None:
        ...

    def warmup(self, simulation: SimulationBackend, warmup_ticks: int = 2) -> None:
        ...

    def capture(self, tick_id: int, sim_time_s: float) -> SensorFrameBundle:
        ...


class PerceptionModule(Protocol):
    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        ...


class LocalizationModule(Protocol):
    def run(self, bundle: SensorFrameBundle) -> EgoPose:
        ...


class MappingModule(Protocol):
    def run(
        self,
        detections: list[ObjectDetection],
        lanes: list[LaneLine],
        drivable_space: DrivableSpaceMask,
        cones: list[ConeDetection],
        traffic_lights: list[TrafficLightDetection],
        ego_pose: EgoPose,
    ) -> LocalMap:
        ...


class PredictionModule(Protocol):
    def run(self, local_map: LocalMap) -> list[AgentPrediction]:
        ...


class BehaviorPlanner(Protocol):
    def run(self, local_map: LocalMap, ego_pose: EgoPose) -> Any:
        ...


class MotionPlanner(Protocol):
    def run(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: Any,
    ) -> EgoTrajectory:
        ...


class Controller(Protocol):
    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        ...


class VisualizationSink(Protocol):
    def attach(self, event_bus: EventBus) -> None:
        ...

    def flush(self) -> None:
        ...


class ReplayWriter(Protocol):
    def record(self, frame: ReplayFrame) -> None:
        ...

    def finalize(self) -> Path | None:
        ...


class ReplayReader(Protocol):
    def read(self) -> list[ReplayFrame]:
        ...


class EvaluationHarness(Protocol):
    def update(self, tick_id: int, snapshot: dict[str, Any]) -> None:
        ...

    def finalize(self) -> EvaluationSummary:
        ...


@dataclass(slots=True)
class RuntimeContext:
    event_bus: EventBus
    record_replay: bool
    enable_visualization: bool
    output_dir: Path
    latency_budget_ms: dict[str, float] = field(default_factory=dict)
