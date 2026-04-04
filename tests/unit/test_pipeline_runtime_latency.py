from __future__ import annotations

from pathlib import Path

import numpy as np

from autonomy_demo.interfaces.contracts import RuntimeContext
from autonomy_demo.interfaces.enums import BehaviorState, TrafficLightState, TopicName
from autonomy_demo.interfaces.types import (
    CameraFrame,
    ControlCommand,
    DrivableSpaceMask,
    EgoPose,
    EgoTrajectory,
    GnssReading,
    ImuReading,
    LaneLine,
    LidarFrame,
    LocalMap,
    Point2D,
    Pose2D,
    RadarFrame,
    ScenarioConfig,
    ScenarioEvalCriteria,
    ScenarioNpcConfig,
    ScenarioPropConfig,
    SensorFrameBundle,
    StaticLaneSegment,
    Waypoint,
)
from autonomy_demo.orchestration.event_bus import InProcessEventBus
from autonomy_demo.orchestration.pipeline_runtime import PipelineRuntime


def _bundle(tick_id: int) -> SensorFrameBundle:
    image = np.zeros((64, 96, 3), dtype=np.float32)
    return SensorFrameBundle(
        tick_id=tick_id,
        sim_time_s=tick_id * 0.05,
        front_camera=CameraFrame("front_camera", image, tick_id * 0.05, frame_id=tick_id),
        rear_camera=CameraFrame("rear_camera", image, tick_id * 0.05, frame_id=tick_id),
        left_camera=CameraFrame("left_camera", image, tick_id * 0.05, frame_id=tick_id),
        right_camera=CameraFrame("right_camera", image, tick_id * 0.05, frame_id=tick_id),
        lidar=LidarFrame(points_xyz=np.zeros((4, 3), dtype=np.float32), timestamp_s=tick_id * 0.05, frame_id=tick_id),
        radar=RadarFrame(detections=np.zeros((1, 4), dtype=np.float32), timestamp_s=tick_id * 0.05, frame_id=tick_id),
        gnss=GnssReading(world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32), timestamp_s=tick_id * 0.05, frame_id=tick_id),
        imu=ImuReading(
            acceleration_xyz=np.zeros(3, dtype=np.float32),
            gyro_xyz=np.zeros(3, dtype=np.float32),
            timestamp_s=tick_id * 0.05,
            frame_id=tick_id,
        ),
        metadata={},
    )


class _FakeSimulation:
    def bootstrap(self, scenario: ScenarioConfig) -> None:  # noqa: ARG002
        return None

    def attach_sensors(self) -> None:
        return None

    def tick(self, tick_id: int) -> None:  # noqa: ARG002
        return None

    def apply_control(self, command: ControlCommand) -> None:  # noqa: ARG002
        return None

    def shutdown(self) -> None:
        return None


class _FakeSensors:
    def setup(self) -> None:
        return None

    def warmup(self, simulation, warmup_ticks: int = 2) -> None:  # noqa: ANN001, ARG002
        return None

    def capture(self, tick_id: int, sim_time_s: float) -> SensorFrameBundle:  # noqa: ARG002
        return _bundle(tick_id)


class _FakePerception:
    def __init__(self) -> None:
        self._calls = 0

    def run(self, bundle: SensorFrameBundle):
        self._calls += 1
        bundle.metadata["drivable_inference_ms"] = 42.0 if self._calls == 1 else 0.0
        bundle.metadata["lane_inference_ms"] = 18.0 if self._calls == 1 else 0.0
        bundle.metadata["perception_summary"] = type(
            "PerceptionSummary",
            (),
            {
                "active_mode": "camera_v1",
                "fallback_state": "camera_only",
                "counts_by_modality": {},
                "active_camera_sensors": ["front_camera"],
                "detection_count": 0,
                "traffic_light_count": 0,
            },
        )()
        drivable = DrivableSpaceMask(
            mask=np.zeros((64, 96), dtype=np.bool_),
            class_probabilities=np.zeros((64, 96, 2), dtype=np.float32),
            source_sensor_id="front_camera",
        )
        return [], [], drivable, [], []


class _FakeLocalization:
    def run(self, bundle: SensorFrameBundle) -> EgoPose:  # noqa: ARG002
        return EgoPose(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            yaw_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            current_lane_id="lane_1",
            frenet_s=0.0,
            frenet_d=0.0,
            heading_error_rad=0.0,
        )


class _FakeMapping:
    def run(self, detections, lanes, drivable_space, cones, traffic_lights, ego_pose):  # noqa: ANN001, ARG002
        lane = StaticLaneSegment(
            lane_id="lane_1",
            centerline_world=np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
            left_boundary_world=np.array([[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]], dtype=np.float32),
            right_boundary_world=np.array([[0.0, -1.75, 0.0], [20.0, -1.75, 0.0]], dtype=np.float32),
            speed_limit_mps=10.0,
        )
        return LocalMap(
            static_lanes=[lane],
            dynamic_agents=[],
            cone_instances=[],
            temporary_boundaries=[],
            closed_lanes=[],
            traffic_signal_states=[],
            perceived_lanes=[],
            drivable_space=drivable_space,
        )


class _FakePrediction:
    def prepare(self, simulation, scenario) -> None:  # noqa: ANN001, ARG002
        return None

    def run(self, local_map: LocalMap):  # noqa: ARG002
        return []


class _FakeBehaviorPlanner:
    def run(self, local_map: LocalMap, ego_pose: EgoPose):  # noqa: ARG002
        return BehaviorState.LANE_KEEP


class _FakeMotionPlanner:
    last_candidates: list[object] = []
    route_plan = None

    def run(self, local_map: LocalMap, ego_pose: EgoPose, predictions, behavior_state: BehaviorState):  # noqa: ARG002
        return EgoTrajectory(
            waypoints=[Waypoint(x=1.0, y=0.0, yaw=0.0, velocity=1.0, timestamp=0.0)],
            cost=0.0,
            behavior_state=behavior_state,
        )


class _FakeController:
    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:  # noqa: ARG002
        return ControlCommand(throttle=0.0, steer=0.0, brake=0.0)


class _FakeEvaluation:
    def update(self, tick_id: int, snapshot: dict[str, object]) -> None:  # noqa: ARG002
        return None

    def set_latency(self, latency) -> None:  # noqa: ANN001
        self.latency = latency


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="SC-LAT",
        name="Latency",
        map_name="Town01",
        ego_spawn=Pose2D(x=0.0, y=0.0, yaw=0.0, z=0.0),
        ego_goal=Point2D(x=10.0, y=0.0, z=0.0),
        max_duration_s=1.0,
        npcs=[],
        props=[],
        triggers=[],
        eval=ScenarioEvalCriteria(min_completion_rate=0.0, max_collisions=0),
    )


def test_pipeline_runtime_reports_current_tick_aux_latency_instead_of_stale_values() -> None:
    bus = InProcessEventBus()
    runtime = PipelineRuntime(
        context=RuntimeContext(
            event_bus=bus,
            record_replay=False,
            enable_visualization=False,
            output_dir=Path("outputs"),
            latency_budget_ms={},
        ),
        simulation=_FakeSimulation(),
        sensors=_FakeSensors(),
        perception=_FakePerception(),
        localization=_FakeLocalization(),
        mapping=_FakeMapping(),
        prediction=_FakePrediction(),
        behavior_planner=_FakeBehaviorPlanner(),
        motion_planner=_FakeMotionPlanner(),
        controller=_FakeController(),
        replay_writer=None,
        visualization=None,
        evaluation=_FakeEvaluation(),
    )

    runtime.run(_scenario(), max_ticks=2)

    snapshot = bus.snapshot()
    latency = snapshot[TopicName.PIPELINE_LATENCY.value]
    assert latency["segformer_drivable"] == 0.0
    assert latency["learned_lanes"] == 0.0
    assert latency["perception_aux_total"] == 0.0
