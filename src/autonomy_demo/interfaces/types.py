from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from autonomy_demo.interfaces.enums import (
    BehaviorState,
    LaneLineType,
    ObjectClass,
    SensorStatus,
    TrackState,
    TrafficLightState,
)

FloatArray: TypeAlias = NDArray[np.float32]


def _float_array(values: Any, shape: tuple[int, ...] | None = None) -> FloatArray:
    array = np.asarray(values, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return array


@dataclass(slots=True)
class CameraFrame:
    sensor_id: str
    frame: FloatArray
    timestamp_s: float
    frame_id: int | None = None
    status: SensorStatus = SensorStatus.OK

    def __post_init__(self) -> None:
        if self.frame.ndim != 3:
            raise ValueError("camera frame must be HxWxC")


@dataclass(slots=True)
class LidarFrame:
    points_xyz: FloatArray
    timestamp_s: float
    frame_id: int | None = None
    intensity: FloatArray | None = None

    def __post_init__(self) -> None:
        self.points_xyz = _float_array(self.points_xyz)
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError("lidar point cloud must be Nx3")
        if self.intensity is not None:
            self.intensity = _float_array(self.intensity)


@dataclass(slots=True)
class RadarFrame:
    detections: FloatArray
    timestamp_s: float
    frame_id: int | None = None

    def __post_init__(self) -> None:
        self.detections = _float_array(self.detections)


@dataclass(slots=True)
class GnssReading:
    world_xyz: FloatArray
    timestamp_s: float
    frame_id: int | None = None

    def __post_init__(self) -> None:
        self.world_xyz = _float_array(self.world_xyz, (3,))


@dataclass(slots=True)
class ImuReading:
    acceleration_xyz: FloatArray
    gyro_xyz: FloatArray
    timestamp_s: float
    frame_id: int | None = None

    def __post_init__(self) -> None:
        self.acceleration_xyz = _float_array(self.acceleration_xyz, (3,))
        self.gyro_xyz = _float_array(self.gyro_xyz, (3,))


@dataclass(slots=True)
class SensorFrameBundle:
    tick_id: int
    sim_time_s: float
    front_camera: CameraFrame
    rear_camera: CameraFrame
    left_camera: CameraFrame
    right_camera: CameraFrame
    lidar: LidarFrame
    radar: RadarFrame
    gnss: GnssReading
    imu: ImuReading
    semantic_camera: CameraFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectDetection:
    track_id: int
    object_class: ObjectClass | str
    world_bbox_3d: FloatArray
    velocity: FloatArray
    confidence: float
    track_state: TrackState
    image_bbox_xyxy: FloatArray | None = None

    def __post_init__(self) -> None:
        self.object_class = ObjectClass(self.object_class)
        self.world_bbox_3d = _float_array(self.world_bbox_3d, (8, 3))
        self.velocity = _float_array(self.velocity, (3,))
        if self.image_bbox_xyxy is not None:
            self.image_bbox_xyxy = _float_array(self.image_bbox_xyxy, (4,))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(slots=True)
class ConeDetection:
    world_xyz: FloatArray
    confidence: float

    def __post_init__(self) -> None:
        self.world_xyz = _float_array(self.world_xyz, (3,))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(slots=True)
class LaneLine:
    lane_id: str
    polyline_image: FloatArray
    polyline_world: FloatArray
    line_type: LaneLineType
    confidence: float

    def __post_init__(self) -> None:
        self.polyline_image = _float_array(self.polyline_image)
        self.polyline_world = _float_array(self.polyline_world)
        if self.polyline_image.ndim != 2 or self.polyline_image.shape[1] != 2:
            raise ValueError("polyline_image must be Nx2")
        if self.polyline_world.ndim != 2 or self.polyline_world.shape[1] != 3:
            raise ValueError("polyline_world must be Nx3")


@dataclass(slots=True)
class TrafficLightDetection:
    world_xyz: FloatArray
    state: TrafficLightState
    stop_line_distance_m: float
    confidence: float
    image_bbox_xyxy: FloatArray | None = None

    def __post_init__(self) -> None:
        self.world_xyz = _float_array(self.world_xyz, (3,))
        if self.image_bbox_xyxy is not None:
            self.image_bbox_xyxy = _float_array(self.image_bbox_xyxy, (4,))


@dataclass(slots=True)
class DrivableSpaceMask:
    mask: NDArray[np.bool_]
    class_probabilities: FloatArray
    source_sensor_id: str

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=np.bool_)
        self.class_probabilities = _float_array(self.class_probabilities)
        if self.mask.ndim != 2:
            raise ValueError("mask must be HxW")


@dataclass(slots=True)
class EgoPose:
    world_xyz: FloatArray
    yaw_rad: float
    speed_mps: float
    acceleration_mps2: float
    current_lane_id: str
    frenet_s: float
    frenet_d: float
    heading_error_rad: float

    def __post_init__(self) -> None:
        self.world_xyz = _float_array(self.world_xyz, (3,))


@dataclass(slots=True)
class StaticLaneSegment:
    lane_id: str
    centerline_world: FloatArray
    speed_limit_mps: float
    left_boundary_world: FloatArray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    right_boundary_world: FloatArray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    predecessor_lane_ids: list[str] = field(default_factory=list)
    successor_lane_ids: list[str] = field(default_factory=list)
    is_junction: bool = False

    def __post_init__(self) -> None:
        self.centerline_world = _float_array(self.centerline_world)
        self.left_boundary_world = _float_array(self.left_boundary_world)
        self.right_boundary_world = _float_array(self.right_boundary_world)


@dataclass(slots=True)
class Waypoint:
    x: float
    y: float
    yaw: float
    velocity: float
    timestamp: float


@dataclass(slots=True)
class RouteWaypoint:
    x: float
    y: float
    z: float
    yaw: float
    cumulative_distance_m: float
    target_speed_mps: float


@dataclass(slots=True)
class RoutePlan:
    waypoints: list[RouteWaypoint]
    goal_xyz: FloatArray
    total_distance_m: float
    goal_tolerance_m: float = 5.0

    def __post_init__(self) -> None:
        self.goal_xyz = _float_array(self.goal_xyz, (3,))
        if self.total_distance_m < 0.0:
            raise ValueError("total_distance_m must be non-negative")


@dataclass(slots=True)
class AgentPrediction:
    track_id: int
    object_class: ObjectClass
    predicted_trajectory: list[Waypoint]
    confidence_by_step: list[float]
    covariance_by_step: list[FloatArray] | None = None


@dataclass(slots=True)
class LocalMap:
    static_lanes: list[StaticLaneSegment]
    dynamic_agents: list[ObjectDetection]
    cone_instances: list[ConeDetection]
    temporary_boundaries: list[LaneLine]
    closed_lanes: list[str]
    traffic_signal_states: list[TrafficLightDetection]
    drivable_space: DrivableSpaceMask | None = None


@dataclass(slots=True)
class EgoTrajectory:
    waypoints: list[Waypoint]
    cost: float
    behavior_state: BehaviorState


@dataclass(slots=True)
class ControlCommand:
    throttle: float
    steer: float
    brake: float
    hand_brake: bool = False
    reverse: bool = False
    emergency_override: bool = False


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0
    z: float = 0.0


@dataclass(slots=True)
class Point2D:
    x: float
    y: float
    z: float = 0.0


@dataclass(slots=True)
class ScenarioNpcConfig:
    model: str
    behavior: str
    spawn: Pose2D
    route: list[Point2D] = field(default_factory=list)


@dataclass(slots=True)
class ScenarioPropConfig:
    type: str
    x: float
    y: float
    z: float = 0.0


@dataclass(slots=True)
class ScenarioTrigger:
    type: str
    at_s: float | None = None
    lane_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioEvalCriteria:
    min_completion_rate: float
    max_collisions: int
    max_red_light_violations: int = 0
    min_pedestrian_clearance_m: float = 0.0


@dataclass(slots=True)
class ScenarioConfig:
    scenario_id: str
    name: str
    map_name: str
    ego_spawn: Pose2D
    ego_goal: Point2D
    max_duration_s: float
    npcs: list[ScenarioNpcConfig]
    props: list[ScenarioPropConfig]
    triggers: list[ScenarioTrigger]
    eval: ScenarioEvalCriteria


@dataclass(slots=True)
class ReplayFrame:
    tick_id: int
    sim_time_s: float
    topic_payloads: dict[str, Any]


@dataclass(slots=True)
class EvaluationSummary:
    scenario_id: str
    success: bool
    completion_rate: float
    collision_count: int
    red_light_violations: int
    pedestrian_clearance_min_m: float
    latency_ms: dict[str, float]
    distance_traveled_m: float = 0.0
    goal_reached: bool = False
    sim_duration_s: float = 0.0
    mean_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeConfig:
    backend: str
    tick_hz: int
    max_ticks: int
    output_dir: Path
    log_level: str
    record_replay: bool
    enable_visualization: bool
    weather_preset: str
    carla_host: str
    carla_port: int
    carla_timeout_s: float
    carla_sync_fps: int
    carla_root: Path
    carla_python_api_wheel: Path
    carla_launch_executable: Path
    town: str
    ego_vehicle_blueprint: str
    perception_mode: str
    perception_device: str
    perception_model_variant: str
    latency_budget_ms: dict[str, float]
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
