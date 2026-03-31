from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import LaneLineType, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    CameraFrame,
    DrivableSpaceMask,
    EgoPose,
    GnssReading,
    ImuReading,
    LaneLine,
    LidarFrame,
    LocalMap,
    ObjectDetection,
    RadarFrame,
    ScenarioConfig,
    ScenarioEvalCriteria,
    ScenarioTrigger,
    SensorFrameBundle,
    StaticLaneSegment,
    TrafficLightDetection,
    Point2D,
    Pose2D,
)
from autonomy_demo.localization.module import MapAwareLocalizationModule
from autonomy_demo.mapping.lane_graph import (
    FrenetProjection,
    LaneGraph,
    LaneGraphProvider,
    build_lane_graph_from_world,
    project_point_to_centerline,
)
from autonomy_demo.mapping.module import MapAwareMappingModule
from autonomy_demo.prediction.module import LaneAwarePredictionModule


def _provider_with_single_lane() -> LaneGraphProvider:
    provider = LaneGraphProvider()
    provider.lane_graph = LaneGraph(
        segments={
            "road_1:section_0:lane_1": StaticLaneSegment(
                lane_id="road_1:section_0:lane_1",
                centerline_world=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
                left_boundary_world=np.array([[0.0, 1.75, 0.0], [10.0, 1.75, 0.0], [20.0, 1.75, 0.0]], dtype=np.float32),
                right_boundary_world=np.array([[0.0, -1.75, 0.0], [10.0, -1.75, 0.0], [20.0, -1.75, 0.0]], dtype=np.float32),
                speed_limit_mps=20.0,
                predecessor_lane_ids=[],
                successor_lane_ids=["road_1:section_0:lane_2"],
                is_junction=False,
            )
        }
    )
    return provider


def _bundle(world_xyz: np.ndarray, *, yaw_rad: float = 0.0) -> SensorFrameBundle:
    image = np.zeros((16, 16, 3), dtype=np.float32)
    return SensorFrameBundle(
        tick_id=0,
        sim_time_s=0.0,
        front_camera=CameraFrame("front_camera", image, 0.0, frame_id=0),
        rear_camera=CameraFrame("rear_camera", image, 0.0, frame_id=0),
        left_camera=CameraFrame("left_camera", image, 0.0, frame_id=0),
        right_camera=CameraFrame("right_camera", image, 0.0, frame_id=0),
        lidar=LidarFrame(points_xyz=np.zeros((1, 3), dtype=np.float32), timestamp_s=0.0, frame_id=0),
        radar=RadarFrame(detections=np.zeros((1, 4), dtype=np.float32), timestamp_s=0.0, frame_id=0),
        gnss=GnssReading(world_xyz=world_xyz, timestamp_s=0.0, frame_id=0),
        imu=ImuReading(
            acceleration_xyz=np.zeros(3, dtype=np.float32),
            gyro_xyz=np.zeros(3, dtype=np.float32),
            timestamp_s=0.0,
            frame_id=0,
        ),
        metadata={
            "ego_yaw_rad": yaw_rad,
            "ego_speed_mps": 8.0,
            "ego_acceleration_mps2": 0.2,
        },
    )


class _FakeLaneType:
    Driving = "Driving"


class _FakeCarlaModule:
    LaneType = _FakeLaneType


class _FakeLaneLocation:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeLaneRotation:
    def __init__(self, yaw: float) -> None:
        self.yaw = yaw


class _FakeLaneTransform:
    def __init__(self, x: float, y: float, z: float, yaw: float) -> None:
        self.location = _FakeLaneLocation(x, y, z)
        self.rotation = _FakeLaneRotation(yaw)


class _FakeLaneWaypoint:
    def __init__(
        self,
        *,
        x: float,
        y: float,
        z: float = 0.0,
        yaw: float = 0.0,
        s: float = 0.0,
        road_id: int = 1,
        section_id: int = 0,
        lane_id: int = 1,
    ) -> None:
        self.transform = _FakeLaneTransform(x, y, z, yaw)
        self.s = s
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.lane_type = _FakeLaneType.Driving
        self.lane_width = 3.5
        self.is_junction = False

    def previous(self, distance_m: float):
        del distance_m
        return []

    def next(self, distance_m: float):
        del distance_m
        return []


class _FakeLaneMap:
    def __init__(self, waypoints: list[_FakeLaneWaypoint]) -> None:
        self._waypoints = list(waypoints)

    def generate_waypoints(self, step_m: float):
        del step_m
        return list(self._waypoints)


class _FakeLaneWorld:
    def __init__(self, waypoints: list[_FakeLaneWaypoint]) -> None:
        self._map = _FakeLaneMap(waypoints)

    def get_map(self) -> _FakeLaneMap:
        return self._map


def test_frenet_projection_on_simple_centerline() -> None:
    projection: FrenetProjection = project_point_to_centerline(
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        np.array([4.0, 2.0, 0.0], dtype=np.float32),
    )
    assert abs(projection.s - 4.0) < 1e-3
    assert abs(projection.d - 2.0) < 1e-3
    assert abs(projection.heading_rad) < 1e-6


def test_map_aware_localization_populates_lane_and_frenet() -> None:
    provider = _provider_with_single_lane()
    module = MapAwareLocalizationModule(provider)
    ego_pose = module.run(_bundle(np.array([6.0, 1.0, 0.0], dtype=np.float32)))
    assert ego_pose.current_lane_id == "road_1:section_0:lane_1"
    assert ego_pose.frenet_s > 5.0
    assert ego_pose.frenet_d > 0.0
    assert abs(ego_pose.heading_error_rad) < 1e-6


def test_lane_graph_prefers_same_height_lane_for_nearest_projection() -> None:
    lane_graph = LaneGraph(
        segments={
            "highway": StaticLaneSegment(
                lane_id="highway",
                centerline_world=np.array([[0.0, 0.0, 10.0], [20.0, 0.0, 10.0]], dtype=np.float32),
                speed_limit_mps=20.0,
            ),
            "underpass": StaticLaneSegment(
                lane_id="underpass",
                centerline_world=np.array([[0.0, 0.4, 0.0], [20.0, 0.4, 0.0]], dtype=np.float32),
                speed_limit_mps=20.0,
            ),
        }
    )
    projection = lane_graph.nearest_projection(np.array([5.0, 0.3, 10.2], dtype=np.float32))
    assert projection is not None
    assert projection.lane_id == "highway"


def test_build_lane_graph_reverses_waypoint_order_when_s_runs_against_lane_heading() -> None:
    world = _FakeLaneWorld(
        [
            _FakeLaneWaypoint(x=10.0, y=0.0, yaw=0.0, s=0.0),
            _FakeLaneWaypoint(x=5.0, y=0.0, yaw=0.0, s=5.0),
            _FakeLaneWaypoint(x=0.0, y=0.0, yaw=0.0, s=10.0),
        ]
    )

    lane_graph = build_lane_graph_from_world(world, _FakeCarlaModule(), step_m=5.0)
    segment = lane_graph.segments["road_1:section_0:lane_1"]

    assert segment.centerline_world[:, 0].tolist() == [0.0, 5.0, 10.0]


def test_lane_graph_nearby_lanes_filters_out_stacked_roads() -> None:
    lane_graph = LaneGraph(
        segments={
            "highway_main": StaticLaneSegment(
                lane_id="highway_main",
                centerline_world=np.array([[0.0, 0.0, 12.0], [20.0, 0.0, 12.0]], dtype=np.float32),
                speed_limit_mps=20.0,
            ),
            "highway_adjacent": StaticLaneSegment(
                lane_id="highway_adjacent",
                centerline_world=np.array([[0.0, 3.5, 12.1], [20.0, 3.5, 12.1]], dtype=np.float32),
                speed_limit_mps=20.0,
            ),
            "lower_road": StaticLaneSegment(
                lane_id="lower_road",
                centerline_world=np.array([[0.0, 0.1, 0.5], [20.0, 0.1, 0.5]], dtype=np.float32),
                speed_limit_mps=20.0,
            ),
        }
    )
    nearby = lane_graph.nearby_lanes(np.array([5.0, 0.2, 12.0], dtype=np.float32), radius_m=10.0, limit=8)
    lane_ids = {lane.lane_id for lane in nearby}
    assert "highway_main" in lane_ids
    assert "highway_adjacent" in lane_ids
    assert "lower_road" not in lane_ids


def test_mapping_module_returns_nearby_lane_graph_segments() -> None:
    provider = _provider_with_single_lane()
    mapping = MapAwareMappingModule(provider, lane_horizon_radius_m=50.0, lane_limit=8)
    ego_pose = EgoPose(
        world_xyz=np.array([5.0, 0.5, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=10.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_1",
        frenet_s=5.0,
        frenet_d=0.5,
        heading_error_rad=0.0,
    )
    local_map = mapping.run(
        detections=[],
        lanes=[
            LaneLine(
                lane_id="temporary_boundary",
                polyline_image=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                polyline_world=np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32),
                line_type=LaneLineType.TEMPORARY,
                confidence=0.9,
            )
        ],
        drivable_space=DrivableSpaceMask(
            mask=np.ones((8, 8), dtype=bool),
            class_probabilities=np.ones((8, 8), dtype=np.float32),
            source_sensor_id="front_camera",
        ),
        cones=[],
        traffic_lights=[
            TrafficLightDetection(
                world_xyz=np.array([20.0, 3.0, 3.0], dtype=np.float32),
                state=TrafficLightState.GREEN,
                stop_line_distance_m=15.0,
                confidence=1.0,
            )
        ],
        ego_pose=ego_pose,
    )
    assert local_map.static_lanes
    assert local_map.static_lanes[0].lane_id == "road_1:section_0:lane_1"
    assert local_map.perceived_lanes
    assert local_map.perceived_lanes[0].source_modality == "camera"
    assert local_map.temporary_boundaries
    assert local_map.closed_lanes == []
    assert local_map.traffic_signal_states


def test_mapping_module_ignores_cones_for_lane_closure() -> None:
    provider = _provider_with_single_lane()
    mapping = MapAwareMappingModule(provider, lane_horizon_radius_m=50.0, lane_limit=8)
    ego_pose = EgoPose(
        world_xyz=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_1",
        frenet_s=5.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    local_map = mapping.run(
        detections=[],
        lanes=[],
        drivable_space=DrivableSpaceMask(
            mask=np.ones((8, 8), dtype=bool),
            class_probabilities=np.ones((8, 8), dtype=np.float32),
            source_sensor_id="front_camera",
        ),
        cones=[],
        traffic_lights=[],
        ego_pose=ego_pose,
    )
    assert local_map.closed_lanes == []


def test_mapping_module_publishes_perceived_lanes_without_lane_graph() -> None:
    mapping = MapAwareMappingModule(lane_graph_provider=None)
    ego_pose = EgoPose(
        world_xyz=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="lane_001",
        frenet_s=5.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    perceived_lane = LaneLine(
        lane_id="lane_left",
        polyline_image=np.array([[10.0, 10.0], [15.0, 20.0]], dtype=np.float32),
        polyline_world=np.array([[5.0, -1.5, 0.0], [15.0, -1.2, 0.0]], dtype=np.float32),
        line_type=LaneLineType.SOLID,
        confidence=0.8,
        source_modality="camera",
        source_sensor_ids=["front_camera"],
        position_estimate_kind="camera_projection",
    )
    local_map = mapping.run(
        detections=[],
        lanes=[perceived_lane],
        drivable_space=DrivableSpaceMask(
            mask=np.ones((8, 8), dtype=bool),
            class_probabilities=np.ones((8, 8), dtype=np.float32),
            source_sensor_id="front_camera",
        ),
        cones=[],
        traffic_lights=[],
        ego_pose=ego_pose,
    )
    assert local_map.perceived_lanes == [perceived_lane]


def test_lane_aware_prediction_follows_lane_centerline() -> None:
    predictor = LaneAwarePredictionModule()
    agent = ObjectDetection(
        track_id=1,
        object_class=ObjectClass.VEHICLE,
        world_bbox_3d=np.array(
            [
                [4.0, -0.5, 0.0],
                [6.0, -0.5, 0.0],
                [6.0, 0.5, 0.0],
                [4.0, 0.5, 0.0],
                [4.0, -0.5, 1.5],
                [6.0, -0.5, 1.5],
                [6.0, 0.5, 1.5],
                [4.0, 0.5, 1.5],
            ],
            dtype=np.float32,
        ),
        velocity=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.95,
        track_state=TrackState.CONFIRMED,
    )
    local_map = LocalMap(
        static_lanes=_provider_with_single_lane().lane_graph.nearby_lanes(np.array([5.0, 0.0, 0.0], dtype=np.float32)),
        dynamic_agents=[agent],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    predictions = predictor.run(local_map)
    assert len(predictions) == 1
    assert predictions[0].predicted_trajectory[1].x > predictions[0].predicted_trajectory[0].x
    assert abs(predictions[0].predicted_trajectory[1].y) < 1e-3


def test_mapping_module_activates_merge_trigger_into_closed_lane_cue() -> None:
    provider = _provider_with_single_lane()
    provider.lane_graph.segments["road_1:section_0:lane_2"] = StaticLaneSegment(
        lane_id="road_1:section_0:lane_2",
        centerline_world=np.array([[0.0, 3.5, 0.0], [10.0, 3.5, 0.0], [20.0, 3.5, 0.0]], dtype=np.float32),
        left_boundary_world=np.array([[0.0, 5.25, 0.0], [10.0, 5.25, 0.0], [20.0, 5.25, 0.0]], dtype=np.float32),
        right_boundary_world=np.array([[0.0, 1.75, 0.0], [10.0, 1.75, 0.0], [20.0, 1.75, 0.0]], dtype=np.float32),
        speed_limit_mps=20.0,
    )
    mapping = MapAwareMappingModule(provider, lane_horizon_radius_m=50.0, lane_limit=8)
    scenario = ScenarioConfig(
        scenario_id="SC-03",
        name="Lane Merge",
        map_name="Town04",
        ego_spawn=Pose2D(x=0.0, y=0.0, z=0.0, yaw=0.0),
        ego_goal=Point2D(x=30.0, y=3.5, z=0.0),
        max_duration_s=20.0,
        npcs=[],
        props=[],
        triggers=[ScenarioTrigger(type="merge_required", at_s=5.0, lane_id=1)],
        eval=ScenarioEvalCriteria(min_completion_rate=0.8, max_collisions=0),
    )
    mapping.prepare(simulation=None, scenario=scenario)
    ego_pose = EgoPose(
        world_xyz=np.array([7.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_1",
        frenet_s=7.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    local_map = mapping.run(
        detections=[],
        lanes=[],
        drivable_space=DrivableSpaceMask(
            mask=np.ones((4, 4), dtype=bool),
            class_probabilities=np.ones((4, 4), dtype=np.float32),
            source_sensor_id="front_camera",
        ),
        cones=[],
        traffic_lights=[],
        ego_pose=ego_pose,
    )
    assert "road_1:section_0:lane_1" in local_map.closed_lanes
    assert local_map.temporary_boundaries

    merged_pose = EgoPose(
        world_xyz=np.array([12.0, 3.5, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_2",
        frenet_s=12.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    merged_local_map = mapping.run(
        detections=[],
        lanes=[],
        drivable_space=DrivableSpaceMask(
            mask=np.ones((4, 4), dtype=bool),
            class_probabilities=np.ones((4, 4), dtype=np.float32),
            source_sensor_id="front_camera",
        ),
        cones=[],
        traffic_lights=[],
        ego_pose=merged_pose,
    )
    assert "road_1:section_0:lane_1" in merged_local_map.closed_lanes
    assert "road_1:section_0:lane_2" not in merged_local_map.closed_lanes
    assert merged_local_map.temporary_boundaries == []
