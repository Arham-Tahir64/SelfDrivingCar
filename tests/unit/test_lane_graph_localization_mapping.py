from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import LaneLineType, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    CameraFrame,
    ConeDetection,
    DrivableSpaceMask,
    EgoPose,
    GnssReading,
    ImuReading,
    LaneLine,
    LidarFrame,
    LocalMap,
    ObjectDetection,
    RadarFrame,
    SensorFrameBundle,
    StaticLaneSegment,
    TrafficLightDetection,
)
from autonomy_demo.localization.module import MapAwareLocalizationModule
from autonomy_demo.mapping.lane_graph import FrenetProjection, LaneGraph, LaneGraphProvider, project_point_to_centerline
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
        cones=[ConeDetection(world_xyz=np.array([4.0, 0.5, 0.0], dtype=np.float32), confidence=1.0)],
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
    assert local_map.temporary_boundaries
    assert local_map.closed_lanes == ["road_1:section_0:lane_1"]
    assert local_map.traffic_signal_states


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
