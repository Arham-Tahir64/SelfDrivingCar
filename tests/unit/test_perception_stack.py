from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import TrackState
from autonomy_demo.interfaces.types import CameraFrame, GnssReading, ImuReading, LidarFrame, RadarFrame, SensorFrameBundle
from autonomy_demo.mapping.module import StubMappingModule
from autonomy_demo.perception.module import PerceptionStack, build_perception_module


def _bundle() -> SensorFrameBundle:
    image = np.zeros((120, 200, 3), dtype=np.float32)
    image[70:110, 85:135, :] = 220.0
    metadata = {
        "synthetic": True,
        "carla_actor_annotations": [
            {
                "track_id": 10,
                "object_class": "vehicle",
                "confidence": 0.98,
                "image_bbox_xyxy": [85.0, 70.0, 135.0, 110.0],
                "world_bbox_3d": [
                    [8.0, -1.0, 0.0],
                    [12.0, -1.0, 0.0],
                    [12.0, 1.0, 0.0],
                    [8.0, 1.0, 0.0],
                    [8.0, -1.0, 1.5],
                    [12.0, -1.0, 1.5],
                    [12.0, 1.0, 1.5],
                    [8.0, 1.0, 1.5],
                ],
                "velocity_xyz": [4.0, 0.0, 0.0],
                "world_xyz": [10.0, 0.0, 0.75],
            },
            {
                "track_id": 21,
                "object_class": "traffic_light",
                "confidence": 1.0,
                "image_bbox_xyxy": [140.0, 15.0, 155.0, 55.0],
                "world_bbox_3d": [
                    [18.0, 4.0, 0.0],
                    [18.5, 4.0, 0.0],
                    [18.5, 4.5, 0.0],
                    [18.0, 4.5, 0.0],
                    [18.0, 4.0, 3.0],
                    [18.5, 4.0, 3.0],
                    [18.5, 4.5, 3.0],
                    [18.0, 4.5, 3.0],
                ],
                "velocity_xyz": [0.0, 0.0, 0.0],
                "world_xyz": [18.2, 4.2, 3.0],
                "traffic_light_state": "GREEN",
            },
        ],
    }
    return SensorFrameBundle(
        tick_id=0,
        sim_time_s=0.0,
        front_camera=CameraFrame("front_camera", image, 0.0, frame_id=0),
        rear_camera=CameraFrame("rear_camera", image, 0.0, frame_id=0),
        left_camera=CameraFrame("left_camera", image, 0.0, frame_id=0),
        right_camera=CameraFrame("right_camera", image, 0.0, frame_id=0),
        lidar=LidarFrame(points_xyz=np.zeros((4, 3), dtype=np.float32), timestamp_s=0.0, frame_id=0),
        radar=RadarFrame(detections=np.zeros((1, 4), dtype=np.float32), timestamp_s=0.0, frame_id=0),
        gnss=GnssReading(world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32), timestamp_s=0.0, frame_id=0),
        imu=ImuReading(
            acceleration_xyz=np.zeros(3, dtype=np.float32),
            gyro_xyz=np.zeros(3, dtype=np.float32),
            timestamp_s=0.0,
            frame_id=0,
        ),
        metadata=metadata,
    )


def test_build_perception_module_respects_runtime_mode() -> None:
    runtime = type(
        "Runtime",
        (),
        {"perception_mode": "camera_v1", "perception_device": "cpu", "perception_model_variant": "bootstrap"},
    )()
    module = build_perception_module(runtime)
    assert isinstance(module, PerceptionStack)


def test_perception_stack_converts_bootstrap_annotations() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    detections, lanes, drivable, traffic_lights, cones = module.run(_bundle())
    assert len(detections) == 1
    assert detections[0].track_id == 10
    assert detections[0].track_state == TrackState.TENTATIVE
    assert detections[0].image_bbox_xyxy is not None
    assert lanes
    assert drivable.mask.shape == (120, 200)
    assert len(traffic_lights) == 1
    assert cones == []


def test_tracker_confirms_persistent_tracks() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    first_bundle = _bundle()
    second_bundle = _bundle()
    first_detections, _, _, _, _ = module.run(first_bundle)
    second_detections, _, _, _, _ = module.run(second_bundle)
    assert first_detections[0].track_state == TrackState.TENTATIVE
    assert second_detections[0].track_state == TrackState.CONFIRMED


def test_perception_stack_degrades_without_crashing() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    bundle = _bundle()
    module.lane_extractor.extract = lambda frame: (_ for _ in ()).throw(RuntimeError("lane fail"))  # type: ignore[method-assign]
    detections, lanes, drivable, traffic_lights, cones = module.run(bundle)
    assert detections == []
    assert lanes == []
    assert traffic_lights == []
    assert cones == []
    assert bundle.metadata["perception_status"] == "degraded"
    assert not bool(drivable.mask.any())


def test_mapping_consumes_perception_outputs() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    detections, lanes, drivable, traffic_lights, cones = module.run(_bundle())
    ego_pose = type(
        "Pose",
        (),
        {
            "current_lane_id": "lane_001",
        },
    )()
    local_map = StubMappingModule().run(detections, lanes, drivable, cones, traffic_lights, ego_pose)
    assert local_map.dynamic_agents == detections
    assert local_map.static_lanes
