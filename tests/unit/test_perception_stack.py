from __future__ import annotations

import numpy as np
import pytest

from autonomy_demo.interfaces.enums import ObjectClass, SensorStatus, TrackState
from autonomy_demo.interfaces.types import CameraFrame, GnssReading, ImuReading, LidarFrame, ObjectDetection, RadarFrame, SensorFrameBundle
from autonomy_demo.mapping.module import StubMappingModule
from autonomy_demo.perception.fusion import fuse_detections
from autonomy_demo.perception.internal_types import FrameDetection2D
from autonomy_demo.perception.module import (
    FusedPerceptionStack,
    LidarPerceptionStack,
    PerceptionStack,
    _PerceptionCameraBudgetPolicy,
    _resolve_perception_camera_budget_policy,
    build_perception_module,
)
from autonomy_demo.perception.object_detection import CameraInferenceRequest, YoloObjectDetector, bootstrap_annotations_from_metadata
from autonomy_demo.perception.tracking import KalmanSortTracker


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


def _lidar_bundle(*, sim_time_s: float = 0.0, ego_x: float = 0.0) -> SensorFrameBundle:
    image = np.zeros((120, 200, 3), dtype=np.float32)
    image[70:110, 85:135, :] = 220.0
    lidar_points = np.array(
        [
            [10.0, 1.0, 0.1],
            [10.4, 1.1, 0.5],
            [10.8, 1.3, 1.0],
            [11.2, 1.0, 1.4],
            [11.5, 0.8, 0.7],
            [11.7, 1.4, 0.2],
            [6.0, -1.5, 0.1],
            [6.1, -1.6, 0.4],
            [6.2, -1.4, 0.8],
            [0.0, 0.0, -2.2],
        ],
        dtype=np.float32,
    )
    return SensorFrameBundle(
        tick_id=int(sim_time_s * 10),
        sim_time_s=sim_time_s,
        front_camera=CameraFrame("front_camera", image, sim_time_s, frame_id=0),
        rear_camera=CameraFrame("rear_camera", image, sim_time_s, frame_id=0),
        left_camera=CameraFrame("left_camera", image, sim_time_s, frame_id=0),
        right_camera=CameraFrame("right_camera", image, sim_time_s, frame_id=0),
        lidar=LidarFrame(points_xyz=lidar_points, timestamp_s=sim_time_s, frame_id=0),
        radar=RadarFrame(detections=np.zeros((1, 4), dtype=np.float32), timestamp_s=sim_time_s, frame_id=0),
        gnss=GnssReading(world_xyz=np.array([ego_x, 50.0, 0.0], dtype=np.float32), timestamp_s=sim_time_s, frame_id=0),
        imu=ImuReading(
            acceleration_xyz=np.zeros(3, dtype=np.float32),
            gyro_xyz=np.zeros(3, dtype=np.float32),
            timestamp_s=sim_time_s,
            frame_id=0,
        ),
        metadata={"synthetic": True, "ego_yaw_rad": 0.0},
    )


def test_build_perception_module_respects_runtime_mode() -> None:
    runtime = type(
        "Runtime",
        (),
        {"perception_mode": "camera_v1", "perception_device": "cpu", "perception_model_variant": "bootstrap"},
    )()
    module = build_perception_module(runtime)
    assert isinstance(module, PerceptionStack)


def test_build_perception_module_supports_lidar_mode() -> None:
    runtime = type(
        "Runtime",
        (),
        {"perception_mode": "lidar_v1", "perception_device": "cpu", "perception_model_variant": "bootstrap"},
    )()
    module = build_perception_module(runtime)
    assert isinstance(module, LidarPerceptionStack)


def test_build_perception_module_supports_fused_mode() -> None:
    runtime = type(
        "Runtime",
        (),
        {"perception_mode": "fused_v1", "perception_device": "cpu", "perception_model_variant": "bootstrap"},
    )()
    module = build_perception_module(runtime)
    assert isinstance(module, FusedPerceptionStack)


def test_perception_stack_converts_bootstrap_annotations() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    bundle = _bundle()
    detections, lanes, drivable, traffic_lights, cones = module.run(bundle)
    assert len(detections) == 1
    assert detections[0].track_id == 10
    assert detections[0].track_state == TrackState.TENTATIVE
    assert detections[0].image_bbox_xyxy is not None  # bootstrap bboxes now passed through for overlay
    assert detections[0].source_modality == "bootstrap"
    assert detections[0].position_estimate_kind == "truth_fallback"
    assert isinstance(lanes, list)
    assert drivable.mask.shape == (120, 200)
    assert len(traffic_lights) == 1
    assert traffic_lights[0].source_modality == "bootstrap"
    assert cones == []
    assert bundle.metadata["perception_summary"].fallback_state == "bootstrap"


class _Scalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return float(self._value)


class _TensorArray:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self._values, dtype=np.float32)


class _FakeBox:
    def __init__(self, bbox_xyxy: list[float], confidence: float, cls_idx: int) -> None:
        self.xyxy = [_TensorArray(bbox_xyxy)]
        self.conf = [_Scalar(confidence)]
        self.cls = [_Scalar(cls_idx)]


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = [_FakeBox([88.0, 72.0, 136.0, 110.0], 0.92, 0)]
        self.names = {0: "car"}


class _FakeAliasResult:
    def __init__(self) -> None:
        self.boxes = [
            _FakeBox([88.0, 72.0, 136.0, 110.0], 0.92, 0),
            _FakeBox([140.0, 15.0, 155.0, 55.0], 0.85, 1),
        ]
        self.names = {0: "vehicle", 1: "traffic_light_red"}


class _FakeModel:
    def predict(self, source, device, verbose):  # noqa: ANN001
        return [_FakeResult()]


class _FakeAliasModel:
    def predict(self, source, device, verbose):  # noqa: ANN001
        return [_FakeAliasResult()]


class _FakeBatchResult:
    def __init__(self, bbox_xyxy: list[float], confidence: float, label_map: dict[int, str], cls_idx: int) -> None:
        self.boxes = [_FakeBox(bbox_xyxy, confidence, cls_idx)]
        self.names = label_map


class _FakeBatchModel:
    def __init__(self) -> None:
        self.last_source_shapes: list[tuple[int, int]] = []

    def predict(self, source, device, verbose):  # noqa: ANN001
        sources = source if isinstance(source, list) else [source]
        self.last_source_shapes = [tuple(frame.shape[:2]) for frame in sources]
        return [
            _FakeBatchResult([16.0, 12.0, 32.0, 28.0], 0.9, {0: "car"}, 0),
            _FakeBatchResult([8.0, 4.0, 24.0, 20.0], 0.95, {0: "traffic_light_red"}, 0),
        ]


def test_perception_stack_prefers_camera_mode_when_yolo_is_available() -> None:
    module = PerceptionStack(device="cpu", model_variant="auto")
    module.detector._model = _FakeModel()
    bundle = _bundle()
    bundle.left_camera.status = SensorStatus.OFFLINE
    bundle.right_camera.status = SensorStatus.OFFLINE
    bundle.rear_camera.status = SensorStatus.OFFLINE
    detections, _, _, traffic_lights, _ = module.run(bundle)
    assert len(detections) == 1
    assert detections[0].source_modality == "camera"
    assert detections[0].position_estimate_kind == "camera_projection"
    assert traffic_lights == []
    assert bundle.metadata["perception_summary"].fallback_state == "camera_only"


def test_perception_stack_supports_custom_model_label_aliases() -> None:
    module = PerceptionStack(device="cpu", model_variant="auto")
    module.detector._model = _FakeAliasModel()
    bundle = _bundle()
    bundle.left_camera.status = SensorStatus.OFFLINE
    bundle.right_camera.status = SensorStatus.OFFLINE
    bundle.rear_camera.status = SensorStatus.OFFLINE
    detections, _, _, traffic_lights, _ = module.run(bundle)
    assert len(detections) == 1
    assert detections[0].object_class == ObjectClass.VEHICLE
    assert len(traffic_lights) == 1
    assert traffic_lights[0].state.value == "RED"


def test_camera_budget_policy_defaults_to_aggressive_live_budget() -> None:
    runtime = type(
        "Runtime",
        (),
        {
            "backend": "carla",
            "perception_device": "cpu",
            "tuning": {},
        },
    )()

    policy = _resolve_perception_camera_budget_policy(runtime)

    assert policy.policy == "aggressive_budget"
    assert policy.enable_batch_inference is True
    assert policy.front_run_every_n_ticks == 1
    assert policy.side_run_every_n_ticks == 4
    assert policy.rear_run_every_n_ticks == 10
    assert policy.front_max_input_long_edge_px == 640
    assert policy.surround_max_input_long_edge_px == 384
    assert policy.skip_stale_frames is True


def test_yolo_detector_batch_resizes_and_filters_surround_traffic_lights() -> None:
    detector = YoloObjectDetector(model_variant="auto", device="cpu")
    detector._model = _FakeBatchModel()
    front_frame = np.zeros((256, 512, 3), dtype=np.float32)
    side_frame = np.zeros((240, 320, 3), dtype=np.float32)

    detections_by_sensor, detector_modes = detector.detect_batch(
        [
            CameraInferenceRequest(
                frame=front_frame,
                bootstrap_annotations=[],
                sensor_id="front_camera",
                max_input_long_edge_px=128,
                allowed_classes=(
                    ObjectClass.VEHICLE,
                    ObjectClass.CYCLIST,
                    ObjectClass.PEDESTRIAN,
                    ObjectClass.TRAFFIC_LIGHT,
                ),
            ),
            CameraInferenceRequest(
                frame=side_frame,
                bootstrap_annotations=[],
                sensor_id="left_camera",
                max_input_long_edge_px=64,
                allowed_classes=(
                    ObjectClass.VEHICLE,
                    ObjectClass.CYCLIST,
                    ObjectClass.PEDESTRIAN,
                ),
            ),
        ]
    )

    assert detector_modes == {"front_camera": "camera", "left_camera": "camera"}
    assert detector._model.last_source_shapes == [(64, 128), (48, 64)]
    assert len(detections_by_sensor["front_camera"]) == 1
    assert detections_by_sensor["front_camera"][0].object_class == ObjectClass.VEHICLE
    assert detections_by_sensor["front_camera"][0].bbox_xyxy[2] > 100.0
    assert detections_by_sensor["left_camera"] == []


def test_tracker_confirms_persistent_tracks() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    first_bundle = _bundle()
    second_bundle = _bundle()
    first_detections, _, _, _, _ = module.run(first_bundle)
    second_detections, _, _, _, _ = module.run(second_bundle)
    assert first_detections[0].track_state == TrackState.TENTATIVE
    assert second_detections[0].track_state == TrackState.CONFIRMED


def test_tracker_predict_only_preserves_confirmed_tracks() -> None:
    tracker = KalmanSortTracker(confirm_hits=2)
    detection = FrameDetection2D(
        bbox_xyxy=np.array([10.0, 10.0, 30.0, 30.0], dtype=np.float32),
        object_class=ObjectClass.VEHICLE,
        confidence=0.9,
        source_sensor_id="front_camera",
        source_sensor_ids=["front_camera"],
    )

    tracker.update([detection])
    tracker.update([detection])
    predicted = tracker.predict_only()

    assert len(predicted) == 1
    assert predicted[0].track_state == TrackState.CONFIRMED


def test_perception_stack_skips_stale_surround_cameras_and_records_metadata() -> None:
    policy = _PerceptionCameraBudgetPolicy(
        policy="aggressive_budget",
        enable_batch_inference=False,
        front_run_every_n_ticks=1,
        side_run_every_n_ticks=4,
        rear_run_every_n_ticks=10,
        front_max_input_long_edge_px=640,
        surround_max_input_long_edge_px=384,
        promotion_window_ticks=10,
        side_promotion_distance_m=15.0,
        rear_promotion_distance_m=20.0,
        skip_stale_frames=True,
    )
    module = PerceptionStack(
        device="cpu",
        model_variant="bootstrap",
        camera_budget_policy=policy,
    )
    first_bundle = _bundle()
    first_bundle.tick_id = 0
    first_bundle.front_camera.frame_id = 0
    first_bundle.left_camera.frame_id = 0
    first_bundle.right_camera.frame_id = 0
    first_bundle.rear_camera.frame_id = 0
    module.run(first_bundle)

    second_bundle = _bundle()
    second_bundle.tick_id = 1
    second_bundle.front_camera.frame_id = 1
    second_bundle.left_camera.frame_id = 0
    second_bundle.right_camera.frame_id = 0
    second_bundle.rear_camera.frame_id = 0
    second_bundle.left_camera.status = SensorStatus.DEGRADED
    second_bundle.right_camera.status = SensorStatus.DEGRADED
    second_bundle.rear_camera.status = SensorStatus.DEGRADED
    module.run(second_bundle)

    assert second_bundle.metadata["perception_cameras_scheduled"] == ["front_camera"]
    assert set(second_bundle.metadata["perception_cameras_skipped"]) == {"left_camera", "right_camera", "rear_camera"}
    assert second_bundle.metadata["perception_inference_ms_total"] == pytest.approx(0.0)


def test_perception_stack_promotes_side_cameras_on_yaw_rate() -> None:
    policy = _PerceptionCameraBudgetPolicy(
        policy="aggressive_budget",
        enable_batch_inference=False,
        front_run_every_n_ticks=1,
        side_run_every_n_ticks=4,
        rear_run_every_n_ticks=10,
        front_max_input_long_edge_px=640,
        surround_max_input_long_edge_px=384,
        promotion_window_ticks=10,
        side_promotion_distance_m=15.0,
        rear_promotion_distance_m=20.0,
        skip_stale_frames=False,
    )
    module = PerceptionStack(
        device="cpu",
        model_variant="bootstrap",
        camera_budget_policy=policy,
    )
    first_bundle = _bundle()
    first_bundle.tick_id = 0
    module.run(first_bundle)

    second_bundle = _bundle()
    second_bundle.tick_id = 1
    second_bundle.imu.gyro_xyz[2] = 0.2
    module.run(second_bundle)

    assert set(second_bundle.metadata["perception_cameras_promoted"]) >= {"left_camera", "right_camera"}
    assert set(second_bundle.metadata["perception_cameras_scheduled"]) >= {"front_camera", "left_camera", "right_camera"}



def test_perception_stack_merges_duplicate_tracks_across_cameras() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    bundle = _bundle()
    front_annotations = bundle.metadata["carla_actor_annotations"]
    bundle.metadata["carla_camera_annotations"] = {
        "front_camera": [],
        "left_camera": front_annotations,
        "right_camera": front_annotations,
        "rear_camera": [],
    }

    detections, _, _, traffic_lights, _ = module.run(bundle)

    assert len(detections) == 1
    assert traffic_lights == []
    assert bundle.metadata["perception_camera_detection_counts"] == {
        "front_camera": 0,
        "left_camera": 1,
        "right_camera": 0,
        "rear_camera": 0,
    }
    assert bundle.metadata["perception_active_cameras"] == [
        "front_camera",
        "left_camera",
        "right_camera",
        "rear_camera",
    ]


def test_perception_stack_degrades_without_crashing() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    bundle = _bundle()
    module.lane_extractor.extract = lambda frame, **kwargs: (_ for _ in ()).throw(RuntimeError("lane fail"))  # type: ignore[method-assign]
    detections, lanes, drivable, traffic_lights, cones = module.run(bundle)
    assert detections == []
    assert lanes == []
    assert traffic_lights == []
    assert cones == []
    assert bundle.metadata["perception_status"] == "degraded"
    assert not bool(drivable.mask.any())


def test_perception_stack_forwards_imu_yaw_rate_to_lane_extractor() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    bundle = _bundle()
    bundle.imu.gyro_xyz[2] = 0.37
    captured_kwargs: dict[str, float] = {}

    def _capture_lane_extract(frame, **kwargs):  # noqa: ANN001
        captured_kwargs.update(kwargs)
        return []

    module.lane_extractor.extract = _capture_lane_extract  # type: ignore[method-assign]

    module.run(bundle)

    assert captured_kwargs["ego_yaw_rate_rad_s"] == pytest.approx(0.37)


def test_lidar_perception_stack_extracts_clusters() -> None:
    module = LidarPerceptionStack()
    bundle = _lidar_bundle()
    detections, lanes, drivable, traffic_lights, cones = module.run(bundle)
    assert len(detections) == 2
    assert any(detection.object_class == ObjectClass.VEHICLE for detection in detections)
    assert all(detection.track_state == TrackState.TENTATIVE for detection in detections)
    assert all(detection.source_modality == "lidar" for detection in detections)
    assert all(detection.position_estimate_kind == "lidar_cluster" for detection in detections)
    assert cones == []
    assert isinstance(lanes, list)
    assert drivable.mask.shape == (120, 200)
    assert traffic_lights == []
    assert bundle.metadata["perception_summary"].fallback_state == "lidar_only"


def test_lidar_perception_tracker_confirms_persistent_clusters() -> None:
    module = LidarPerceptionStack()
    first_detections, _, _, _, _ = module.run(_lidar_bundle(sim_time_s=0.0, ego_x=0.0))
    second_detections, _, _, _, _ = module.run(_lidar_bundle(sim_time_s=0.1, ego_x=0.0))
    assert first_detections[0].track_state == TrackState.TENTATIVE
    assert second_detections[0].track_state == TrackState.CONFIRMED
    assert second_detections[0].track_id == first_detections[0].track_id


def test_lidar_perception_filters_oversized_static_clusters() -> None:
    module = LidarPerceptionStack()
    bundle = _lidar_bundle()
    oversized_cluster = np.array(
        [
            [12.0, 2.0, 0.2],
            [20.0, 2.0, 0.4],
            [20.0, 5.5, 1.2],
            [12.0, 5.5, 1.0],
            [16.0, 3.5, 2.8],
            [18.0, 4.5, 3.4],
        ],
        dtype=np.float32,
    )
    bundle.lidar.points_xyz = np.vstack([bundle.lidar.points_xyz, oversized_cluster])
    detections, _, _, _, _ = module.run(bundle)
    assert len(detections) == 2
    for detection in detections:
        bbox = np.asarray(detection.world_bbox_3d, dtype=np.float32)
        size_xyz = np.max(bbox, axis=0) - np.min(bbox, axis=0)
        assert max(float(size_xyz[0]), float(size_xyz[1])) < 10.0


def _object_detection(
    *,
    track_id: int,
    object_class: ObjectClass,
    center_x: float,
    center_y: float,
    confidence: float,
    source_modality: str,
) -> ObjectDetection:
    return ObjectDetection(
        track_id=track_id,
        object_class=object_class,
        world_bbox_3d=np.array(
            [
                [center_x - 1.0, center_y - 0.5, 0.0],
                [center_x + 1.0, center_y - 0.5, 0.0],
                [center_x + 1.0, center_y + 0.5, 0.0],
                [center_x - 1.0, center_y + 0.5, 0.0],
                [center_x - 1.0, center_y - 0.5, 1.5],
                [center_x + 1.0, center_y - 0.5, 1.5],
                [center_x + 1.0, center_y + 0.5, 1.5],
                [center_x - 1.0, center_y + 0.5, 1.5],
            ],
            dtype=np.float32,
        ),
        velocity=np.zeros(3, dtype=np.float32),
        confidence=confidence,
        track_state=TrackState.CONFIRMED,
        image_bbox_xyxy=np.array([50.0, 50.0, 90.0, 90.0], dtype=np.float32),
        source_modality=source_modality,
        source_sensor_ids=["front_camera"] if source_modality != "lidar" else ["lidar"],
        position_estimate_kind="camera_projection" if source_modality != "lidar" else "lidar_cluster",
    )


def test_fuse_detections_prefers_lidar_geometry_and_camera_semantics_when_matched() -> None:
    camera_detection = _object_detection(
        track_id=10,
        object_class=ObjectClass.PEDESTRIAN,
        center_x=12.0,
        center_y=1.0,
        confidence=0.9,
        source_modality="camera",
    )
    lidar_detection = _object_detection(
        track_id=20,
        object_class=ObjectClass.CYCLIST,
        center_x=12.5,
        center_y=1.2,
        confidence=0.7,
        source_modality="lidar",
    )
    fused, used_lidar = fuse_detections([camera_detection], [lidar_detection])
    assert len(fused) == 1
    assert fused[0].source_modality == "fused"
    assert fused[0].track_id == 20
    assert fused[0].object_class == ObjectClass.PEDESTRIAN
    assert fused[0].position_estimate_kind == "fusion"
    assert fused[0].source_sensor_ids == ["front_camera", "lidar"]
    assert 0 in used_lidar


def test_fuse_detections_returns_no_fused_when_unmatched() -> None:
    camera_detection = _object_detection(
        track_id=1,
        object_class=ObjectClass.VEHICLE,
        center_x=5.0,
        center_y=0.0,
        confidence=0.8,
        source_modality="camera",
    )
    lidar_detection = _object_detection(
        track_id=2,
        object_class=ObjectClass.VEHICLE,
        center_x=20.0,
        center_y=0.0,
        confidence=0.8,
        source_modality="lidar",
    )
    fused, used_lidar = fuse_detections([camera_detection], [lidar_detection], match_distance_m=3.0)
    assert fused == []
    assert used_lidar == set()


def test_fuse_detections_drops_conflicting_unmatched_classes() -> None:
    camera_detection = _object_detection(
        track_id=3,
        object_class=ObjectClass.VEHICLE,
        center_x=10.0,
        center_y=0.0,
        confidence=0.8,
        source_modality="camera",
    )
    lidar_detection = _object_detection(
        track_id=4,
        object_class=ObjectClass.PEDESTRIAN,
        center_x=10.2,
        center_y=0.1,
        confidence=0.8,
        source_modality="lidar",
    )
    fused, used_lidar = fuse_detections([camera_detection], [lidar_detection], match_distance_m=2.0)
    assert fused == []
    assert used_lidar == set()


def test_fused_perception_stack_publishes_camera_only_detections_when_no_fused_match() -> None:
    module = FusedPerceptionStack(device="cpu", model_variant="bootstrap")
    camera_detection = _object_detection(
        track_id=31,
        object_class=ObjectClass.VEHICLE,
        center_x=10.0,
        center_y=0.0,
        confidence=0.3,
        source_modality="camera",
    )
    lidar_detection = _object_detection(
        track_id=41,
        object_class=ObjectClass.VEHICLE,
        center_x=24.0,
        center_y=0.0,
        confidence=0.9,
        source_modality="lidar",
    )
    module.camera_stack.detect_dynamic = lambda bundle: (  # type: ignore[method-assign]
        [camera_detection],
        [],
        "camera_only",
        ["front_camera"],
        {"front_camera": []},
    )
    module.lidar_stack.detect_dynamic = lambda bundle: [lidar_detection]  # type: ignore[method-assign]

    detections, _, _, traffic_lights, _ = module.run(_bundle())

    assert len(detections) == 2
    track_ids = {d.track_id for d in detections}
    assert 31 in track_ids  # unmatched camera detection recovered
    assert 41 in track_ids  # unmatched lidar detection recovered
    assert traffic_lights == []


def test_fused_perception_stack_keeps_unmatched_camera_objects_alongside_fused_matches() -> None:
    module = FusedPerceptionStack(device="cpu", model_variant="bootstrap")
    fused_camera_detection = _object_detection(
        track_id=50,
        object_class=ObjectClass.VEHICLE,
        center_x=12.0,
        center_y=0.0,
        confidence=0.9,
        source_modality="camera",
    )
    unmatched_camera_detection = _object_detection(
        track_id=51,
        object_class=ObjectClass.VEHICLE,
        center_x=24.0,
        center_y=0.0,
        confidence=0.4,
        source_modality="camera",
    )
    lidar_detection = _object_detection(
        track_id=60,
        object_class=ObjectClass.VEHICLE,
        center_x=12.5,
        center_y=0.2,
        confidence=0.8,
        source_modality="lidar",
    )
    module.camera_stack.detect_dynamic = lambda bundle: (  # type: ignore[method-assign]
        [fused_camera_detection, unmatched_camera_detection],
        [],
        "camera_only",
        ["front_camera"],
        {"front_camera": []},
    )
    module.lidar_stack.detect_dynamic = lambda bundle: [lidar_detection]  # type: ignore[method-assign]

    detections, _, _, _, _ = module.run(_bundle())

    assert len(detections) == 2
    assert {detection.source_modality for detection in detections} == {"fused", "camera"}


def test_mapping_consumes_perception_outputs() -> None:
    module = PerceptionStack(device="cpu", model_variant="bootstrap")
    detections, lanes, drivable, traffic_lights, cones = module.run(_bundle())
    ego_pose = type(
        "Pose",
        (),
        {
            "current_lane_id": "lane_001",
            "world_xyz": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        },
    )()
    local_map = StubMappingModule().run(detections, lanes, drivable, cones, traffic_lights, ego_pose)
    assert local_map.dynamic_agents == detections
    assert local_map.static_lanes
