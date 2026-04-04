from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import Counter

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import LaneLineType, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    ConeDetection,
    DrivableSpaceMask,
    LaneLine,
    ObjectDetection,
    PerceptionStatus,
    SensorFrameBundle,
    TrafficLightDetection,
)
from autonomy_demo.perception.drivable_space import DrivableSpaceExtractor
from autonomy_demo.perception.fusion import fuse_detections
from autonomy_demo.perception.internal_types import (
    FrameDetection2D,
    TrackedDetection2D,
    TrackedLidarClusterDetection,
)
from autonomy_demo.perception.lane_extraction import LaneExtractor
from autonomy_demo.perception.lidar_detection import LidarObstacleDetector
from autonomy_demo.perception.lidar_tracking import KalmanCentroidTracker3D, SimpleCentroidTracker3D
from autonomy_demo.perception.object_detection import (
    CameraInferenceRequest,
    YoloObjectDetector,
    bootstrap_annotations_from_metadata,
)
from autonomy_demo.perception.tracking import KalmanSortTracker, SimpleSortTracker

# Lazy imports for learned perception (optional heavy dependencies)
_segformer_cls = None
_learned_lane_cls = None


@dataclass(slots=True)
class _PerceptionAuxPolicy:
    policy: str = "max_fidelity"
    enable_segformer: bool = True
    enable_learned_lanes: bool = True
    allow_online_lane_training: bool = True
    segformer_run_every_n_ticks: int = 5
    lane_run_every_n_ticks: int = 1
    segformer_max_input_long_edge_px: int | None = None


@dataclass(slots=True)
class _PerceptionCameraBudgetPolicy:
    policy: str = "max_fidelity"
    enable_batch_inference: bool = False
    front_run_every_n_ticks: int = 1
    side_run_every_n_ticks: int = 1
    rear_run_every_n_ticks: int = 1
    front_max_input_long_edge_px: int | None = None
    surround_max_input_long_edge_px: int | None = None
    promotion_window_ticks: int = 10
    side_promotion_distance_m: float = 15.0
    rear_promotion_distance_m: float = 20.0
    skip_stale_frames: bool = False


@dataclass(slots=True)
class _CameraRuntimeState:
    last_inferred_frame_id: int | None = None
    last_inference_tick: int = -1
    promotion_until_tick: int = -1
    last_detector_mode: str = "bootstrap"
    last_raw_detections: list[FrameDetection2D] = field(default_factory=list)


def _resolve_perception_aux_policy(runtime_config) -> _PerceptionAuxPolicy:
    tuning = dict(getattr(runtime_config, "tuning", {}) or {})
    aux_tuning = dict(tuning.get("perception_aux", {}) or {})
    policy_name = str(aux_tuning.get("policy") or "aggressive_budget").strip().lower()
    is_live = str(getattr(runtime_config, "backend", "stub")).lower() == "carla"
    device = str(getattr(runtime_config, "perception_device", "cpu")).lower()
    enable_learned = bool(getattr(runtime_config, "enable_learned_perception", True))

    if is_live:
        if policy_name == "aggressive_budget":
            if device == "cpu":
                policy = _PerceptionAuxPolicy(
                    policy="aggressive_budget",
                    enable_segformer=False,
                    enable_learned_lanes=False,
                    allow_online_lane_training=False,
                    segformer_run_every_n_ticks=10,
                    lane_run_every_n_ticks=10,
                    segformer_max_input_long_edge_px=512,
                )
            else:
                policy = _PerceptionAuxPolicy(
                    policy="aggressive_budget",
                    enable_segformer=True,
                    enable_learned_lanes=False,
                    allow_online_lane_training=False,
                    segformer_run_every_n_ticks=10,
                    lane_run_every_n_ticks=10,
                    segformer_max_input_long_edge_px=512,
                )
        elif policy_name == "balanced":
            policy = _PerceptionAuxPolicy(
                policy="balanced",
                enable_segformer=device != "cpu",
                enable_learned_lanes=False,
                allow_online_lane_training=False,
                segformer_run_every_n_ticks=5,
                lane_run_every_n_ticks=5,
                segformer_max_input_long_edge_px=768,
            )
        else:
            policy = _PerceptionAuxPolicy(
                policy="max_fidelity",
                enable_segformer=enable_learned,
                enable_learned_lanes=enable_learned and device != "cpu",
                allow_online_lane_training=device != "cpu",
                segformer_run_every_n_ticks=1,
                lane_run_every_n_ticks=1,
                segformer_max_input_long_edge_px=None,
            )
    else:
        policy = _PerceptionAuxPolicy(
            policy=policy_name,
            enable_segformer=enable_learned,
            enable_learned_lanes=enable_learned,
            allow_online_lane_training=True,
            segformer_run_every_n_ticks=5,
            lane_run_every_n_ticks=1,
            segformer_max_input_long_edge_px=None,
        )

    if "enable_segformer" in aux_tuning:
        policy.enable_segformer = bool(aux_tuning["enable_segformer"])
    if "enable_learned_lanes" in aux_tuning:
        policy.enable_learned_lanes = bool(aux_tuning["enable_learned_lanes"])
    if "allow_online_lane_training" in aux_tuning:
        policy.allow_online_lane_training = bool(aux_tuning["allow_online_lane_training"])
    if "segformer_run_every_n_ticks" in aux_tuning:
        policy.segformer_run_every_n_ticks = max(int(aux_tuning["segformer_run_every_n_ticks"]), 1)
    if "lane_run_every_n_ticks" in aux_tuning:
        policy.lane_run_every_n_ticks = max(int(aux_tuning["lane_run_every_n_ticks"]), 1)
    if "segformer_max_input_long_edge_px" in aux_tuning:
        value = aux_tuning["segformer_max_input_long_edge_px"]
        policy.segformer_max_input_long_edge_px = None if value in {None, 0, "0"} else int(value)

    if not enable_learned:
        policy.enable_segformer = False
        policy.enable_learned_lanes = False
        policy.allow_online_lane_training = False

    return policy


def _resolve_perception_camera_budget_policy(runtime_config) -> _PerceptionCameraBudgetPolicy:
    tuning = dict(getattr(runtime_config, "tuning", {}) or {})
    camera_tuning = dict(tuning.get("perception_camera_budget", {}) or {})
    policy_name = str(camera_tuning.get("policy") or "aggressive_budget").strip().lower()
    is_live = str(getattr(runtime_config, "backend", "stub")).lower() == "carla"
    device = str(getattr(runtime_config, "perception_device", "cpu")).lower()

    if is_live and policy_name == "aggressive_budget":
        if device == "cpu":
            policy = _PerceptionCameraBudgetPolicy(
                policy="aggressive_budget",
                enable_batch_inference=True,
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
        else:
            policy = _PerceptionCameraBudgetPolicy(
                policy="aggressive_budget",
                enable_batch_inference=True,
                front_run_every_n_ticks=1,
                side_run_every_n_ticks=4,
                rear_run_every_n_ticks=10,
                front_max_input_long_edge_px=960,
                surround_max_input_long_edge_px=512,
                promotion_window_ticks=10,
                side_promotion_distance_m=15.0,
                rear_promotion_distance_m=20.0,
                skip_stale_frames=True,
            )
    elif is_live and policy_name == "balanced":
        policy = _PerceptionCameraBudgetPolicy(
            policy="balanced",
            enable_batch_inference=True,
            front_run_every_n_ticks=1,
            side_run_every_n_ticks=2,
            rear_run_every_n_ticks=4,
            front_max_input_long_edge_px=960 if device != "cpu" else 640,
            surround_max_input_long_edge_px=640 if device != "cpu" else 448,
            promotion_window_ticks=10,
            side_promotion_distance_m=18.0,
            rear_promotion_distance_m=24.0,
            skip_stale_frames=True,
        )
    elif is_live and policy_name == "coverage_first":
        policy = _PerceptionCameraBudgetPolicy(
            policy="coverage_first",
            enable_batch_inference=True,
            front_run_every_n_ticks=1,
            side_run_every_n_ticks=1,
            rear_run_every_n_ticks=2,
            front_max_input_long_edge_px=960 if device != "cpu" else 640,
            surround_max_input_long_edge_px=640 if device != "cpu" else 448,
            promotion_window_ticks=10,
            side_promotion_distance_m=20.0,
            rear_promotion_distance_m=24.0,
            skip_stale_frames=True,
        )
    else:
        policy = _PerceptionCameraBudgetPolicy(
            policy=policy_name,
            enable_batch_inference=False,
            front_run_every_n_ticks=1,
            side_run_every_n_ticks=1,
            rear_run_every_n_ticks=1,
            front_max_input_long_edge_px=None,
            surround_max_input_long_edge_px=None,
            promotion_window_ticks=10,
            side_promotion_distance_m=15.0,
            rear_promotion_distance_m=20.0,
            skip_stale_frames=False,
        )

    if "enable_batch_inference" in camera_tuning:
        policy.enable_batch_inference = bool(camera_tuning["enable_batch_inference"])
    if "front_run_every_n_ticks" in camera_tuning:
        policy.front_run_every_n_ticks = max(int(camera_tuning["front_run_every_n_ticks"]), 1)
    if "side_run_every_n_ticks" in camera_tuning:
        policy.side_run_every_n_ticks = max(int(camera_tuning["side_run_every_n_ticks"]), 1)
    if "rear_run_every_n_ticks" in camera_tuning:
        policy.rear_run_every_n_ticks = max(int(camera_tuning["rear_run_every_n_ticks"]), 1)
    if "front_max_input_long_edge_px" in camera_tuning:
        value = camera_tuning["front_max_input_long_edge_px"]
        policy.front_max_input_long_edge_px = None if value in {None, 0, "0"} else int(value)
    if "surround_max_input_long_edge_px" in camera_tuning:
        value = camera_tuning["surround_max_input_long_edge_px"]
        policy.surround_max_input_long_edge_px = None if value in {None, 0, "0"} else int(value)
    if "promotion_window_ticks" in camera_tuning:
        policy.promotion_window_ticks = max(int(camera_tuning["promotion_window_ticks"]), 1)
    if "side_promotion_distance_m" in camera_tuning:
        policy.side_promotion_distance_m = float(camera_tuning["side_promotion_distance_m"])
    if "rear_promotion_distance_m" in camera_tuning:
        policy.rear_promotion_distance_m = float(camera_tuning["rear_promotion_distance_m"])
    if "skip_stale_frames" in camera_tuning:
        policy.skip_stale_frames = bool(camera_tuning["skip_stale_frames"])

    return policy


def _world_to_ego_xy(world_xyz: np.ndarray, ego_xyz: np.ndarray, ego_yaw_rad: float) -> np.ndarray:
    delta_xy = np.asarray(world_xyz, dtype=np.float32)[:2] - np.asarray(ego_xyz, dtype=np.float32)[:2]
    cos_yaw = float(np.cos(ego_yaw_rad))
    sin_yaw = float(np.sin(ego_yaw_rad))
    return np.array(
        [
            cos_yaw * delta_xy[0] + sin_yaw * delta_xy[1],
            -sin_yaw * delta_xy[0] + cos_yaw * delta_xy[1],
        ],
        dtype=np.float32,
    )


def _rotate_world_velocity_to_ego(velocity_xyz: np.ndarray, ego_yaw_rad: float) -> np.ndarray:
    velocity_xy = np.asarray(velocity_xyz, dtype=np.float32)[:2]
    cos_yaw = float(np.cos(ego_yaw_rad))
    sin_yaw = float(np.sin(ego_yaw_rad))
    return np.array(
        [
            cos_yaw * velocity_xy[0] + sin_yaw * velocity_xy[1],
            -sin_yaw * velocity_xy[0] + cos_yaw * velocity_xy[1],
        ],
        dtype=np.float32,
    )


def _get_segformer_class():
    global _segformer_cls
    if _segformer_cls is None:
        try:
            from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor
            _segformer_cls = SegFormerDrivableExtractor
        except ImportError:
            _segformer_cls = False  # type: ignore[assignment]
    return _segformer_cls if _segformer_cls is not False else None


def _get_learned_lane_class():
    global _learned_lane_cls
    if _learned_lane_cls is None:
        try:
            from autonomy_demo.perception.learned_lane_detection import LearnedLaneExtractor
            _learned_lane_cls = LearnedLaneExtractor
        except ImportError:
            _learned_lane_cls = False  # type: ignore[assignment]
    return _learned_lane_cls if _learned_lane_cls is not False else None


def _count_by_modality(
    detections: list[ObjectDetection],
    traffic_lights: list[TrafficLightDetection],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for detection in detections:
        counts[str(detection.source_modality)] += 1
    for traffic_light in traffic_lights:
        counts[str(traffic_light.source_modality)] += 1
    return dict(sorted(counts.items()))


def _build_perception_status(
    *,
    active_mode: str,
    fallback_state: str,
    detections: list[ObjectDetection],
    traffic_lights: list[TrafficLightDetection],
    active_camera_sensors: list[str],
) -> PerceptionStatus:
    return PerceptionStatus(
        active_mode=active_mode,
        fallback_state=fallback_state,
        counts_by_modality=_count_by_modality(detections, traffic_lights),
        active_camera_sensors=active_camera_sensors,
        detection_count=len(detections),
        traffic_light_count=len(traffic_lights),
    )


def _record_perception_metadata(
    bundle: SensorFrameBundle,
    *,
    detections: list[ObjectDetection],
    lanes: list[LaneLine],
    drivable_space: DrivableSpaceMask,
    traffic_lights: list[TrafficLightDetection],
    status_summary: PerceptionStatus,
) -> None:
    bundle.metadata["perception_status"] = "ok"
    bundle.metadata["perception_detection_count"] = len(detections)
    bundle.metadata["perception_lane_count"] = len(lanes)
    bundle.metadata["perception_drivable_pixels"] = int(np.count_nonzero(drivable_space.mask))
    bundle.metadata["perception_active_cameras"] = list(status_summary.active_camera_sensors)
    bundle.metadata["perception_summary"] = status_summary


def _public_image_bbox(
    bbox_xyxy: np.ndarray | None,
    *,
    position_estimate_kind: str,
) -> np.ndarray | None:
    if bbox_xyxy is None:
        return None
    return np.asarray(bbox_xyxy, dtype=np.float32)


def _serialize_camera_detections_by_sensor(
    detections_by_sensor: dict[str, list[FrameDetection2D]],
) -> dict[str, list[dict[str, Any]]]:
    serialized: dict[str, list[dict[str, Any]]] = {}
    for sensor_id, detections in detections_by_sensor.items():
        serialized[sensor_id] = [
            {
                "bbox_xyxy": np.asarray(detection.bbox_xyxy, dtype=np.float32).tolist(),
                "object_class": detection.object_class.value,
                "confidence": float(detection.confidence),
                "source_modality": str(detection.source_modality),
                "source_sensor_id": str(detection.source_sensor_id),
            }
            for detection in detections
        ]
    return serialized


def _empty_outputs(
    bundle: SensorFrameBundle,
) -> tuple[
    list[ObjectDetection],
    list[LaneLine],
    DrivableSpaceMask,
    list[TrafficLightDetection],
    list[ConeDetection],
]:
    height, width = bundle.front_camera.frame.shape[:2]
    drivable = DrivableSpaceMask(
        mask=np.zeros((height, width), dtype=np.bool_),
        class_probabilities=np.zeros((height, width, 2), dtype=np.float32),
        source_sensor_id=bundle.front_camera.sensor_id,
    )
    return [], [], drivable, [], []


class StubPerceptionModule:
    """TODO(PRD 3.2.3): replace with model adapters and tracking."""

    def run(self, bundle: SensorFrameBundle):
        detection = ObjectDetection(
            track_id=1,
            object_class=ObjectClass.VEHICLE,
            world_bbox_3d=np.array(
                [
                    [10.0, 1.0, 0.0],
                    [11.0, 1.0, 0.0],
                    [11.0, 2.0, 0.0],
                    [10.0, 2.0, 0.0],
                    [10.0, 1.0, 1.5],
                    [11.0, 1.0, 1.5],
                    [11.0, 2.0, 1.5],
                    [10.0, 2.0, 1.5],
                ],
                dtype=np.float32,
            ),
            velocity=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            confidence=0.95,
            track_state=TrackState.CONFIRMED,
            image_bbox_xyxy=None,
            source_modality="bootstrap",
            source_sensor_ids=["front_camera"],
            position_estimate_kind="truth_fallback",
        )
        lane = LaneLine(
            lane_id="lane_001",
            polyline_image=np.array([[0, 5], [10, 5], [20, 5]], dtype=np.float32),
            polyline_world=np.array([[0, 50, 0], [10, 50, 0], [20, 50, 0]], dtype=np.float32),
            line_type=LaneLineType.SOLID,
            confidence=0.9,
            source_modality="bootstrap",
            source_sensor_ids=["front_camera"],
            position_estimate_kind="truth_fallback",
        )
        drivable = DrivableSpaceMask(
            mask=np.ones((32, 32), dtype=np.bool_),
            class_probabilities=np.ones((32, 32, 2), dtype=np.float32) * 0.5,
            source_sensor_id=bundle.front_camera.sensor_id,
        )
        traffic_light = TrafficLightDetection(
            world_xyz=np.array([30.0, 50.0, 5.0], dtype=np.float32),
            state=TrafficLightState.GREEN,
            stop_line_distance_m=20.0,
            confidence=0.8,
            image_bbox_xyxy=None,
            source_modality="bootstrap",
            source_sensor_ids=["front_camera"],
            position_estimate_kind="truth_fallback",
        )
        return [detection], [lane], drivable, [traffic_light], []


class _CameraSceneContextMixin:
    def _scene_context(
        self,
        bundle: SensorFrameBundle,
        *,
        lane_extractor: LaneExtractor,
        drivable_extractor: DrivableSpaceExtractor,
        learned_drivable_extractor=None,
        learned_lane_extractor=None,
    ) -> tuple[list[LaneLine], DrivableSpaceMask]:
        ego_xyz = np.asarray(bundle.gnss.world_xyz, dtype=np.float32)
        ego_yaw = float(bundle.metadata.get("ego_yaw_rad", 0.0))
        ego_yaw_rate = float(bundle.imu.gyro_xyz[2])
        bundle.metadata["drivable_inference_ms"] = 0.0
        bundle.metadata["lane_inference_ms"] = 0.0

        # --- Drivable space: try learned model first, fall back to heuristic ---
        drivable_space = None
        if learned_drivable_extractor is not None:
            drivable_space = learned_drivable_extractor.extract(
                bundle.front_camera.frame,
                bundle.front_camera.sensor_id,
            )
            if drivable_space is not None:
                bundle.metadata["drivable_source"] = "segformer"
                if getattr(learned_drivable_extractor, "ran_inference_last_call", False):
                    bundle.metadata["drivable_inference_ms"] = learned_drivable_extractor.last_inference_ms
        if drivable_space is None:
            drivable_space = drivable_extractor.extract(
                bundle.front_camera.frame,
                bundle.front_camera.sensor_id,
            )
            bundle.metadata.setdefault("drivable_source", "heuristic")

        # --- Lanes: always run heuristic (for training data + fallback) ---
        heuristic_lanes = lane_extractor.extract(
            bundle.front_camera.frame,
            sensor_id=bundle.front_camera.sensor_id,
            ego_world_xyz=ego_xyz,
            ego_yaw_rad=ego_yaw,
            ego_yaw_rate_rad_s=ego_yaw_rate,
        )

        # Try learned lane detector; it uses heuristic output for self-supervised training
        lanes = heuristic_lanes
        if learned_lane_extractor is not None:
            learned_lanes = learned_lane_extractor.extract(
                bundle.front_camera.frame,
                sensor_id=bundle.front_camera.sensor_id,
                ego_world_xyz=ego_xyz,
                ego_yaw_rad=ego_yaw,
                ego_yaw_rate_rad_s=ego_yaw_rate,
                heuristic_lanes=heuristic_lanes,
            )
            if learned_lanes is not None:
                lanes = learned_lanes
                bundle.metadata["lane_source"] = "learned"
                if getattr(learned_lane_extractor, "ran_inference_last_call", False):
                    bundle.metadata["lane_inference_ms"] = learned_lane_extractor.last_inference_ms
            else:
                bundle.metadata.setdefault("lane_source", "heuristic")
                if hasattr(learned_lane_extractor, "is_trained") and getattr(
                    learned_lane_extractor,
                    "_allow_online_training",
                    False,
                ):
                    bundle.metadata["lane_model_warmup"] = not learned_lane_extractor.is_trained
        else:
            bundle.metadata.setdefault("lane_source", "heuristic")

        return lanes, drivable_space


class PerceptionStack(_CameraSceneContextMixin):
    """Camera-first perception v1 with YOLO-primary detections and explicit bootstrap fallback."""

    def __init__(
        self,
        *,
        device: str,
        model_variant: str,
        enable_learned_perception: bool = True,
        aux_policy: _PerceptionAuxPolicy | None = None,
        camera_budget_policy: _PerceptionCameraBudgetPolicy | None = None,
    ) -> None:
        self.detector = YoloObjectDetector(model_variant=model_variant, device=device)
        self.tracker = KalmanSortTracker()
        self.lane_extractor = LaneExtractor()
        self.drivable_extractor = DrivableSpaceExtractor()
        self.learned_drivable_extractor = None
        self.learned_lane_extractor = None
        self.aux_policy = aux_policy or _PerceptionAuxPolicy()
        self.camera_budget_policy = camera_budget_policy or _PerceptionCameraBudgetPolicy()
        if enable_learned_perception:
            segformer_cls = _get_segformer_class()
            if segformer_cls is not None and self.aux_policy.enable_segformer:
                self.learned_drivable_extractor = segformer_cls(
                    device=device,
                    run_every_n_ticks=self.aux_policy.segformer_run_every_n_ticks,
                    max_input_long_edge_px=self.aux_policy.segformer_max_input_long_edge_px,
                )
            learned_lane_cls = _get_learned_lane_class()
            if learned_lane_cls is not None and self.aux_policy.enable_learned_lanes:
                self.learned_lane_extractor = learned_lane_cls(
                    device=device,
                    run_every_n_ticks=self.aux_policy.lane_run_every_n_ticks,
                    allow_online_training=self.aux_policy.allow_online_lane_training,
                )
        self.logger = get_logger(__name__, perception_mode="camera_v1")
        self._camera_order = ("front_camera", "left_camera", "right_camera", "rear_camera")
        self._camera_runtime_state = {
            sensor_id: _CameraRuntimeState()
            for sensor_id in self._camera_order
        }
        self._last_tracked_detections: list[TrackedDetection2D] = []

    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        try:
            (
                object_detections,
                traffic_lights,
                fallback_state,
                active_camera_sensors,
                camera_detection_debug,
            ) = self.detect_dynamic(bundle)
            lanes, drivable_space = self._scene_context(
                bundle,
                lane_extractor=self.lane_extractor,
                drivable_extractor=self.drivable_extractor,
                learned_drivable_extractor=self.learned_drivable_extractor,
                learned_lane_extractor=self.learned_lane_extractor,
            )
            status_summary = _build_perception_status(
                active_mode="camera_v1",
                fallback_state=fallback_state,
                detections=object_detections,
                traffic_lights=traffic_lights,
                active_camera_sensors=active_camera_sensors,
            )
            _record_perception_metadata(
                bundle,
                detections=object_detections,
                lanes=lanes,
                drivable_space=drivable_space,
                traffic_lights=traffic_lights,
                status_summary=status_summary,
            )
            bundle.metadata["perception_camera_detection_counts"] = self._camera_detection_counts(
                object_detections,
                traffic_lights,
            )
            bundle.metadata["perception_camera_detections"] = camera_detection_debug
            return object_detections, lanes, drivable_space, traffic_lights, []
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            bundle.metadata["perception_camera_detections"] = {}
            self.logger.warning("Perception degraded for tick %s: %s", bundle.tick_id, exc)
            return _empty_outputs(bundle)
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            bundle.metadata["perception_camera_detections"] = {}
            self.logger.error("Unexpected perception failure for tick %s", bundle.tick_id, exc_info=True)
            return _empty_outputs(bundle)

    def detect_dynamic(
        self,
        bundle: SensorFrameBundle,
    ) -> tuple[list[ObjectDetection], list[TrafficLightDetection], str, list[str], dict[str, list[dict[str, Any]]]]:
        detections_2d, detector_modes, active_camera_sensors, detections_by_sensor = self._camera_detections(bundle)
        tracked_detections = self.tracker.update(detections_2d) if detections_2d else self.tracker.predict_only()
        self._last_tracked_detections = list(tracked_detections)
        object_detections, traffic_lights = self._convert_tracked(tracked_detections)
        return (
            object_detections,
            traffic_lights,
            self._camera_fallback_state(detector_modes),
            active_camera_sensors,
            _serialize_camera_detections_by_sensor(detections_by_sensor),
        )

    def _camera_detections(
        self,
        bundle: SensorFrameBundle,
    ) -> tuple[list[FrameDetection2D], dict[str, str], list[str], dict[str, list[FrameDetection2D]]]:
        detections_by_camera: list[FrameDetection2D] = []
        detector_modes: dict[str, str] = {}
        active_camera_sensors = self._active_camera_sensors(bundle)
        detections_by_sensor: dict[str, list[FrameDetection2D]] = {
            sensor_id: list(self._camera_runtime_state[sensor_id].last_raw_detections)
            for sensor_id in active_camera_sensors
        }
        requests, scheduled_sensors, skipped_sensors, promoted_sensors = self._select_cameras_for_inference(
            bundle,
            active_camera_sensors,
        )

        if requests:
            if self.camera_budget_policy.enable_batch_inference:
                batched_detections_by_sensor, batched_detector_modes = self.detector.detect_batch(requests)
            else:
                batched_detections_by_sensor = {}
                batched_detector_modes = {}
                total_ms = 0.0
                per_sensor_ms: dict[str, float] = {}
                for request in requests:
                    detections, detector_mode = self.detector.detect(
                        request.frame,
                        request.bootstrap_annotations,
                        sensor_id=request.sensor_id,
                        ego_world_xyz=request.ego_world_xyz,
                        ego_yaw_rad=request.ego_yaw_rad,
                    )
                    batched_detections_by_sensor[request.sensor_id] = detections
                    batched_detector_modes[request.sensor_id] = detector_mode
                    sensor_ms = float(self.detector.last_inference_ms_by_sensor.get(request.sensor_id, self.detector.last_inference_ms_total))
                    total_ms += sensor_ms
                    per_sensor_ms[request.sensor_id] = sensor_ms
                self.detector.last_inference_ms_total = total_ms
                self.detector.last_inference_ms_by_sensor = per_sensor_ms

            for request in requests:
                sensor_id = request.sensor_id
                camera = getattr(bundle, sensor_id)
                detections = self._filter_camera_detections(
                    sensor_id,
                    batched_detections_by_sensor.get(sensor_id, []),
                )
                detector_mode = batched_detector_modes.get(sensor_id, "bootstrap")
                state = self._camera_runtime_state[sensor_id]
                state.last_inference_tick = int(bundle.tick_id)
                state.last_inferred_frame_id = None if camera.frame_id is None else int(camera.frame_id)
                state.last_detector_mode = detector_mode
                state.last_raw_detections = list(detections)
                detector_modes[sensor_id] = detector_mode
                detections_by_sensor[sensor_id] = list(detections)
                detections_by_camera.extend(detections)

        for sensor_id in active_camera_sensors:
            state = self._camera_runtime_state[sensor_id]
            detector_modes.setdefault(sensor_id, state.last_detector_mode)
            detections_by_sensor.setdefault(sensor_id, list(state.last_raw_detections))

        bundle.metadata["perception_inference_ms_total"] = float(self.detector.last_inference_ms_total if requests else 0.0)
        bundle.metadata["perception_inference_ms_by_sensor"] = {
            sensor_id: float(value)
            for sensor_id, value in (self.detector.last_inference_ms_by_sensor if requests else {}).items()
        }
        bundle.metadata["perception_cameras_scheduled"] = list(scheduled_sensors)
        bundle.metadata["perception_cameras_skipped"] = list(skipped_sensors)
        bundle.metadata["perception_cameras_promoted"] = list(promoted_sensors)
        return (
            self._merge_camera_detections(detections_by_camera),
            detector_modes,
            active_camera_sensors,
            detections_by_sensor,
        )

    def _active_camera_sensors(self, bundle: SensorFrameBundle) -> list[str]:
        return [
            sensor_id
            for sensor_id in self._camera_order
            if getattr(bundle, sensor_id).status.value != "OFFLINE"
        ]

    def _select_cameras_for_inference(
        self,
        bundle: SensorFrameBundle,
        active_camera_sensors: list[str],
    ) -> tuple[list[CameraInferenceRequest], list[str], list[str], list[str]]:
        self._update_camera_promotions(bundle)
        requests: list[CameraInferenceRequest] = []
        scheduled_sensors: list[str] = []
        skipped_sensors: list[str] = []
        promoted_sensors = [
            sensor_id
            for sensor_id in active_camera_sensors
            if self._camera_runtime_state[sensor_id].promotion_until_tick >= int(bundle.tick_id)
        ]
        for sensor_id in active_camera_sensors:
            camera = getattr(bundle, sensor_id)
            state = self._camera_runtime_state[sensor_id]
            if self.camera_budget_policy.skip_stale_frames:
                if camera.status.value == "DEGRADED" and sensor_id != "front_camera":
                    skipped_sensors.append(sensor_id)
                    continue
                if camera.frame_id is not None and state.last_inferred_frame_id == int(camera.frame_id):
                    skipped_sensors.append(sensor_id)
                    continue
            cadence = self._camera_cadence(sensor_id, promoted=sensor_id in promoted_sensors)
            if (
                state.last_inference_tick >= 0
                and int(bundle.tick_id) > state.last_inference_tick
                and (int(bundle.tick_id) - state.last_inference_tick) < cadence
            ):
                skipped_sensors.append(sensor_id)
                continue
            requests.append(self._build_inference_request(bundle, sensor_id))
            scheduled_sensors.append(sensor_id)
        return requests, scheduled_sensors, skipped_sensors, promoted_sensors

    def _update_camera_promotions(self, bundle: SensorFrameBundle) -> None:
        current_tick = int(bundle.tick_id)
        ego_xyz = np.asarray(bundle.gnss.world_xyz, dtype=np.float32)
        ego_yaw = float(bundle.metadata.get("ego_yaw_rad", 0.0))
        ego_yaw_rate = float(bundle.imu.gyro_xyz[2])
        if abs(ego_yaw_rate) > 0.08:
            self._promote_camera("left_camera", current_tick)
            self._promote_camera("right_camera", current_tick)
        front_status = bundle.front_camera.status.value
        if front_status in {"DEGRADED", "OFFLINE"}:
            self._promote_camera("left_camera", current_tick)
            self._promote_camera("right_camera", current_tick)
        for detection in self._last_tracked_detections:
            if detection.track_state != TrackState.CONFIRMED or detection.world_xyz is None:
                continue
            relative_xy = _world_to_ego_xy(detection.world_xyz, ego_xyz, ego_yaw)
            distance_m = float(np.linalg.norm(relative_xy))
            source_sensor_id = str(detection.source_sensor_id)
            if (
                source_sensor_id == "left_camera"
                and relative_xy[1] <= 0.0
                and distance_m <= self.camera_budget_policy.side_promotion_distance_m
            ):
                self._promote_camera("left_camera", current_tick)
            if (
                source_sensor_id == "right_camera"
                and relative_xy[1] >= 0.0
                and distance_m <= self.camera_budget_policy.side_promotion_distance_m
            ):
                self._promote_camera("right_camera", current_tick)
            if (
                source_sensor_id == "rear_camera"
                and distance_m <= self.camera_budget_policy.rear_promotion_distance_m
                and self._is_rear_closing(detection, ego_yaw, relative_xy[0])
            ):
                self._promote_camera("rear_camera", current_tick)

    def _promote_camera(self, sensor_id: str, current_tick: int) -> None:
        state = self._camera_runtime_state[sensor_id]
        state.promotion_until_tick = max(
            state.promotion_until_tick,
            current_tick + self.camera_budget_policy.promotion_window_ticks,
        )

    def _is_rear_closing(self, detection: TrackedDetection2D, ego_yaw_rad: float, relative_x: float) -> bool:
        velocity_xyz = (
            np.asarray(detection.velocity_xyz, dtype=np.float32)
            if detection.velocity_xyz is not None
            else np.zeros(3, dtype=np.float32)
        )
        ego_velocity_xy = _rotate_world_velocity_to_ego(velocity_xyz, ego_yaw_rad)
        return relative_x < 0.0 and ego_velocity_xy[0] > 0.0

    def _camera_cadence(self, sensor_id: str, *, promoted: bool) -> int:
        if sensor_id == "front_camera":
            return 1 if promoted else self.camera_budget_policy.front_run_every_n_ticks
        if sensor_id == "rear_camera":
            return 2 if promoted else self.camera_budget_policy.rear_run_every_n_ticks
        return 1 if promoted else self.camera_budget_policy.side_run_every_n_ticks

    def _build_inference_request(self, bundle: SensorFrameBundle, sensor_id: str) -> CameraInferenceRequest:
        camera = getattr(bundle, sensor_id)
        ego_yaw = float(bundle.metadata.get("ego_yaw_rad", 0.0))
        bootstrap_annotations = bootstrap_annotations_from_metadata(
            bundle.metadata,
            sensor_id=sensor_id,
        )
        if sensor_id == "front_camera":
            allowed_classes = (
                ObjectClass.VEHICLE,
                ObjectClass.CYCLIST,
                ObjectClass.PEDESTRIAN,
                ObjectClass.TRAFFIC_LIGHT,
            )
            min_confidence = 0.0
            min_bbox_area_px = 0.0
            max_input_long_edge_px = self.camera_budget_policy.front_max_input_long_edge_px
        else:
            allowed_classes = (
                ObjectClass.VEHICLE,
                ObjectClass.CYCLIST,
                ObjectClass.PEDESTRIAN,
            )
            min_confidence = 0.45
            min_bbox_area_px = 300.0
            max_input_long_edge_px = self.camera_budget_policy.surround_max_input_long_edge_px
        return CameraInferenceRequest(
            frame=camera.frame,
            bootstrap_annotations=bootstrap_annotations,
            sensor_id=sensor_id,
            ego_world_xyz=np.asarray(bundle.gnss.world_xyz, dtype=np.float32),
            ego_yaw_rad=ego_yaw,
            max_input_long_edge_px=max_input_long_edge_px,
            allowed_classes=allowed_classes,
            min_confidence=min_confidence,
            min_bbox_area_px=min_bbox_area_px,
        )

    def _filter_camera_detections(
        self,
        sensor_id: str,
        detections: list[FrameDetection2D],
    ) -> list[FrameDetection2D]:
        filtered: list[FrameDetection2D] = []
        for detection in detections:
            if sensor_id != "front_camera" and detection.object_class == ObjectClass.TRAFFIC_LIGHT:
                continue
            bbox_xyxy = np.asarray(detection.bbox_xyxy, dtype=np.float32)
            bbox_area = max(0.0, float(bbox_xyxy[2] - bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
            if sensor_id != "front_camera" and (float(detection.confidence) < 0.45 or bbox_area < 300.0):
                continue
            filtered.append(detection)
        return filtered

    def _merge_camera_detections(
        self,
        detections: list[FrameDetection2D],
    ) -> list[FrameDetection2D]:
        merged_by_track: dict[tuple[int, ObjectClass], FrameDetection2D] = {}
        passthrough: list[FrameDetection2D] = []
        for detection in detections:
            if detection.preferred_track_id is None:
                passthrough.append(detection)
                continue
            key = (int(detection.preferred_track_id), detection.object_class)
            existing = merged_by_track.get(key)
            if existing is None or self._camera_rank(detection) > self._camera_rank(existing) or (
                self._camera_rank(detection) == self._camera_rank(existing)
                and detection.confidence > existing.confidence
            ):
                merged_by_track[key] = detection
        return list(merged_by_track.values()) + passthrough

    def _camera_rank(self, detection: FrameDetection2D) -> int:
        try:
            return len(self._camera_order) - self._camera_order.index(detection.source_sensor_id)
        except ValueError:
            return 0

    def _camera_detection_counts(
        self,
        detections: list[ObjectDetection],
        traffic_lights: list[TrafficLightDetection],
    ) -> dict[str, int]:
        counts = {sensor_id: 0 for sensor_id in self._camera_order}
        for detection in detections:
            for sensor_id in detection.source_sensor_ids:
                if sensor_id in counts:
                    counts[sensor_id] += 1
        for traffic_light in traffic_lights:
            for sensor_id in traffic_light.source_sensor_ids:
                if sensor_id in counts:
                    counts[sensor_id] += 1
        return counts

    def _camera_fallback_state(self, detector_modes: dict[str, str]) -> str:
        if any(mode == "camera" for mode in detector_modes.values()):
            return "camera_only"
        return "bootstrap"

    def _convert_tracked(
        self, detections: list[TrackedDetection2D]
    ) -> tuple[list[ObjectDetection], list[TrafficLightDetection]]:
        object_detections: list[ObjectDetection] = []
        traffic_lights: list[TrafficLightDetection] = []
        for detection in detections:
            world_bbox = (
                np.asarray(detection.world_bbox_3d, dtype=np.float32)
                if detection.world_bbox_3d is not None
                else np.zeros((8, 3), dtype=np.float32)
            )
            velocity = (
                np.asarray(detection.velocity_xyz, dtype=np.float32)
                if detection.velocity_xyz is not None
                else np.zeros(3, dtype=np.float32)
            )
            world_xyz = (
                np.asarray(detection.world_xyz, dtype=np.float32)
                if detection.world_xyz is not None
                else np.mean(world_bbox, axis=0).astype(np.float32)
            )
            if detection.object_class == ObjectClass.TRAFFIC_LIGHT:
                traffic_lights.append(
                    TrafficLightDetection(
                        world_xyz=world_xyz,
                        state=detection.traffic_light_state or TrafficLightState.UNKNOWN,
                        stop_line_distance_m=max(0.0, float(world_xyz[0])),
                        confidence=float(detection.confidence),
                        image_bbox_xyxy=_public_image_bbox(
                            detection.bbox_xyxy,
                            position_estimate_kind=detection.position_estimate_kind,
                        ),
                        source_modality=detection.source_modality,
                        source_sensor_ids=list(detection.source_sensor_ids or [detection.source_sensor_id]),
                        position_estimate_kind=detection.position_estimate_kind,
                    )
                )
                continue
            object_detections.append(
                ObjectDetection(
                    track_id=int(detection.track_id),
                    object_class=detection.object_class,
                    world_bbox_3d=world_bbox,
                    velocity=velocity,
                    confidence=float(detection.confidence),
                    track_state=detection.track_state,
                    image_bbox_xyxy=_public_image_bbox(
                        detection.bbox_xyxy,
                        position_estimate_kind=detection.position_estimate_kind,
                    ),
                    source_modality=detection.source_modality,
                    source_sensor_ids=list(detection.source_sensor_ids or [detection.source_sensor_id]),
                    position_estimate_kind=detection.position_estimate_kind,
                    gt_actor_id=detection.preferred_track_id,
                )
            )
        return object_detections, traffic_lights


class LidarPerceptionStack(_CameraSceneContextMixin):
    """LiDAR-first perception v1 with deterministic clustering and centroid tracking."""

    def __init__(self) -> None:
        self.detector = LidarObstacleDetector()
        self.tracker = KalmanCentroidTracker3D()
        self.lane_extractor = LaneExtractor()
        self.drivable_extractor = DrivableSpaceExtractor()
        self.logger = get_logger(__name__, perception_mode="lidar_v1")

    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        try:
            object_detections = self.detect_dynamic(bundle)
            lanes, drivable_space = self._scene_context(
                bundle,
                lane_extractor=self.lane_extractor,
                drivable_extractor=self.drivable_extractor,
            )
            active_camera_sensors = [
                sensor_id
                for sensor_id in ("front_camera", "left_camera", "right_camera", "rear_camera")
                if getattr(bundle, sensor_id).status.value in {"OK", "DEGRADED"}
            ]
            status_summary = _build_perception_status(
                active_mode="lidar_v1",
                fallback_state="lidar_only",
                detections=object_detections,
                traffic_lights=[],
                active_camera_sensors=active_camera_sensors,
            )
            _record_perception_metadata(
                bundle,
                detections=object_detections,
                lanes=lanes,
                drivable_space=drivable_space,
                traffic_lights=[],
                status_summary=status_summary,
            )
            bundle.metadata["perception_lidar_cluster_count"] = len(object_detections)
            bundle.metadata["perception_camera_detections"] = {}
            return object_detections, lanes, drivable_space, [], []
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            bundle.metadata["perception_camera_detections"] = {}
            self.logger.warning("LiDAR perception degraded for tick %s: %s", bundle.tick_id, exc)
            return _empty_outputs(bundle)
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            bundle.metadata["perception_camera_detections"] = {}
            self.logger.error("Unexpected LiDAR perception failure for tick %s", bundle.tick_id, exc_info=True)
            return _empty_outputs(bundle)

    def detect_dynamic(
        self,
        bundle: SensorFrameBundle,
    ) -> list[ObjectDetection]:
        detections_3d = self.detector.detect(bundle)
        tracked_detections = self.tracker.update(detections_3d, timestamp_s=float(bundle.sim_time_s))
        return self._convert_tracked(tracked_detections)

    def _convert_tracked(
        self,
        detections: list[TrackedLidarClusterDetection],
    ) -> list[ObjectDetection]:
        outputs: list[ObjectDetection] = []
        for detection in detections:
            outputs.append(
                ObjectDetection(
                    track_id=int(detection.track_id),
                    object_class=detection.object_class,
                    world_bbox_3d=np.asarray(detection.world_bbox_3d, dtype=np.float32),
                    velocity=(
                        np.asarray(detection.velocity_xyz, dtype=np.float32)
                        if detection.velocity_xyz is not None
                        else np.zeros(3, dtype=np.float32)
                    ),
                    confidence=float(detection.confidence),
                    track_state=detection.track_state,
                    image_bbox_xyxy=None,
                    source_modality=detection.source_modality,
                    source_sensor_ids=list(detection.source_sensor_ids),
                    position_estimate_kind=detection.position_estimate_kind,
                )
            )
        return outputs


class FusedPerceptionStack(_CameraSceneContextMixin):
    """Object-level camera/LiDAR fusion with camera lanes and LiDAR geometry preference."""

    def __init__(
        self,
        *,
        device: str,
        model_variant: str,
        enable_learned_perception: bool = True,
        aux_policy: _PerceptionAuxPolicy | None = None,
        camera_budget_policy: _PerceptionCameraBudgetPolicy | None = None,
    ) -> None:
        self.camera_stack = PerceptionStack(
            device=device,
            model_variant=model_variant,
            enable_learned_perception=enable_learned_perception,
            aux_policy=aux_policy,
            camera_budget_policy=camera_budget_policy,
        )
        self.lidar_stack = LidarPerceptionStack()
        self.lane_extractor = self.camera_stack.lane_extractor
        self.drivable_extractor = self.camera_stack.drivable_extractor
        self.learned_drivable_extractor = self.camera_stack.learned_drivable_extractor
        self.learned_lane_extractor = self.camera_stack.learned_lane_extractor
        self.logger = get_logger(__name__, perception_mode="fused_v1")

    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        try:
            camera_detections, traffic_lights, camera_fallback_state, active_camera_sensors, camera_detection_debug = (
                self.camera_stack.detect_dynamic(bundle)
            )
            lidar_detections = self.lidar_stack.detect_dynamic(bundle)
            fused_objects = fuse_detections(camera_detections, lidar_detections)
            canonical_objects = self._canonical_detections(camera_detections, fused_objects)
            lanes, drivable_space = self._scene_context(
                bundle,
                lane_extractor=self.lane_extractor,
                drivable_extractor=self.drivable_extractor,
                learned_drivable_extractor=self.learned_drivable_extractor,
                learned_lane_extractor=self.learned_lane_extractor,
            )
            status_summary = _build_perception_status(
                    active_mode="fused_v1",
                    fallback_state=self._fused_fallback_state(
                        detections=canonical_objects,
                        camera_fallback_state=camera_fallback_state,
                    ),
                    detections=canonical_objects,
                    traffic_lights=traffic_lights,
                    active_camera_sensors=active_camera_sensors,
                )
            _record_perception_metadata(
                bundle,
                detections=canonical_objects,
                lanes=lanes,
                drivable_space=drivable_space,
                traffic_lights=traffic_lights,
                status_summary=status_summary,
            )
            bundle.metadata["perception_camera_detections"] = camera_detection_debug
            return canonical_objects, lanes, drivable_space, traffic_lights, []
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            bundle.metadata["perception_camera_detections"] = {}
            self.logger.warning("Fused perception degraded for tick %s: %s", bundle.tick_id, exc)
            return _empty_outputs(bundle)
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            bundle.metadata["perception_camera_detections"] = {}
            self.logger.error("Unexpected fused perception failure for tick %s", bundle.tick_id, exc_info=True)
            return _empty_outputs(bundle)

    def _canonical_detections(
        self,
        camera_detections: list[ObjectDetection],
        fused_detections: list[ObjectDetection],
    ) -> list[ObjectDetection]:
        actual_camera_detections = [
            detection for detection in camera_detections if str(detection.source_modality) == "camera"
        ]
        if not fused_detections:
            return actual_camera_detections
        canonical = list(fused_detections)
        for detection in actual_camera_detections:
            if any(self._camera_detection_is_represented(detection, fused) for fused in fused_detections):
                continue
            canonical.append(detection)
        return canonical

    def _camera_detection_is_represented(
        self,
        camera_detection: ObjectDetection,
        fused_detection: ObjectDetection,
        *,
        match_distance_m: float = 5.0,
    ) -> bool:
        if not self._class_compatible(camera_detection, fused_detection):
            return False
        camera_center = np.mean(np.asarray(camera_detection.world_bbox_3d, dtype=np.float32), axis=0)
        fused_center = np.mean(np.asarray(fused_detection.world_bbox_3d, dtype=np.float32), axis=0)
        return float(np.linalg.norm(camera_center[:2] - fused_center[:2])) <= match_distance_m

    def _class_compatible(
        self,
        first: ObjectDetection,
        second: ObjectDetection,
    ) -> bool:
        if first.object_class == second.object_class:
            return True
        soft_pair = {"pedestrian", "cyclist"}
        return first.object_class.value in soft_pair and second.object_class.value in soft_pair

    def _fused_fallback_state(
        self,
        *,
        detections: list[ObjectDetection],
        camera_fallback_state: str,
    ) -> str:
        modalities = {detection.source_modality for detection in detections}
        if "fused" in modalities:
            return "fused"
        if "camera" in modalities:
            return "camera_only"
        if camera_fallback_state == "bootstrap":
            return "bootstrap"
        return "camera_only"


def build_perception_module(runtime_config):
    enable_learned = getattr(runtime_config, "enable_learned_perception", True)
    aux_policy = _resolve_perception_aux_policy(runtime_config)
    camera_budget_policy = _resolve_perception_camera_budget_policy(runtime_config)
    if runtime_config.perception_mode == "camera_v1":
        return PerceptionStack(
            device=runtime_config.perception_device,
            model_variant=runtime_config.perception_model_variant,
            enable_learned_perception=enable_learned,
            aux_policy=aux_policy,
            camera_budget_policy=camera_budget_policy,
        )
    if runtime_config.perception_mode == "lidar_v1":
        return LidarPerceptionStack()
    if runtime_config.perception_mode == "fused_v1":
        return FusedPerceptionStack(
            device=runtime_config.perception_device,
            model_variant=runtime_config.perception_model_variant,
            enable_learned_perception=enable_learned,
            aux_policy=aux_policy,
            camera_budget_policy=camera_budget_policy,
        )
    return StubPerceptionModule()
