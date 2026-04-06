from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np

_logger = logging.getLogger(__name__)

from autonomy_demo.interfaces.enums import ObjectClass, TrafficLightState
from autonomy_demo.perception.internal_types import FrameDetection2D


_YOLO_CLASS_MAP = {
    "car": ObjectClass.VEHICLE,
    "vehicle": ObjectClass.VEHICLE,
    "truck": ObjectClass.VEHICLE,
    "bus": ObjectClass.VEHICLE,
    "motorcycle": ObjectClass.CYCLIST,
    "motobike": ObjectClass.CYCLIST,
    "bicycle": ObjectClass.CYCLIST,
    "bike": ObjectClass.CYCLIST,
    "person": ObjectClass.PEDESTRIAN,
    "pedestrian": ObjectClass.PEDESTRIAN,
    "traffic_light": ObjectClass.TRAFFIC_LIGHT,
    "traffic light": ObjectClass.TRAFFIC_LIGHT,
    "traffic-light": ObjectClass.TRAFFIC_LIGHT,
    "traffic_light_red": ObjectClass.TRAFFIC_LIGHT,
    "traffic_light_orange": ObjectClass.TRAFFIC_LIGHT,
    "traffic_light_green": ObjectClass.TRAFFIC_LIGHT,
}


def _object_class_from_label(label: str) -> ObjectClass | None:
    return _YOLO_CLASS_MAP.get(label.strip().lower())


def _traffic_light_state_from_label(label: str) -> TrafficLightState | None:
    normalized = label.strip().lower()
    if normalized in {"traffic_light_red", "traffic light red", "traffic-light-red"}:
        return TrafficLightState.RED
    if normalized in {"traffic_light_orange", "traffic_light_amber", "traffic light orange", "traffic light amber"}:
        return TrafficLightState.AMBER
    if normalized in {"traffic_light_green", "traffic light green", "traffic-light-green"}:
        return TrafficLightState.GREEN
    return None


def _sensor_mount(sensor_id: str) -> tuple[np.ndarray, float]:
    mounts = {
        "front_camera": (np.array([2.3, 0.0, 0.8], dtype=np.float32), 0.0),
        "rear_camera": (np.array([-2.0, 0.0, 1.0], dtype=np.float32), np.pi),
        "left_camera": (np.array([0.0, -0.8, 1.0], dtype=np.float32), -np.pi * 0.5),
        "right_camera": (np.array([0.0, 0.8, 1.0], dtype=np.float32), np.pi * 0.5),
    }
    return mounts.get(sensor_id, (np.zeros(3, dtype=np.float32), 0.0))


def _rotate_xy(vector_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    rotation = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rotation @ np.asarray(vector_xyz, dtype=np.float32)).astype(np.float32)


def _pseudo_world_box(
    bbox_xyxy: np.ndarray,
    object_class: ObjectClass,
    image_shape: tuple[int, int, int],
    *,
    sensor_id: str,
    ego_world_xyz: np.ndarray,
    ego_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    image_height, image_width = image_shape[:2]
    bbox_width = max(float(bbox_xyxy[2] - bbox_xyxy[0]), 1.0)
    bbox_height = max(float(bbox_xyxy[3] - bbox_xyxy[1]), 1.0)
    center_x = float((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5)
    normalized_x = ((center_x / max(image_width, 1)) - 0.5) * 2.0
    forward_scale = 1200.0
    lateral_scale = 0.8
    if sensor_id == "front_camera" and object_class == ObjectClass.VEHICLE:
        # Front-camera lead vehicles should appear conservative and lane-centered enough
        # for downstream braking to react before the final few meters.
        forward_scale = 650.0
        lateral_scale = 0.35
    forward_distance = float(np.clip(forward_scale / bbox_height, 3.0, 45.0))
    # Monocular depth is inherently noisy — uncertainty scales with estimated distance.
    position_uncertainty_m = forward_distance * 0.3
    lateral_offset = float(normalized_x * forward_distance * lateral_scale)
    size_map = {
        ObjectClass.VEHICLE: (4.5, 2.0, 1.6),
        ObjectClass.CYCLIST: (1.8, 0.8, 1.5),
        ObjectClass.PEDESTRIAN: (0.8, 0.8, 1.8),
        ObjectClass.TRAFFIC_LIGHT: (0.6, 0.6, 3.0),
    }
    length_m, width_m, height_m = size_map.get(object_class, (2.0, 1.0, 1.5))
    center_sensor = np.array(
        [forward_distance, lateral_offset, height_m * 0.5],
        dtype=np.float32,
    )
    dx = length_m * 0.5
    dy = width_m * 0.5
    dz = height_m * 0.5
    sensor_bbox = np.array(
        [
            [center_sensor[0] - dx, center_sensor[1] - dy, center_sensor[2] - dz],
            [center_sensor[0] + dx, center_sensor[1] - dy, center_sensor[2] - dz],
            [center_sensor[0] + dx, center_sensor[1] + dy, center_sensor[2] - dz],
            [center_sensor[0] - dx, center_sensor[1] + dy, center_sensor[2] - dz],
            [center_sensor[0] - dx, center_sensor[1] - dy, center_sensor[2] + dz],
            [center_sensor[0] + dx, center_sensor[1] - dy, center_sensor[2] + dz],
            [center_sensor[0] + dx, center_sensor[1] + dy, center_sensor[2] + dz],
            [center_sensor[0] - dx, center_sensor[1] + dy, center_sensor[2] + dz],
        ],
        dtype=np.float32,
    )
    sensor_velocity = np.array(
        [max(0.0, min(20.0, bbox_width / max(image_width, 1) * 20.0)), 0.0, 0.0],
        dtype=np.float32,
    )
    sensor_offset, sensor_yaw_rad = _sensor_mount(sensor_id)
    center_vehicle = _rotate_xy(center_sensor, sensor_yaw_rad) + sensor_offset
    world_center = _rotate_xy(center_vehicle, ego_yaw_rad) + np.asarray(ego_world_xyz, dtype=np.float32)
    world_bbox = np.asarray(
        [
            _rotate_xy(_rotate_xy(corner, sensor_yaw_rad) + sensor_offset, ego_yaw_rad)
            + np.asarray(ego_world_xyz, dtype=np.float32)
            for corner in sensor_bbox
        ],
        dtype=np.float32,
    )
    world_velocity = _rotate_xy(_rotate_xy(sensor_velocity, sensor_yaw_rad), ego_yaw_rad)
    return world_bbox, world_velocity, world_center.astype(np.float32), position_uncertainty_m


@dataclass(slots=True)
class BootstrapAnnotation:
    track_id: int
    object_class: ObjectClass
    confidence: float
    image_bbox_xyxy: np.ndarray
    world_bbox_3d: np.ndarray
    velocity_xyz: np.ndarray
    world_xyz: np.ndarray
    traffic_light_state: TrafficLightState | None = None


@dataclass(slots=True)
class CameraInferenceRequest:
    frame: np.ndarray
    bootstrap_annotations: list[BootstrapAnnotation]
    sensor_id: str = "front_camera"
    ego_world_xyz: np.ndarray | None = None
    ego_yaw_rad: float = 0.0
    max_input_long_edge_px: int | None = None
    allowed_classes: tuple[ObjectClass, ...] | None = None
    min_confidence: float = 0.0
    min_bbox_area_px: float = 0.0


def _resize_image(frame: np.ndarray, max_input_long_edge_px: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_input_long_edge_px is None or max_input_long_edge_px <= 0:
        return frame, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    height, width = frame.shape[:2]
    current_long_edge = max(height, width)
    if current_long_edge <= max_input_long_edge_px:
        return frame, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    scale = max_input_long_edge_px / float(current_long_edge)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    except ImportError:  # pragma: no cover
        y_indices = np.linspace(0, height - 1, resized_height).astype(np.int32)
        x_indices = np.linspace(0, width - 1, resized_width).astype(np.int32)
        resized = frame[np.ix_(y_indices, x_indices)]
    scale_back = np.array(
        [
            width / float(resized_width),
            height / float(resized_height),
            width / float(resized_width),
            height / float(resized_height),
        ],
        dtype=np.float32,
    )
    return resized, scale_back


class YoloObjectDetector:
    """TODO(PRD 3.2.3): swap this bootstrap-first adapter for the user's trained YOLO model config."""

    def __init__(self, *, model_variant: str, device: str) -> None:
        self.model_variant = model_variant
        self.device = device
        self._model: Any | None = None
        self._load_error: str | None = None
        self.last_inference_ms_total: float = 0.0
        self.last_inference_ms_by_sensor: dict[str, float] = {}

    def detect(
        self,
        frame: np.ndarray,
        bootstrap_annotations: list[BootstrapAnnotation],
        *,
        sensor_id: str = "front_camera",
        ego_world_xyz: np.ndarray | None = None,
        ego_yaw_rad: float = 0.0,
    ) -> tuple[list[FrameDetection2D], str]:
        detections_by_sensor, detector_modes = self.detect_batch(
            [
                CameraInferenceRequest(
                    frame=frame,
                    bootstrap_annotations=bootstrap_annotations,
                    sensor_id=sensor_id,
                    ego_world_xyz=ego_world_xyz,
                    ego_yaw_rad=ego_yaw_rad,
                )
            ]
        )
        return detections_by_sensor.get(sensor_id, []), detector_modes.get(sensor_id, "bootstrap")

    def detect_batch(
        self,
        requests: Sequence[CameraInferenceRequest],
    ) -> tuple[dict[str, list[FrameDetection2D]], dict[str, str]]:
        detections_by_sensor: dict[str, list[FrameDetection2D]] = {}
        detector_modes: dict[str, str] = {}
        self.last_inference_ms_total = 0.0
        self.last_inference_ms_by_sensor = {}
        if not requests:
            return detections_by_sensor, detector_modes
        if self._uses_explicit_bootstrap():
            _logger.warning("Camera detector using bootstrap (ground-truth) mode: explicit config")
            return self._bootstrap_batch(requests)
        if self._model is None and self._load_error is None:
            try:
                self._model = self._load_model()
            except Exception as exc:  # pragma: no cover - depends on optional runtime deps
                self._load_error = str(exc)
                _logger.warning("Camera detector falling back to bootstrap: model load failed (%s)", exc)
                return self._bootstrap_batch(requests)
        if self._model is None:
            _logger.warning("Camera detector falling back to bootstrap: no model available (prior load error: %s)", self._load_error)
            return self._bootstrap_batch(requests)

        prepared_requests: list[tuple[CameraInferenceRequest, np.ndarray, np.ndarray]] = []
        sources: list[np.ndarray] = []
        for request in requests:
            resized_frame, scale_back = _resize_image(request.frame, request.max_input_long_edge_px)
            prepared_requests.append((request, resized_frame, scale_back))
            sources.append(resized_frame.astype(np.uint8))

        try:
            start = perf_counter()
            results = self._model.predict(
                source=sources if len(sources) > 1 else sources[0],
                device=self.device,
                verbose=False,
            )
            self.last_inference_ms_total = max(0.0, (perf_counter() - start) * 1000.0)
        except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover - depends on optional runtime deps
            _logger.warning("Camera detector falling back to bootstrap: YOLO predict failed (%s)", exc)
            return self._bootstrap_batch(requests)

        normalized_results = self._normalize_results(results)
        if len(normalized_results) != len(prepared_requests):
            _logger.warning("Camera detector falling back to bootstrap: result count mismatch (%d vs %d)", len(normalized_results), len(prepared_requests))
            return self._bootstrap_batch(requests)

        inference_share = self.last_inference_ms_total / float(max(len(prepared_requests), 1))
        for (request, resized_frame, scale_back), result in zip(prepared_requests, normalized_results):
            detections_by_sensor[request.sensor_id] = self._detections_from_result(
                result,
                request=request,
                scale_back=scale_back,
            )
            detector_modes[request.sensor_id] = "camera"
            self.last_inference_ms_by_sensor[request.sensor_id] = inference_share
        return detections_by_sensor, detector_modes

    def _normalize_results(self, results: Any) -> list[Any]:
        if results is None:
            return []
        if isinstance(results, list):
            return list(results)
        if isinstance(results, tuple):
            return list(results)
        return [results]

    def _bootstrap_batch(
        self,
        requests: Sequence[CameraInferenceRequest],
    ) -> tuple[dict[str, list[FrameDetection2D]], dict[str, str]]:
        detections_by_sensor: dict[str, list[FrameDetection2D]] = {}
        detector_modes: dict[str, str] = {}
        for request in requests:
            detections_by_sensor[request.sensor_id] = self._from_bootstrap(
                request.bootstrap_annotations,
                sensor_id=request.sensor_id,
            )
            detector_modes[request.sensor_id] = "bootstrap"
            self.last_inference_ms_by_sensor[request.sensor_id] = 0.0
        self.last_inference_ms_total = 0.0
        return detections_by_sensor, detector_modes

    def _detections_from_result(
        self,
        result: Any,
        *,
        request: CameraInferenceRequest,
        scale_back: np.ndarray,
    ) -> list[FrameDetection2D]:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        detections: list[FrameDetection2D] = []
        if boxes is None:
            return detections
        ego_xyz = np.asarray(
            np.zeros(3, dtype=np.float32) if request.ego_world_xyz is None else request.ego_world_xyz,
            dtype=np.float32,
        )
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            label = str(names.get(cls_idx, cls_idx))
            object_class = _object_class_from_label(label)
            if object_class is None:
                continue
            if request.allowed_classes is not None and object_class not in request.allowed_classes:
                continue
            confidence = float(box.conf[0].item())
            if confidence < request.min_confidence:
                continue
            bbox_xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32) * scale_back
            bbox_area = max(0.0, float(bbox_xyxy[2] - bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
            if bbox_area < request.min_bbox_area_px:
                continue
            world_bbox, velocity_xyz, world_xyz, uncertainty_m = _pseudo_world_box(
                bbox_xyxy,
                object_class,
                request.frame.shape,
                sensor_id=request.sensor_id,
                ego_world_xyz=ego_xyz,
                ego_yaw_rad=request.ego_yaw_rad,
            )
            detections.append(
                FrameDetection2D(
                    bbox_xyxy=bbox_xyxy,
                    object_class=object_class,
                    confidence=confidence,
                    source_sensor_id=request.sensor_id,
                    source_modality="camera",
                    source_sensor_ids=[request.sensor_id],
                    position_estimate_kind="camera_projection",
                    world_bbox_3d=world_bbox,
                    velocity_xyz=velocity_xyz,
                    world_xyz=world_xyz,
                    position_uncertainty_m=uncertainty_m,
                    traffic_light_state=_traffic_light_state_from_label(label),
                )
            )
        return detections

    def _uses_explicit_bootstrap(self) -> bool:
        return self.model_variant.strip().lower() == "bootstrap"

    def _candidate_variants(self) -> list[str]:
        variant = self.model_variant.strip()
        if variant.lower() in {"", "default", "auto", "none"}:
            return ["yolo11n.pt", "yolov8n.pt"]
        return [variant]

    def _load_model(self):
        from ultralytics import YOLO  # type: ignore

        last_error: Exception | None = None
        for candidate in self._candidate_variants():
            try:
                return YOLO(candidate)
            except Exception as exc:  # pragma: no cover - depends on optional runtime deps
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("no YOLO model candidates were available")

    def _from_bootstrap(
        self,
        annotations: list[BootstrapAnnotation],
        *,
        sensor_id: str,
    ) -> list[FrameDetection2D]:
        return [
            FrameDetection2D(
                bbox_xyxy=np.asarray(annotation.image_bbox_xyxy, dtype=np.float32),
                object_class=annotation.object_class,
                confidence=float(annotation.confidence),
                source_sensor_id=sensor_id,
                source_modality="bootstrap",
                source_sensor_ids=[sensor_id],
                position_estimate_kind="truth_fallback",
                world_bbox_3d=np.asarray(annotation.world_bbox_3d, dtype=np.float32),
                velocity_xyz=np.asarray(annotation.velocity_xyz, dtype=np.float32),
                world_xyz=np.asarray(annotation.world_xyz, dtype=np.float32),
                preferred_track_id=int(annotation.track_id),
                traffic_light_state=annotation.traffic_light_state,
            )
            for annotation in annotations
        ]


def bootstrap_annotations_from_metadata(
    metadata: dict[str, Any],
    *,
    sensor_id: str = "front_camera",
) -> list[BootstrapAnnotation]:
    annotations: list[BootstrapAnnotation] = []
    camera_annotations = metadata.get("carla_camera_annotations", {})
    if isinstance(camera_annotations, dict) and sensor_id in camera_annotations:
        source_annotations = camera_annotations.get(sensor_id, [])
    else:
        source_annotations = metadata.get("carla_actor_annotations", [])
    for annotation in source_annotations:
        label = annotation.get("object_class")
        object_class = label if isinstance(label, ObjectClass) else ObjectClass(str(label))
        traffic_state = annotation.get("traffic_light_state")
        traffic_light_state = (
            None if traffic_state is None else TrafficLightState(str(traffic_state))
        )
        annotations.append(
            BootstrapAnnotation(
                track_id=int(annotation["track_id"]),
                object_class=object_class,
                confidence=float(annotation.get("confidence", 1.0)),
                image_bbox_xyxy=np.asarray(annotation["image_bbox_xyxy"], dtype=np.float32),
                world_bbox_3d=np.asarray(annotation["world_bbox_3d"], dtype=np.float32),
                velocity_xyz=np.asarray(annotation.get("velocity_xyz", [0.0, 0.0, 0.0]), dtype=np.float32),
                world_xyz=np.asarray(annotation.get("world_xyz", [0.0, 0.0, 0.0]), dtype=np.float32),
                traffic_light_state=traffic_light_state,
            )
        )
    return annotations
