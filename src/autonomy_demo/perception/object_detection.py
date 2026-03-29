from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autonomy_demo.interfaces.enums import ObjectClass, TrafficLightState
from autonomy_demo.perception.internal_types import FrameDetection2D


_YOLO_CLASS_MAP = {
    "car": ObjectClass.VEHICLE,
    "truck": ObjectClass.VEHICLE,
    "bus": ObjectClass.VEHICLE,
    "motorcycle": ObjectClass.CYCLIST,
    "bicycle": ObjectClass.CYCLIST,
    "person": ObjectClass.PEDESTRIAN,
    "pedestrian": ObjectClass.PEDESTRIAN,
    "traffic_light": ObjectClass.TRAFFIC_LIGHT,
    "traffic light": ObjectClass.TRAFFIC_LIGHT,
    "traffic-light": ObjectClass.TRAFFIC_LIGHT,
}


def _object_class_from_label(label: str) -> ObjectClass | None:
    return _YOLO_CLASS_MAP.get(label.strip().lower())


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_height, image_width = image_shape[:2]
    bbox_width = max(float(bbox_xyxy[2] - bbox_xyxy[0]), 1.0)
    bbox_height = max(float(bbox_xyxy[3] - bbox_xyxy[1]), 1.0)
    center_x = float((bbox_xyxy[0] + bbox_xyxy[2]) * 0.5)
    normalized_x = ((center_x / max(image_width, 1)) - 0.5) * 2.0
    forward_distance = float(np.clip(1200.0 / bbox_height, 4.0, 60.0))
    lateral_offset = float(normalized_x * forward_distance * 0.8)
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
    return world_bbox, world_velocity, world_center.astype(np.float32)


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


class YoloObjectDetector:
    """TODO(PRD 3.2.3): swap this bootstrap-first adapter for the user's trained YOLO model config."""

    def __init__(self, *, model_variant: str, device: str) -> None:
        self.model_variant = model_variant
        self.device = device
        self._model: Any | None = None
        self._load_error: str | None = None

    def detect(
        self,
        frame: np.ndarray,
        bootstrap_annotations: list[BootstrapAnnotation],
        *,
        sensor_id: str = "front_camera",
        ego_world_xyz: np.ndarray | None = None,
        ego_yaw_rad: float = 0.0,
    ) -> tuple[list[FrameDetection2D], str]:
        if self._uses_explicit_bootstrap():
            return self._from_bootstrap(bootstrap_annotations, sensor_id=sensor_id), "bootstrap"
        if self._model is None and self._load_error is None:
            try:
                self._model = self._load_model()
            except Exception as exc:  # pragma: no cover - depends on optional runtime deps
                self._load_error = str(exc)
                return self._from_bootstrap(bootstrap_annotations, sensor_id=sensor_id), "bootstrap"
        if self._model is None:
            return self._from_bootstrap(bootstrap_annotations, sensor_id=sensor_id), "bootstrap"
        try:
            results = self._model.predict(
                source=frame.astype(np.uint8),
                device=self.device,
                verbose=False,
            )
        except Exception:  # pragma: no cover - depends on optional runtime deps
            return self._from_bootstrap(bootstrap_annotations, sensor_id=sensor_id), "bootstrap"
        if not results:
            return [], "camera"
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        detections: list[FrameDetection2D] = []
        if boxes is None:
            return [], "camera"
        ego_xyz = np.asarray(
            np.zeros(3, dtype=np.float32) if ego_world_xyz is None else ego_world_xyz,
            dtype=np.float32,
        )
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            label = str(names.get(cls_idx, cls_idx))
            object_class = _object_class_from_label(label)
            if object_class is None:
                continue
            bbox_xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
            confidence = float(box.conf[0].item())
            world_bbox, velocity_xyz, world_xyz = _pseudo_world_box(
                bbox_xyxy,
                object_class,
                frame.shape,
                sensor_id=sensor_id,
                ego_world_xyz=ego_xyz,
                ego_yaw_rad=ego_yaw_rad,
            )
            detections.append(
                FrameDetection2D(
                    bbox_xyxy=bbox_xyxy,
                    object_class=object_class,
                    confidence=confidence,
                    source_sensor_id=sensor_id,
                    source_modality="camera",
                    source_sensor_ids=[sensor_id],
                    position_estimate_kind="camera_projection",
                    world_bbox_3d=world_bbox,
                    velocity_xyz=velocity_xyz,
                    world_xyz=world_xyz,
                )
            )
        return detections, "camera"

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
