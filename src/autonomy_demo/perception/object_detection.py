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


def _pseudo_world_box(
    bbox_xyxy: np.ndarray,
    object_class: ObjectClass,
    image_shape: tuple[int, int, int],
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
    center = np.array(
        [forward_distance, lateral_offset, height_m * 0.5],
        dtype=np.float32,
    )
    dx = length_m * 0.5
    dy = width_m * 0.5
    dz = height_m * 0.5
    world_bbox = np.array(
        [
            [center[0] - dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] + dy, center[2] + dz],
            [center[0] - dx, center[1] + dy, center[2] + dz],
        ],
        dtype=np.float32,
    )
    velocity = np.array(
        [max(0.0, min(20.0, bbox_width / max(image_width, 1) * 20.0)), 0.0, 0.0],
        dtype=np.float32,
    )
    return world_bbox, velocity, center


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

    def detect(self, frame: np.ndarray, bootstrap_annotations: list[BootstrapAnnotation]) -> list[FrameDetection2D]:
        if self.model_variant.strip().lower() in {"", "bootstrap", "none"}:
            return self._from_bootstrap(bootstrap_annotations)
        if self._model is None and self._load_error is None:
            try:
                from ultralytics import YOLO  # type: ignore

                self._model = YOLO(self.model_variant)
            except Exception as exc:  # pragma: no cover - depends on optional runtime deps
                self._load_error = str(exc)
                return self._from_bootstrap(bootstrap_annotations)
        if self._model is None:
            return self._from_bootstrap(bootstrap_annotations)
        try:
            results = self._model.predict(
                source=frame.astype(np.uint8),
                device=self.device,
                verbose=False,
            )
        except Exception:  # pragma: no cover - depends on optional runtime deps
            return self._from_bootstrap(bootstrap_annotations)
        if not results:
            return self._from_bootstrap(bootstrap_annotations)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        detections: list[FrameDetection2D] = []
        if boxes is None:
            return self._from_bootstrap(bootstrap_annotations)
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            label = str(names.get(cls_idx, cls_idx))
            object_class = _object_class_from_label(label)
            if object_class is None:
                continue
            bbox_xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
            confidence = float(box.conf[0].item())
            matched = self._match_bootstrap(bbox_xyxy, object_class, bootstrap_annotations)
            if matched is not None:
                detections.append(
                    FrameDetection2D(
                        bbox_xyxy=bbox_xyxy,
                        object_class=object_class,
                        confidence=confidence,
                        world_bbox_3d=matched.world_bbox_3d,
                        velocity_xyz=matched.velocity_xyz,
                        world_xyz=matched.world_xyz,
                        preferred_track_id=matched.track_id,
                        traffic_light_state=matched.traffic_light_state,
                    )
                )
                continue
            world_bbox, velocity_xyz, world_xyz = _pseudo_world_box(bbox_xyxy, object_class, frame.shape)
            detections.append(
                FrameDetection2D(
                    bbox_xyxy=bbox_xyxy,
                    object_class=object_class,
                    confidence=confidence,
                    world_bbox_3d=world_bbox,
                    velocity_xyz=velocity_xyz,
                    world_xyz=world_xyz,
                )
            )
        return detections

    def _from_bootstrap(self, annotations: list[BootstrapAnnotation]) -> list[FrameDetection2D]:
        return [
            FrameDetection2D(
                bbox_xyxy=np.asarray(annotation.image_bbox_xyxy, dtype=np.float32),
                object_class=annotation.object_class,
                confidence=float(annotation.confidence),
                world_bbox_3d=np.asarray(annotation.world_bbox_3d, dtype=np.float32),
                velocity_xyz=np.asarray(annotation.velocity_xyz, dtype=np.float32),
                world_xyz=np.asarray(annotation.world_xyz, dtype=np.float32),
                preferred_track_id=int(annotation.track_id),
                traffic_light_state=annotation.traffic_light_state,
            )
            for annotation in annotations
        ]

    def _match_bootstrap(
        self,
        bbox_xyxy: np.ndarray,
        object_class: ObjectClass,
        annotations: list[BootstrapAnnotation],
    ) -> BootstrapAnnotation | None:
        if not annotations:
            return None
        bbox_center = np.array(
            [(bbox_xyxy[0] + bbox_xyxy[2]) * 0.5, (bbox_xyxy[1] + bbox_xyxy[3]) * 0.5],
            dtype=np.float32,
        )
        same_class = [annotation for annotation in annotations if annotation.object_class == object_class]
        if not same_class:
            return None
        return min(
            same_class,
            key=lambda annotation: float(
                np.linalg.norm(
                    bbox_center
                    - np.array(
                        [
                            (annotation.image_bbox_xyxy[0] + annotation.image_bbox_xyxy[2]) * 0.5,
                            (annotation.image_bbox_xyxy[1] + annotation.image_bbox_xyxy[3]) * 0.5,
                        ],
                        dtype=np.float32,
                    )
                )
            ),
        )


def bootstrap_annotations_from_metadata(metadata: dict[str, Any]) -> list[BootstrapAnnotation]:
    annotations: list[BootstrapAnnotation] = []
    for annotation in metadata.get("carla_actor_annotations", []):
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
