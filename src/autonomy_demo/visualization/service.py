from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.interfaces.enums import TopicName
from autonomy_demo.interfaces.types import CameraFrame, DrivableSpaceMask, LaneLine, ObjectDetection, TrafficLightDetection


class NullVisualizationService:
    """Read-only subscriber that captures the latest events without affecting the pipeline."""

    def __init__(self, enabled: bool = True, output_dir: Path | None = None) -> None:
        self.enabled = enabled
        self.logger = get_logger(__name__, enabled=enabled)
        self.events: deque[tuple[str, Any]] = deque(maxlen=32)
        self.output_dir = ensure_directory(output_dir) if output_dir else None
        self._latest_front_camera: CameraFrame | None = None
        self._latest_detections: list[ObjectDetection] = []
        self._latest_lanes: list[LaneLine] = []
        self._latest_traffic_lights: list[TrafficLightDetection] = []
        self._latest_drivable: DrivableSpaceMask | None = None
        self._latest_overlay: np.ndarray | None = None

    def attach(self, event_bus) -> None:
        if not self.enabled:
            return
        event_bus.subscribe("*", self._handle)
        self.logger.info("Visualization subscriber attached")

    def _handle(self, topic: str, payload: Any) -> None:
        self.events.append((topic, payload))
        if topic == TopicName.SENSOR_CAMERA_FRONT.value and isinstance(payload, CameraFrame):
            self._latest_front_camera = payload
        elif topic == TopicName.PERCEPTION_DETECTIONS.value and isinstance(payload, list):
            self._latest_detections = [item for item in payload if isinstance(item, ObjectDetection)]
        elif topic == TopicName.PERCEPTION_LANES.value and isinstance(payload, list):
            self._latest_lanes = [item for item in payload if isinstance(item, LaneLine)]
        elif topic == TopicName.PERCEPTION_TRAFFIC_LIGHTS.value and isinstance(payload, list):
            self._latest_traffic_lights = [item for item in payload if isinstance(item, TrafficLightDetection)]
        elif topic == TopicName.PERCEPTION_DRIVABLE_SPACE.value and isinstance(payload, DrivableSpaceMask):
            self._latest_drivable = payload
        elif topic == TopicName.CONTROL_VEHICLE_COMMAND.value:
            self._latest_overlay = self._render_overlay()

    def flush(self) -> None:
        if self.enabled:
            if self._latest_overlay is not None and self.output_dir is not None:
                self._write_overlay(self._latest_overlay)
            self.logger.info("Visualization captured %s events", len(self.events))

    def _render_overlay(self) -> np.ndarray | None:
        if self._latest_front_camera is None:
            return None
        frame = np.clip(self._latest_front_camera.frame.copy(), 0.0, 255.0).astype(np.uint8)
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - optional dependency path
            return frame
        if self._latest_drivable is not None and self._latest_drivable.mask.shape[:2] == frame.shape[:2]:
            overlay = frame.copy()
            overlay[self._latest_drivable.mask] = (
                0.7 * overlay[self._latest_drivable.mask] + np.array([0, 60, 0], dtype=np.uint8)
            )
            frame = overlay.astype(np.uint8)
        for detection in self._latest_detections:
            if detection.image_bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = detection.image_bbox_xyxy.astype(int).tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"{detection.object_class.value}:{detection.track_id}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        for lane in self._latest_lanes:
            polyline = lane.polyline_image.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [polyline], False, (255, 255, 0), 2)
        for traffic_light in self._latest_traffic_lights:
            if traffic_light.image_bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = traffic_light.image_bbox_xyxy.astype(int).tolist()
            color = {
                "RED": (255, 0, 0),
                "AMBER": (255, 200, 0),
                "GREEN": (0, 255, 0),
            }.get(traffic_light.state.value, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        try:
            cv2.imshow("autonomy_demo/perception", frame[:, :, ::-1])
            cv2.waitKey(1)
        except Exception:  # pragma: no cover - GUI path
            pass
        return frame

    def _write_overlay(self, frame: np.ndarray) -> None:
        path = self.output_dir / "latest_overlay.png"
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - optional dependency path
            return
        cv2.imwrite(str(path), frame[:, :, ::-1])
