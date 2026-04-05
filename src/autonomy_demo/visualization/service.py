from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.interfaces.enums import TopicName
from autonomy_demo.interfaces.types import CameraFrame, ControlCommand, DrivableSpaceMask, EgoPose, EgoTrajectory, LaneLine, LocalMap, ObjectDetection, SemanticSegMap, TrafficLightDetection


def _has_image_bbox(bbox_xyxy: np.ndarray | None) -> bool:
    return bbox_xyxy is not None

def _is_camera_grounded(position_estimate_kind: str) -> bool:
    return str(position_estimate_kind) == "camera_projection"


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
        self._latest_semantic_seg: SemanticSegMap | None = None
        self._latest_ego_pose: EgoPose | None = None
        self._latest_local_map: LocalMap | None = None
        self._latest_trajectory: EgoTrajectory | None = None
        self._latest_command: ControlCommand | None = None
        self._latest_overlay: np.ndarray | None = None
        self._event_bus: Any = None

    def attach(self, event_bus) -> None:
        if not self.enabled:
            return
        self._event_bus = event_bus
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
        elif topic == TopicName.PERCEPTION_SEMANTIC_SEG.value and isinstance(payload, SemanticSegMap):
            self._latest_semantic_seg = payload
        elif topic == TopicName.LOCALIZATION_EGO_POSE.value and isinstance(payload, EgoPose):
            self._latest_ego_pose = payload
        elif topic == TopicName.MAP_LOCAL_MAP.value and isinstance(payload, LocalMap):
            self._latest_local_map = payload
        elif topic == TopicName.PLANNING_EGO_TRAJECTORY.value and isinstance(payload, EgoTrajectory):
            self._latest_trajectory = payload
        elif topic == TopicName.CONTROL_VEHICLE_COMMAND.value and isinstance(payload, ControlCommand):
            self._latest_command = payload
            self._latest_overlay = self._render_overlay()
            if self._latest_overlay is not None and self._event_bus is not None:
                self._event_bus.publish(
                    TopicName.VISUALIZATION_CAMERA_OVERLAY.value,
                    self._latest_overlay,
                )

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
        if self._latest_semantic_seg is not None and self._latest_semantic_seg.label_map.shape[:2] == frame.shape[:2]:
            # Full semantic segmentation overlay
            try:
                from autonomy_demo.perception.cityscapes_palette import CITYSCAPES_PALETTE
                color_map = CITYSCAPES_PALETTE[self._latest_semantic_seg.label_map]  # (H, W, 3)
                alpha = 0.45
                frame = ((1.0 - alpha) * frame + alpha * color_map).astype(np.uint8)
            except Exception:
                pass
        elif self._latest_drivable is not None and self._latest_drivable.mask.shape[:2] == frame.shape[:2]:
            # Fallback: binary green drivable mask
            overlay = frame.copy()
            overlay[self._latest_drivable.mask] = (
                0.7 * overlay[self._latest_drivable.mask] + np.array([0, 60, 0], dtype=np.uint8)
            )
            frame = overlay.astype(np.uint8)
        for detection in self._latest_detections:
            if not _has_image_bbox(detection.image_bbox_xyxy):
                continue
            x1, y1, x2, y2 = detection.image_bbox_xyxy.astype(int).tolist()
            is_ml = _is_camera_grounded(detection.position_estimate_kind)
            box_color = (0, 255, 255) if is_ml else (120, 180, 180)
            thickness = 2 if is_ml else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            label = f"{detection.object_class.value}:{detection.track_id}"
            if not is_ml:
                label += " [gt]"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA,
            )
        for lane in self._latest_lanes:
            polyline = lane.polyline_image.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [polyline], False, (255, 255, 0), 2)
        for traffic_light in self._latest_traffic_lights:
            if not _has_image_bbox(traffic_light.image_bbox_xyxy):
                continue
            x1, y1, x2, y2 = traffic_light.image_bbox_xyxy.astype(int).tolist()
            color = {
                "RED": (255, 0, 0),
                "AMBER": (255, 200, 0),
                "GREEN": (0, 255, 0),
            }.get(traffic_light.state.value, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if self._latest_ego_pose is not None:
            overlay_lines = [
                f"lane: {self._latest_ego_pose.current_lane_id}",
                f"s: {self._latest_ego_pose.frenet_s:.1f} m",
                f"d: {self._latest_ego_pose.frenet_d:.2f} m",
                f"heading err: {self._latest_ego_pose.heading_error_rad:.2f} rad",
            ]
            if self._latest_trajectory is not None:
                target_speed = max((waypoint.velocity for waypoint in self._latest_trajectory.waypoints), default=0.0)
                overlay_lines.append(f"behavior: {self._latest_trajectory.behavior_state.value}")
                overlay_lines.append(f"target speed: {target_speed:.2f} m/s")
            if self._latest_command is not None:
                overlay_lines.append(
                    f"emergency: {'ON' if self._latest_command.emergency_override else 'off'}"
                )
            if self._latest_local_map is not None:
                if self._latest_local_map.closed_lanes:
                    overlay_lines.append(
                        f"closed lane cue: {self._latest_local_map.closed_lanes[0]}"
                    )
                if self._latest_local_map.temporary_boundaries:
                    overlay_lines.append(
                        f"temporary boundaries: {len(self._latest_local_map.temporary_boundaries)}"
                    )
            for index, text in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    text,
                    (16, 24 + (index * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
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
