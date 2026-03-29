from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autonomy_demo.interfaces.enums import TrackState
from autonomy_demo.perception.internal_types import FrameDetection2D, TrackedDetection2D


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: FrameDetection2D
    hits: int
    age: int
    missed: int


class SimpleSortTracker:
    """TODO(PRD 3.2.3): replace with a full SORT/DeepSORT-style tracker with motion filtering."""

    def __init__(self, *, iou_threshold: float = 0.2, max_missed: int = 5, confirm_hits: int = 2) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[FrameDetection2D]) -> list[TrackedDetection2D]:
        matched_track_ids: set[int] = set()
        outputs: list[TrackedDetection2D] = []

        for detection in detections:
            if detection.preferred_track_id is not None and detection.preferred_track_id in self._tracks:
                track = self._tracks[detection.preferred_track_id]
                track.detection = detection
                track.hits += 1
                track.age += 1
                track.missed = 0
                matched_track_ids.add(track.track_id)
                outputs.append(self._to_output(track))
                continue

            best_track: _Track | None = None
            best_iou = 0.0
            for track in self._tracks.values():
                if track.track_id in matched_track_ids:
                    continue
                if track.detection.object_class != detection.object_class:
                    continue
                overlap = _iou(track.detection.bbox_xyxy, detection.bbox_xyxy)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track
            if best_track is not None and best_iou >= self.iou_threshold:
                best_track.detection = detection
                best_track.hits += 1
                best_track.age += 1
                best_track.missed = 0
                matched_track_ids.add(best_track.track_id)
                outputs.append(self._to_output(best_track))
                continue

            new_track_id = detection.preferred_track_id or self._next_track_id
            self._next_track_id = max(self._next_track_id, new_track_id + 1)
            new_track = _Track(
                track_id=new_track_id,
                detection=detection,
                hits=1,
                age=1,
                missed=0,
            )
            self._tracks[new_track.track_id] = new_track
            matched_track_ids.add(new_track.track_id)
            outputs.append(self._to_output(new_track))

        deleted_track_ids: list[int] = []
        for track in self._tracks.values():
            if track.track_id in matched_track_ids:
                continue
            track.age += 1
            track.missed += 1
            if track.missed > self.max_missed:
                deleted_track_ids.append(track.track_id)
        for track_id in deleted_track_ids:
            self._tracks.pop(track_id, None)

        return outputs

    def _to_output(self, track: _Track) -> TrackedDetection2D:
        state = TrackState.CONFIRMED if track.hits >= self.confirm_hits else TrackState.TENTATIVE
        detection = track.detection
        return TrackedDetection2D(
            bbox_xyxy=np.asarray(detection.bbox_xyxy, dtype=np.float32),
            object_class=detection.object_class,
            confidence=float(detection.confidence),
            source_sensor_id=detection.source_sensor_id,
            world_bbox_3d=None if detection.world_bbox_3d is None else np.asarray(detection.world_bbox_3d, dtype=np.float32),
            velocity_xyz=None if detection.velocity_xyz is None else np.asarray(detection.velocity_xyz, dtype=np.float32),
            world_xyz=None if detection.world_xyz is None else np.asarray(detection.world_xyz, dtype=np.float32),
            preferred_track_id=detection.preferred_track_id,
            traffic_light_state=detection.traffic_light_state,
            track_id=track.track_id,
            track_state=state,
        )
