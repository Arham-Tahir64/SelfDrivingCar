from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def _xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, aspect_ratio, height]."""
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    cx = float(bbox[0]) + w / 2.0
    cy = float(bbox[1]) + h / 2.0
    aspect = w / max(h, 1e-6)
    return np.array([cx, cy, aspect, h], dtype=np.float64)


def _xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, aspect_ratio, height] back to [x1, y1, x2, y2]."""
    cx, cy, a, h = float(xyah[0]), float(xyah[1]), float(xyah[2]), float(xyah[3])
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class _KalmanBoxTracker:
    """Per-object Kalman filter with constant-velocity model on [cx, cy, a, h] space."""

    # State: [cx, cy, a, h, dcx, dcy, da, dh]
    DIM_X = 8
    DIM_Z = 4

    def __init__(self, bbox_xyxy: np.ndarray) -> None:
        z = _xyxy_to_xyah(bbox_xyxy)
        self.x = np.zeros(self.DIM_X, dtype=np.float64)
        self.x[:4] = z

        # State transition (constant velocity)
        self.F = np.eye(self.DIM_X, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix
        self.H = np.zeros((self.DIM_Z, self.DIM_X), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        # Covariance
        self.P = np.eye(self.DIM_X, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on velocities

        # Process noise
        self.Q = np.eye(self.DIM_X, dtype=np.float64)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(self.DIM_Z, dtype=np.float64)
        self.R[2, 2] *= 10.0  # aspect ratio is noisier

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _xyah_to_xyxy(self.x[:4])

    def update(self, bbox_xyxy: np.ndarray) -> None:
        z = _xyxy_to_xyah(bbox_xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.DIM_X) - K @ self.H) @ self.P

    @property
    def predicted_bbox(self) -> np.ndarray:
        return _xyah_to_xyxy(self.x[:4])


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: FrameDetection2D
    hits: int
    age: int
    missed: int


@dataclass(slots=True)
class _KalmanTrack:
    track_id: int
    detection: FrameDetection2D
    kf: _KalmanBoxTracker
    hits: int
    age: int
    missed: int


class SimpleSortTracker:
    """Legacy greedy IoU tracker kept for backwards compatibility."""

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
            source_modality=detection.source_modality,
            source_sensor_ids=list(detection.source_sensor_ids),
            position_estimate_kind=detection.position_estimate_kind,
            world_bbox_3d=None if detection.world_bbox_3d is None else np.asarray(detection.world_bbox_3d, dtype=np.float32),
            velocity_xyz=None if detection.velocity_xyz is None else np.asarray(detection.velocity_xyz, dtype=np.float32),
            world_xyz=None if detection.world_xyz is None else np.asarray(detection.world_xyz, dtype=np.float32),
            preferred_track_id=detection.preferred_track_id,
            traffic_light_state=detection.traffic_light_state,
            track_id=track.track_id,
            track_state=state,
        )


class KalmanSortTracker:
    """SORT-style tracker: Kalman filter per track + Hungarian assignment on IoU.

    Implements the core SORT algorithm (Bewley et al., 2016):
    1. Predict all existing tracks forward via Kalman filter
    2. Compute IoU cost matrix between predictions and new detections
    3. Solve assignment with the Hungarian algorithm
    4. Update matched tracks, create new tracks, age unmatched tracks
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.2,
        max_missed: int = 5,
        confirm_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self._next_track_id = 1
        self._tracks: dict[int, _KalmanTrack] = {}
        self.id_switches = 0

    def update(self, detections: list[FrameDetection2D]) -> list[TrackedDetection2D]:
        # --- 1. Predict all existing tracks ---
        predicted_bboxes: dict[int, np.ndarray] = {}
        for tid, track in self._tracks.items():
            predicted_bboxes[tid] = track.kf.predict()

        # --- 2. Handle preferred_track_id matches first (GT bootstrap) ---
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        outputs: list[TrackedDetection2D] = []

        for i, det in enumerate(detections):
            if det.preferred_track_id is not None and det.preferred_track_id in self._tracks:
                track = self._tracks[det.preferred_track_id]
                track.kf.update(det.bbox_xyxy)
                track.detection = det
                track.hits += 1
                track.age += 1
                track.missed = 0
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(i)
                outputs.append(self._to_output(track))

        # --- 3. Build IoU cost matrix for remaining ---
        remaining_dets = [(i, detections[i]) for i in range(len(detections)) if i not in matched_det_indices]
        remaining_tracks = [(tid, t) for tid, t in self._tracks.items() if tid not in matched_track_ids]

        if remaining_dets and remaining_tracks:
            cost_matrix = np.zeros((len(remaining_dets), len(remaining_tracks)), dtype=np.float64)
            for di, (_, det) in enumerate(remaining_dets):
                for ti, (tid, track) in enumerate(remaining_tracks):
                    if track.detection.object_class != det.object_class:
                        cost_matrix[di, ti] = 1.0
                    else:
                        cost_matrix[di, ti] = 1.0 - _iou(det.bbox_xyxy, predicted_bboxes[tid])

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                iou_score = 1.0 - cost_matrix[row, col]
                if iou_score < self.iou_threshold:
                    continue
                det_idx, det = remaining_dets[row]
                tid, track = remaining_tracks[col]
                track.kf.update(det.bbox_xyxy)
                track.detection = det
                track.hits += 1
                track.age += 1
                track.missed = 0
                matched_track_ids.add(tid)
                matched_det_indices.add(det_idx)
                outputs.append(self._to_output(track))

        # --- 4. Create new tracks for unmatched detections ---
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue
            new_id = det.preferred_track_id or self._next_track_id
            self._next_track_id = max(self._next_track_id, new_id + 1)
            new_track = _KalmanTrack(
                track_id=new_id,
                detection=det,
                kf=_KalmanBoxTracker(det.bbox_xyxy),
                hits=1,
                age=1,
                missed=0,
            )
            self._tracks[new_id] = new_track
            matched_track_ids.add(new_id)
            outputs.append(self._to_output(new_track))

        # --- 5. Age unmatched tracks, delete stale ones ---
        to_delete: list[int] = []
        for tid, track in self._tracks.items():
            if tid in matched_track_ids:
                continue
            track.age += 1
            track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            self._tracks.pop(tid, None)

        return outputs

    def _to_output(self, track: _KalmanTrack) -> TrackedDetection2D:
        state = TrackState.CONFIRMED if track.hits >= self.confirm_hits else TrackState.TENTATIVE
        det = track.detection
        return TrackedDetection2D(
            bbox_xyxy=track.kf.predicted_bbox,
            object_class=det.object_class,
            confidence=float(det.confidence),
            source_sensor_id=det.source_sensor_id,
            source_modality=det.source_modality,
            source_sensor_ids=list(det.source_sensor_ids),
            position_estimate_kind=det.position_estimate_kind,
            world_bbox_3d=None if det.world_bbox_3d is None else np.asarray(det.world_bbox_3d, dtype=np.float32),
            velocity_xyz=None if det.velocity_xyz is None else np.asarray(det.velocity_xyz, dtype=np.float32),
            world_xyz=None if det.world_xyz is None else np.asarray(det.world_xyz, dtype=np.float32),
            preferred_track_id=det.preferred_track_id,
            traffic_light_state=det.traffic_light_state,
            track_id=track.track_id,
            track_state=state,
        )
