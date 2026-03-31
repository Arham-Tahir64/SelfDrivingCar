from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from autonomy_demo.interfaces.enums import TrackState
from autonomy_demo.perception.internal_types import LidarClusterDetection, TrackedLidarClusterDetection


def _distance_xy(point_a: np.ndarray, point_b: np.ndarray) -> float:
    delta = np.asarray(point_a, dtype=np.float32)[:2] - np.asarray(point_b, dtype=np.float32)[:2]
    return float(np.linalg.norm(delta))


class _KalmanTracker3D:
    """Per-object Kalman filter with constant-velocity model in 3D world space.

    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """

    DIM_X = 6
    DIM_Z = 3

    def __init__(self, centroid_xyz: np.ndarray, dt: float = 0.05) -> None:
        self.dt = dt
        self.x = np.zeros(self.DIM_X, dtype=np.float64)
        self.x[:3] = np.asarray(centroid_xyz, dtype=np.float64)[:3]

        self.F = np.eye(self.DIM_X, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 3] = dt

        self.H = np.zeros((self.DIM_Z, self.DIM_X), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)

        self.P = np.eye(self.DIM_X, dtype=np.float64)
        self.P[:3, :3] *= 1.0
        self.P[3:, 3:] *= 50.0  # high uncertainty on initial velocity

        self.Q = np.eye(self.DIM_X, dtype=np.float64)
        self.Q[:3, :3] *= 0.5
        self.Q[3:, 3:] *= 1.0

        self.R = np.eye(self.DIM_Z, dtype=np.float64) * 0.5

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].astype(np.float32)

    def update(self, centroid_xyz: np.ndarray) -> None:
        z = np.asarray(centroid_xyz, dtype=np.float64)[:3]
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.DIM_X) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].astype(np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].astype(np.float32)


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: LidarClusterDetection
    hits: int
    age: int
    missed: int
    last_timestamp_s: float
    velocity_xyz: np.ndarray


@dataclass(slots=True)
class _KalmanTrack3D:
    track_id: int
    detection: LidarClusterDetection
    kf: _KalmanTracker3D
    hits: int
    age: int
    missed: int
    last_timestamp_s: float


class SimpleCentroidTracker3D:
    """Legacy deterministic tracker for LiDAR clusters based on XY centroid matching."""

    def __init__(
        self,
        *,
        max_match_distance_m: float = 4.0,
        max_missed: int = 4,
        confirm_hits: int = 2,
    ) -> None:
        self.max_match_distance_m = max_match_distance_m
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self,
        detections: list[LidarClusterDetection],
        *,
        timestamp_s: float,
    ) -> list[TrackedLidarClusterDetection]:
        matched_track_ids: set[int] = set()
        outputs: list[TrackedLidarClusterDetection] = []

        for detection in detections:
            best_track: _Track | None = None
            best_distance = float("inf")
            for track in self._tracks.values():
                if track.track_id in matched_track_ids:
                    continue
                if track.detection.object_class != detection.object_class:
                    continue
                distance_m = _distance_xy(track.detection.centroid_xyz, detection.centroid_xyz)
                if distance_m < best_distance:
                    best_distance = distance_m
                    best_track = track
            if best_track is not None and best_distance <= self.max_match_distance_m:
                dt_s = max(timestamp_s - best_track.last_timestamp_s, 1e-3)
                velocity_xyz = (
                    (np.asarray(detection.centroid_xyz, dtype=np.float32) - best_track.detection.centroid_xyz) / dt_s
                ).astype(np.float32)
                best_track.detection = detection
                best_track.hits += 1
                best_track.age += 1
                best_track.missed = 0
                best_track.last_timestamp_s = timestamp_s
                best_track.velocity_xyz = velocity_xyz
                matched_track_ids.add(best_track.track_id)
                outputs.append(self._to_output(best_track))
                continue

            track = _Track(
                track_id=self._next_track_id,
                detection=detection,
                hits=1,
                age=1,
                missed=0,
                last_timestamp_s=timestamp_s,
                velocity_xyz=np.zeros(3, dtype=np.float32),
            )
            self._tracks[track.track_id] = track
            self._next_track_id += 1
            matched_track_ids.add(track.track_id)
            outputs.append(self._to_output(track))

        to_delete: list[int] = []
        for track in self._tracks.values():
            if track.track_id in matched_track_ids:
                continue
            track.age += 1
            track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(track.track_id)
        for track_id in to_delete:
            self._tracks.pop(track_id, None)

        return outputs

    def _to_output(self, track: _Track) -> TrackedLidarClusterDetection:
        state = TrackState.CONFIRMED if track.hits >= self.confirm_hits else TrackState.TENTATIVE
        detection = track.detection
        return TrackedLidarClusterDetection(
            centroid_xyz=np.asarray(detection.centroid_xyz, dtype=np.float32),
            world_bbox_3d=np.asarray(detection.world_bbox_3d, dtype=np.float32),
            object_class=detection.object_class,
            confidence=float(detection.confidence),
            point_count=int(detection.point_count),
            source_modality=detection.source_modality,
            source_sensor_ids=list(detection.source_sensor_ids),
            position_estimate_kind=detection.position_estimate_kind,
            track_id=track.track_id,
            track_state=state,
            velocity_xyz=np.asarray(track.velocity_xyz, dtype=np.float32),
        )


class KalmanCentroidTracker3D:
    """Kalman-filtered 3D tracker with Hungarian assignment on centroid distance.

    Each track maintains a 6-state Kalman filter [x, y, z, vx, vy, vz].
    Association uses the Hungarian algorithm on predicted-to-detection distance.
    """

    def __init__(
        self,
        *,
        max_match_distance_m: float = 4.0,
        max_missed: int = 4,
        confirm_hits: int = 2,
    ) -> None:
        self.max_match_distance_m = max_match_distance_m
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self._next_track_id = 1
        self._tracks: dict[int, _KalmanTrack3D] = {}
        self.id_switches = 0

    def update(
        self,
        detections: list[LidarClusterDetection],
        *,
        timestamp_s: float,
    ) -> list[TrackedLidarClusterDetection]:
        # --- 1. Predict all existing tracks ---
        predicted_positions: dict[int, np.ndarray] = {}
        for tid, track in self._tracks.items():
            dt = max(timestamp_s - track.last_timestamp_s, 1e-3)
            track.kf.F[0, 3] = dt
            track.kf.F[1, 4] = dt
            track.kf.F[2, 5] = dt
            predicted_positions[tid] = track.kf.predict()

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        outputs: list[TrackedLidarClusterDetection] = []

        track_list = [(tid, t) for tid, t in self._tracks.items()]

        # --- 2. Hungarian matching on distance ---
        if detections and track_list:
            cost_matrix = np.full(
                (len(detections), len(track_list)), 1e6, dtype=np.float64
            )
            for di, det in enumerate(detections):
                for ti, (tid, track) in enumerate(track_list):
                    if track.detection.object_class != det.object_class:
                        continue
                    cost_matrix[di, ti] = _distance_xy(
                        det.centroid_xyz, predicted_positions[tid]
                    )

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > self.max_match_distance_m:
                    continue
                det = detections[row]
                tid, track = track_list[col]
                track.kf.update(det.centroid_xyz)
                track.detection = det
                track.hits += 1
                track.age += 1
                track.missed = 0
                track.last_timestamp_s = timestamp_s
                matched_track_ids.add(tid)
                matched_det_indices.add(row)
                outputs.append(self._to_output(track))

        # --- 3. Create new tracks for unmatched detections ---
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue
            new_track = _KalmanTrack3D(
                track_id=self._next_track_id,
                detection=det,
                kf=_KalmanTracker3D(det.centroid_xyz),
                hits=1,
                age=1,
                missed=0,
                last_timestamp_s=timestamp_s,
            )
            self._tracks[new_track.track_id] = new_track
            self._next_track_id += 1
            outputs.append(self._to_output(new_track))

        # --- 4. Age unmatched tracks ---
        to_delete: list[int] = []
        for tid, track in self._tracks.items():
            if tid in matched_track_ids:
                continue
            if any(tid == t.track_id for t in [nt for nt in self._tracks.values() if nt.age == 1 and nt.missed == 0]):
                continue
            track.age += 1
            track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            self._tracks.pop(tid, None)

        return outputs

    def _to_output(self, track: _KalmanTrack3D) -> TrackedLidarClusterDetection:
        state = TrackState.CONFIRMED if track.hits >= self.confirm_hits else TrackState.TENTATIVE
        det = track.detection
        return TrackedLidarClusterDetection(
            centroid_xyz=track.kf.position,
            world_bbox_3d=np.asarray(det.world_bbox_3d, dtype=np.float32),
            object_class=det.object_class,
            confidence=float(det.confidence),
            point_count=int(det.point_count),
            source_modality=det.source_modality,
            source_sensor_ids=list(det.source_sensor_ids),
            position_estimate_kind=det.position_estimate_kind,
            track_id=track.track_id,
            track_state=state,
            velocity_xyz=track.kf.velocity,
        )
