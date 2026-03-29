from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autonomy_demo.interfaces.enums import TrackState
from autonomy_demo.perception.internal_types import LidarClusterDetection, TrackedLidarClusterDetection


def _distance_xy(point_a: np.ndarray, point_b: np.ndarray) -> float:
    delta = np.asarray(point_a, dtype=np.float32)[:2] - np.asarray(point_b, dtype=np.float32)[:2]
    return float(np.linalg.norm(delta))


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: LidarClusterDetection
    hits: int
    age: int
    missed: int
    last_timestamp_s: float
    velocity_xyz: np.ndarray


class SimpleCentroidTracker3D:
    """Small deterministic tracker for LiDAR clusters based on XY centroid matching."""

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
            track_id=track.track_id,
            track_state=state,
            velocity_xyz=np.asarray(track.velocity_xyz, dtype=np.float32),
        )
