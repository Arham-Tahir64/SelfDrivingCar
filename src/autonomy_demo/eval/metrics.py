from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autonomy_demo.interfaces.types import ObjectDetection


@dataclass(slots=True)
class LatencyAccumulator:
    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, module_name: str, duration_ms: float) -> None:
        if module_name not in self.samples:
            self.samples[module_name] = []
        self.samples[module_name].append(duration_ms)

    def mean(self) -> dict[str, float]:
        return {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in self.samples.items()
        }

    def percentile(self, p: float) -> dict[str, float]:
        return {
            name: float(np.percentile(values, p)) if values else 0.0
            for name, values in self.samples.items()
        }

    def latest(self) -> dict[str, float]:
        return {
            name: values[-1] if values else 0.0
            for name, values in self.samples.items()
        }


@dataclass(slots=True)
class MOTAccumulator:
    """Accumulates per-frame multi-object tracking metrics (MOTA / MOTP).

    Compares tracked detections (with ``track_id``) against ground-truth
    detections (identified by ``preferred_track_id`` from the CARLA backend).
    Matching is done on 3D Euclidean distance between ``world_xyz`` positions.
    """

    match_distance_m: float = 4.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    total_distance: float = 0.0  # sum of matched distances (for MOTP)
    total_gt: int = 0
    _prev_matches: dict[int, int] = field(default_factory=dict)  # gt_id → track_id

    def update(self, detections: list[ObjectDetection]) -> None:
        """Process one frame of tracked detections vs ground-truth.

        Each detection carries ``gt_actor_id`` (CARLA actor ID from bootstrap)
        and ``track_id`` (assigned by the tracker). Comparing these measures
        how well the tracker maintains identity.
        """
        # Build GT map: gt_actor_id → centroid (from detections that have GT labels)
        gt_map: dict[int, np.ndarray] = {}
        # Build tracked map: track_id → centroid
        tracked_map: dict[int, np.ndarray] = {}
        # Map from track_id → gt_actor_id for ID switch detection
        track_to_gt: dict[int, int] = {}

        for det in detections:
            centroid = det.centroid_xyz
            if det.gt_actor_id is not None and det.gt_actor_id >= 0:
                gt_map[det.gt_actor_id] = centroid
            if det.track_id is not None and det.track_id >= 0:
                tracked_map[det.track_id] = centroid
                if det.gt_actor_id is not None:
                    track_to_gt[det.track_id] = det.gt_actor_id

        self.total_gt += len(gt_map)

        if not gt_map or not tracked_map:
            self.false_negatives += len(gt_map)
            self.false_positives += len(tracked_map)
            return

        # Distance matrix
        gt_ids = list(gt_map.keys())
        tr_ids = list(tracked_map.keys())
        cost = np.full((len(gt_ids), len(tr_ids)), 1e6, dtype=np.float64)
        for gi, gid in enumerate(gt_ids):
            for ti, tid in enumerate(tr_ids):
                dist = float(np.linalg.norm(gt_map[gid][:2] - tracked_map[tid][:2]))
                if dist <= self.match_distance_m:
                    cost[gi, ti] = dist

        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt: set[int] = set()
        matched_tr: set[int] = set()
        frame_matches: dict[int, int] = {}

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.match_distance_m:
                continue
            gid = gt_ids[r]
            tid = tr_ids[c]
            self.true_positives += 1
            self.total_distance += cost[r, c]
            matched_gt.add(gid)
            matched_tr.add(tid)
            frame_matches[gid] = tid

            # Check for ID switch
            if gid in self._prev_matches and self._prev_matches[gid] != tid:
                self.id_switches += 1

        self.false_negatives += len(gt_map) - len(matched_gt)
        self.false_positives += len(tracked_map) - len(matched_tr)
        self._prev_matches = frame_matches

    @property
    def mota(self) -> float:
        """Multi-Object Tracking Accuracy: 1 - (FP + FN + IDSW) / total_gt."""
        if self.total_gt == 0:
            return 0.0
        return 1.0 - (self.false_positives + self.false_negatives + self.id_switches) / self.total_gt

    @property
    def motp(self) -> float:
        """Multi-Object Tracking Precision: mean distance of matched pairs."""
        if self.true_positives == 0:
            return float("inf")
        return self.total_distance / self.true_positives

    def summary(self) -> dict[str, float]:
        return {
            "mota": round(self.mota, 4),
            "motp_m": round(self.motp, 4),
            "true_positives": float(self.true_positives),
            "false_positives": float(self.false_positives),
            "false_negatives": float(self.false_negatives),
            "id_switches": float(self.id_switches),
            "total_gt": float(self.total_gt),
        }

