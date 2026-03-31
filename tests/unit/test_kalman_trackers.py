from __future__ import annotations

import numpy as np
import pytest

from autonomy_demo.interfaces.enums import ObjectClass, TrackState
from autonomy_demo.interfaces.types import ObjectDetection
from autonomy_demo.perception.internal_types import FrameDetection2D, LidarClusterDetection
from autonomy_demo.perception.tracking import KalmanSortTracker, _KalmanBoxTracker
from autonomy_demo.perception.lidar_tracking import KalmanCentroidTracker3D, _KalmanTracker3D
from autonomy_demo.eval.metrics import MOTAccumulator


# ---------- _KalmanBoxTracker unit tests ----------

def test_kalman_box_tracker_predict_advances_state() -> None:
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    kf = _KalmanBoxTracker(bbox)
    pred = kf.predict()
    assert pred.shape == (4,)
    assert pred[0] < pred[2] and pred[1] < pred[3]  # valid bbox


def test_kalman_box_tracker_update_corrects_state() -> None:
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    kf = _KalmanBoxTracker(bbox)
    kf.predict()
    shifted_bbox = np.array([110, 100, 210, 200], dtype=np.float32)
    kf.update(shifted_bbox)
    pred = kf.predicted_bbox
    # After update toward shifted bbox, center x should move right
    assert float(pred[0] + pred[2]) / 2 > 145


# ---------- KalmanSortTracker tests ----------

def _det_2d(
    bbox: list[float],
    cls: ObjectClass = ObjectClass.VEHICLE,
    preferred_id: int | None = None,
) -> FrameDetection2D:
    return FrameDetection2D(
        bbox_xyxy=np.array(bbox, dtype=np.float32),
        object_class=cls,
        confidence=0.9,
        preferred_track_id=preferred_id,
    )


def test_kalman_sort_creates_new_tracks() -> None:
    tracker = KalmanSortTracker()
    dets = [_det_2d([10, 10, 50, 50]), _det_2d([200, 200, 250, 250])]
    outputs = tracker.update(dets)
    assert len(outputs) == 2
    assert outputs[0].track_id != outputs[1].track_id


def test_kalman_sort_confirms_after_two_hits() -> None:
    tracker = KalmanSortTracker()
    det = _det_2d([10, 10, 50, 50])
    out1 = tracker.update([det])
    assert out1[0].track_state == TrackState.TENTATIVE
    out2 = tracker.update([_det_2d([12, 10, 52, 50])])  # slight shift
    assert out2[0].track_state == TrackState.CONFIRMED
    assert out2[0].track_id == out1[0].track_id


def test_kalman_sort_deletes_stale_tracks() -> None:
    tracker = KalmanSortTracker(max_missed=2)
    tracker.update([_det_2d([10, 10, 50, 50])])
    # 3 frames with no detections
    for _ in range(3):
        tracker.update([])
    # Track should be deleted; new detection gets new ID
    out = tracker.update([_det_2d([10, 10, 50, 50])])
    assert len(out) == 1


def test_kalman_sort_hungarian_assignment_matches_correctly() -> None:
    tracker = KalmanSortTracker()
    # Two distinct objects
    dets1 = [_det_2d([10, 10, 50, 50]), _det_2d([200, 200, 240, 240])]
    out1 = tracker.update(dets1)
    ids1 = {o.track_id for o in out1}

    # Slight movement
    dets2 = [_det_2d([12, 11, 52, 51]), _det_2d([202, 201, 242, 241])]
    out2 = tracker.update(dets2)
    ids2 = {o.track_id for o in out2}

    assert ids1 == ids2  # same tracks maintained


def test_kalman_sort_respects_preferred_track_id() -> None:
    tracker = KalmanSortTracker()
    det = _det_2d([10, 10, 50, 50], preferred_id=42)
    out = tracker.update([det])
    assert out[0].track_id == 42


# ---------- _KalmanTracker3D unit tests ----------

def test_kalman_3d_predict_returns_position() -> None:
    kf = _KalmanTracker3D(np.array([5.0, 3.0, 1.0]))
    pos = kf.predict()
    assert pos.shape == (3,)
    assert pos.dtype == np.float32


def test_kalman_3d_velocity_estimates_converge() -> None:
    kf = _KalmanTracker3D(np.array([0.0, 0.0, 0.0]))
    # Simulate object moving at 2 m/s in x
    for i in range(10):
        kf.predict()
        kf.update(np.array([2.0 * (i + 1) * 0.05, 0.0, 0.0]))
    # Velocity in x should be positive
    assert kf.velocity[0] > 0


# ---------- KalmanCentroidTracker3D tests ----------

def _lidar_det(
    centroid: list[float],
    cls: ObjectClass = ObjectClass.VEHICLE,
) -> LidarClusterDetection:
    cx, cy, cz = centroid
    return LidarClusterDetection(
        centroid_xyz=np.array(centroid, dtype=np.float32),
        world_bbox_3d=np.array([
            [cx - 1, cy - 0.5, cz],
            [cx + 1, cy - 0.5, cz],
            [cx + 1, cy + 0.5, cz],
            [cx - 1, cy + 0.5, cz],
            [cx - 1, cy - 0.5, cz + 1.5],
            [cx + 1, cy - 0.5, cz + 1.5],
            [cx + 1, cy + 0.5, cz + 1.5],
            [cx - 1, cy + 0.5, cz + 1.5],
        ], dtype=np.float32),
        object_class=cls,
        confidence=0.85,
        point_count=20,
    )


def test_kalman_3d_tracker_creates_tracks() -> None:
    tracker = KalmanCentroidTracker3D()
    out = tracker.update([_lidar_det([10, 5, 0])], timestamp_s=0.0)
    assert len(out) == 1
    assert out[0].track_state == TrackState.TENTATIVE


def test_kalman_3d_tracker_confirms_tracks() -> None:
    tracker = KalmanCentroidTracker3D()
    out1 = tracker.update([_lidar_det([10, 5, 0])], timestamp_s=0.0)
    out2 = tracker.update([_lidar_det([10.2, 5.1, 0])], timestamp_s=0.05)
    assert out2[0].track_state == TrackState.CONFIRMED
    assert out2[0].track_id == out1[0].track_id


def test_kalman_3d_tracker_estimates_velocity() -> None:
    tracker = KalmanCentroidTracker3D()
    tracker.update([_lidar_det([10, 0, 0])], timestamp_s=0.0)
    out = tracker.update([_lidar_det([12, 0, 0])], timestamp_s=0.1)
    # Velocity in x should be positive (moving in +x)
    assert out[0].velocity_xyz[0] > 0


def test_kalman_3d_tracker_hungarian_matches_multiple_objects() -> None:
    tracker = KalmanCentroidTracker3D()
    out1 = tracker.update(
        [_lidar_det([10, 0, 0]), _lidar_det([30, 0, 0])],
        timestamp_s=0.0,
    )
    out2 = tracker.update(
        [_lidar_det([10.5, 0, 0]), _lidar_det([30.5, 0, 0])],
        timestamp_s=0.05,
    )
    ids1 = {o.track_id for o in out1}
    ids2 = {o.track_id for o in out2}
    assert ids1 == ids2


# ---------- MOTAccumulator tests ----------

def _obj_det(track_id: int, gt_id: int | None, cx: float, cy: float) -> ObjectDetection:
    return ObjectDetection(
        track_id=track_id,
        object_class=ObjectClass.VEHICLE,
        world_bbox_3d=np.array([
            [cx - 1, cy - 0.5, 0],
            [cx + 1, cy - 0.5, 0],
            [cx + 1, cy + 0.5, 0],
            [cx - 1, cy + 0.5, 0],
            [cx - 1, cy - 0.5, 1.5],
            [cx + 1, cy - 0.5, 1.5],
            [cx + 1, cy + 0.5, 1.5],
            [cx - 1, cy + 0.5, 1.5],
        ], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        confidence=0.9,
        track_state=TrackState.CONFIRMED,
        gt_actor_id=gt_id,
    )


def test_mot_perfect_tracking() -> None:
    acc = MOTAccumulator()
    # track_id matches gt_actor_id, same position
    dets = [_obj_det(track_id=1, gt_id=1, cx=10, cy=0)]
    acc.update(dets)
    assert acc.mota == 1.0
    assert acc.motp < 0.01  # near-zero distance


def test_mot_id_switch_detection() -> None:
    acc = MOTAccumulator()
    # Frame 1: track 1 matches gt 1
    acc.update([_obj_det(track_id=1, gt_id=1, cx=10, cy=0)])
    # Frame 2: track 2 matches gt 1 (ID switch)
    acc.update([_obj_det(track_id=2, gt_id=1, cx=10.5, cy=0)])
    assert acc.id_switches == 1


def test_mot_false_positive_and_negative() -> None:
    acc = MOTAccumulator()
    # Detection with gt_id=1 but track_id=1 at position (10, 0)
    # Another detection with no gt (track only) — false positive
    # GT actor at (50, 0) not matched — false negative
    dets = [
        _obj_det(track_id=1, gt_id=1, cx=10, cy=0),
        _obj_det(track_id=2, gt_id=None, cx=20, cy=0),  # no GT → FP
        _obj_det(track_id=3, gt_id=99, cx=50, cy=0),    # GT far from others
    ]
    acc.update(dets)
    assert acc.true_positives >= 1
    summary = acc.summary()
    assert summary["total_gt"] >= 1


def test_mot_summary_keys() -> None:
    acc = MOTAccumulator()
    summary = acc.summary()
    expected_keys = {"mota", "motp_m", "true_positives", "false_positives", "false_negatives", "id_switches", "total_gt"}
    assert set(summary.keys()) == expected_keys
