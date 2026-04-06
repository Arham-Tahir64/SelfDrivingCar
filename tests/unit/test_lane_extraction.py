from __future__ import annotations

import numpy as np
import pytest

from autonomy_demo.perception.internal_types import CameraSegmentationResult
from autonomy_demo.perception.lane_extraction import LaneExtractor


def _lane_frame(*, left_base_x: int = 130, right_base_x: int = 510) -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.line(image, (left_base_x, 350), (left_base_x + 120, 170), (255, 255, 255), 8)
    cv2.line(image, (right_base_x, 350), (right_base_x - 120, 170), (255, 255, 255), 8)
    return image


def test_lane_extractor_detects_camera_lanes_with_world_projection() -> None:
    extractor = LaneExtractor()
    lanes = extractor.extract(
        _lane_frame(),
        sensor_id="front_camera",
        ego_world_xyz=np.array([100.0, 50.0, 2.0], dtype=np.float32),
        ego_yaw_rad=0.25,
    )
    assert len(lanes) == 2
    assert {lane.lane_id for lane in lanes} == {"lane_left", "lane_right"}
    for lane in lanes:
        assert lane.source_modality == "camera"
        assert lane.source_sensor_ids == ["front_camera"]
        assert lane.position_estimate_kind == "camera_projection"
        assert lane.polyline_world.shape[1] == 3
        assert float(np.max(lane.polyline_world[:, 0])) > 100.0


def test_lane_extractor_smooths_successive_lane_polylines() -> None:
    extractor = LaneExtractor(smoothing_alpha=0.5)
    first_lanes = extractor.extract(
        _lane_frame(left_base_x=130, right_base_x=510),
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        ego_yaw_rate_rad_s=0.05,
    )
    second_lanes = extractor.extract(
        _lane_frame(left_base_x=150, right_base_x=490),
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        ego_yaw_rate_rad_s=0.05,
    )
    first_left = next(lane for lane in first_lanes if lane.lane_id == "lane_left")
    second_left = next(lane for lane in second_lanes if lane.lane_id == "lane_left")
    raw_shift_px = 20.0
    observed_shift_px = abs(float(second_left.polyline_image[0, 0] - first_left.polyline_image[0, 0]))
    assert observed_shift_px < raw_shift_px


def test_lane_extractor_disables_temporal_smoothing_during_high_yaw_rate_turns() -> None:
    extractor = LaneExtractor(smoothing_alpha=0.5)
    first_frame = _lane_frame(left_base_x=130, right_base_x=510)
    second_frame = _lane_frame(left_base_x=170, right_base_x=470)
    extractor.extract(
        first_frame,
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        ego_yaw_rate_rad_s=0.05,
    )
    second_image = np.asarray(second_frame, dtype=np.uint8)
    raw_left_lane, raw_right_lane = extractor._extract_with_opencv(second_image)
    raw_left_lane, raw_right_lane = extractor._sanitize_lane_pair(raw_left_lane, raw_right_lane, second_image.shape[1])
    raw_left_lane, _ = extractor._recover_lane_pair(
        raw_left_lane,
        raw_right_lane,
        second_image.shape[1],
        allow_stale_pair_recovery=False,
    )
    assert raw_left_lane is not None

    extractor.extract(
        second_frame,
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        ego_yaw_rate_rad_s=0.25,
    )

    assert np.allclose(extractor._previous_polylines["lane_left"], raw_left_lane.astype(np.float32))
    assert extractor._turn_smoothing_suppressed is True


def test_lane_extractor_produces_parallel_world_boundaries() -> None:
    extractor = LaneExtractor()
    lanes = extractor.extract(
        _lane_frame(),
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
    )
    assert len(lanes) == 2
    left_lane = next(lane for lane in lanes if lane.lane_id == "lane_left")
    right_lane = next(lane for lane in lanes if lane.lane_id == "lane_right")
    separation = np.asarray(left_lane.polyline_world[:, 1] - right_lane.polyline_world[:, 1], dtype=np.float32)
    assert float(np.std(separation)) < 0.5
    assert float(np.mean(np.abs(separation))) > 2.0


def test_lane_extractor_recovers_missing_boundary_from_previous_pair() -> None:
    extractor = LaneExtractor()
    first_lanes = extractor.extract(
        _lane_frame(),
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
    )
    assert len(first_lanes) == 2

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.line(image, (130, 350), (250, 170), (255, 255, 255), 8)
    recovered_lanes = extractor.extract(
        image,
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
    )
    assert len(recovered_lanes) == 2


def test_lane_extractor_does_not_recover_stale_pair_when_turn_smoothing_is_suppressed() -> None:
    extractor = LaneExtractor()
    seeded_lanes = extractor.extract(
        _lane_frame(),
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        ego_yaw_rate_rad_s=0.05,
    )
    assert len(seeded_lanes) == 2

    blank_image = np.zeros((360, 640, 3), dtype=np.uint8)
    lanes_during_turn = extractor.extract(
        blank_image,
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        ego_yaw_rate_rad_s=0.25,
    )
    assert lanes_during_turn == []


def test_lane_extractor_accepts_segmentation_priors_without_regressing_detection() -> None:
    extractor = LaneExtractor()
    frame = _lane_frame()
    priors = CameraSegmentationResult(
        semantic_label_map=np.zeros(frame.shape[:2], dtype=np.uint8),
        task_label_map=np.zeros(frame.shape[:2], dtype=np.uint8),
        task_probabilities=np.zeros(frame.shape[:2] + (7,), dtype=np.float32),
        drivable_prob=np.ones(frame.shape[:2], dtype=np.float32) * 0.9,
        lane_boundary_prob=np.zeros(frame.shape[:2], dtype=np.float32),
        curb_boundary_prob=np.zeros(frame.shape[:2], dtype=np.float32),
        uncertainty=np.zeros(frame.shape[:2], dtype=np.float32),
        source_sensor_id="front_camera",
        model_name="test_model",
        model_version="test_model",
    )
    priors.lane_boundary_prob[:, 120:150] = 0.9
    priors.lane_boundary_prob[:, 490:520] = 0.9

    lanes = extractor.extract(
        frame,
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
        segmentation_priors=priors,
    )

    assert len(lanes) == 2
