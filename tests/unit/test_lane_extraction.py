from __future__ import annotations

import numpy as np
import pytest

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
    )
    second_lanes = extractor.extract(
        _lane_frame(left_base_x=150, right_base_x=490),
        sensor_id="front_camera",
        ego_world_xyz=np.zeros(3, dtype=np.float32),
        ego_yaw_rad=0.0,
    )
    first_left = next(lane for lane in first_lanes if lane.lane_id == "lane_left")
    second_left = next(lane for lane in second_lanes if lane.lane_id == "lane_left")
    raw_shift_px = 20.0
    observed_shift_px = abs(float(second_left.polyline_image[0, 0] - first_left.polyline_image[0, 0]))
    assert observed_shift_px < raw_shift_px


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
