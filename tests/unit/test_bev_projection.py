from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import DrivableSpaceMask, EgoPose
from autonomy_demo.perception.bev_projection import BEVDrivableProjector, GRID_SIZE


def _ego_pose() -> EgoPose:
    return EgoPose(
        world_xyz=np.zeros(3, dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        current_lane_id="lane_001",
        frenet_s=0.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )


def _drivable(mask: np.ndarray, confidence: float = 0.9) -> DrivableSpaceMask:
    probabilities = np.zeros(mask.shape + (2,), dtype=np.float32)
    probabilities[..., 1] = np.where(mask, confidence, 0.0).astype(np.float32)
    probabilities[..., 0] = 1.0 - probabilities[..., 1]
    return DrivableSpaceMask(
        mask=mask.astype(np.bool_),
        class_probabilities=probabilities,
        source_sensor_id="front_camera",
    )


def _pitched_calibration(height: int, width: int) -> dict[str, object]:
    return {
        "fov_deg": 90.0,
        "image_width": width,
        "image_height": height,
        "mount_xyz": [2.3, 0.0, 1.2],
        "mount_rpy_deg": [0.0, 12.0, 0.0],
    }


def test_bev_projection_projects_centered_road_band_into_centered_ribbon() -> None:
    projector = BEVDrivableProjector()
    mask = np.zeros((120, 200), dtype=np.bool_)
    mask[60:, 70:130] = True

    grid = projector.project(
        _drivable(mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*mask.shape),
    )

    occupied = np.argwhere(grid > 0)
    assert occupied.size > 0
    occupied_rows = occupied[:, 0]
    occupied_cols = occupied[:, 1]
    assert np.ptp(occupied_rows) >= 6
    assert 4 <= np.ptp(occupied_cols) <= 20
    assert abs(float(occupied_cols.mean()) - (GRID_SIZE / 2.0)) < 6.0


def test_bev_projection_preserves_left_right_lateral_structure() -> None:
    projector = BEVDrivableProjector()
    left_mask = np.zeros((120, 200), dtype=np.bool_)
    right_mask = np.zeros((120, 200), dtype=np.bool_)
    left_mask[65:, 20:70] = True
    right_mask[65:, 130:180] = True

    left_grid = projector.project(
        _drivable(left_mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*left_mask.shape),
    )
    center_mask = np.zeros((120, 200), dtype=np.bool_)
    center_mask[65:, 75:125] = True
    center_grid = projector.project(
        _drivable(center_mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*center_mask.shape),
    )
    right_grid = projector.project(
        _drivable(right_mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*right_mask.shape),
    )

    left_cols = np.argwhere(left_grid > 0)[:, 1]
    center_cols = np.argwhere(center_grid > 0)[:, 1]
    right_cols = np.argwhere(right_grid > 0)[:, 1]
    assert left_cols.size > 0
    assert center_cols.size > 0
    assert right_cols.size > 0
    assert float(left_cols.mean()) < float(center_cols.mean())
    assert float(center_cols.mean()) < float(right_cols.mean())
    assert float(right_cols.mean()) - float(left_cols.mean()) >= 6.0


def test_bev_projection_discards_pixels_above_horizon_or_behind_camera() -> None:
    projector = BEVDrivableProjector()
    mask = np.zeros((120, 200), dtype=np.bool_)
    mask[:18, 75:125] = True

    grid = projector.project(
        _drivable(mask),
        _ego_pose(),
        camera_calibration={
            "fov_deg": 90.0,
            "image_width": 200,
            "image_height": 120,
            "mount_xyz": [2.3, 0.0, 1.2],
            "mount_rpy_deg": [0.0, 0.0, 0.0],
        },
    )

    assert not grid.any()


def test_bev_projection_turns_sparse_stripes_into_contiguous_ground_patch() -> None:
    projector = BEVDrivableProjector()
    mask = np.zeros((120, 200), dtype=np.bool_)
    for col in range(70, 130, 4):
        mask[62:, col : col + 2] = True

    grid = projector.project(
        _drivable(mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*mask.shape),
    )

    occupied = np.argwhere(grid > 0)
    assert occupied.size > 0
    row_min, col_min = occupied.min(axis=0)
    row_max, col_max = occupied.max(axis=0)
    window = grid[row_min : row_max + 1, col_min : col_max + 1] > 0
    density = float(window.mean())
    assert density >= 0.55
