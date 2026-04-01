from __future__ import annotations

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import DrivableSpaceMask, EgoPose

logger = get_logger(__name__)

# BEV grid parameters
GRID_SIZE = 100  # 100x100 cells
CELL_SIZE_M = 0.5  # 0.5 m per cell
FORWARD_RANGE_M = GRID_SIZE * CELL_SIZE_M  # 50 m forward
LATERAL_RANGE_M = (GRID_SIZE // 2) * CELL_SIZE_M  # 25 m each side

# Camera heuristic constants
_HORIZON_RATIO = 0.42
_MAX_FORWARD_M = 46.0  # 4.0 + 42.0
_MIN_FORWARD_M = 4.0
# Effective lateral half-angle factor.  CARLA's 90° HFOV → tan(45°)=1.0,
# but the road only occupies the central portion of the image.  A factor
# of 0.55 keeps the projection within realistic road widths (~±8 m at 15 m
# depth) while still fanning out with perspective.
_LATERAL_FACTOR = 0.55


class BEVDrivableProjector:
    """Projects a camera-space drivable mask into a top-down BEV occupancy grid.

    Uses a simplified IPM heuristic: pixel y → depth, pixel x → lateral
    offset (scaled by depth for perspective).  The output is a 100×100
    uint8 grid (0–255 confidence) centred on the ego vehicle, 0.5 m/cell.

    Row 0 = 50 m ahead, row 99 = ego position.
    Col 0 = 25 m left, col 99 = 25 m right.
    """

    def __init__(self) -> None:
        self._pixel_to_bev_row: np.ndarray | None = None
        self._pixel_to_bev_col: np.ndarray | None = None
        self._in_bounds: np.ndarray | None = None
        self._last_shape: tuple[int, int] = (0, 0)

    def _build_lookup(self, img_h: int, img_w: int) -> None:
        """Pre-compute pixel → BEV cell mapping for the given image size."""
        horizon_y = img_h * _HORIZON_RATIO

        vs = np.arange(img_h, dtype=np.float32)
        us = np.arange(img_w, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)  # (H, W)

        # Clamp pixels above horizon
        clamped_v = np.clip(vv, horizon_y, img_h - 1)
        depth_ratio = (clamped_v - horizon_y) / max(img_h - horizon_y, 1.0)

        # Forward distance: bottom of image → close, horizon → far
        forward_m = _MIN_FORWARD_M + ((1.0 - depth_ratio) ** 2) * (_MAX_FORWARD_M - _MIN_FORWARD_M)

        # Lateral offset: perspective-correct (scales with depth)
        normalised_x = (uu / max(img_w, 1.0) - 0.5) * 2.0
        lateral_m = normalised_x * forward_m * _LATERAL_FACTOR

        # BEV grid coordinates (float)
        bev_row_f = (FORWARD_RANGE_M - forward_m) / CELL_SIZE_M
        bev_col_f = (lateral_m + LATERAL_RANGE_M) / CELL_SIZE_M

        # Validity mask: only pixels that map inside the grid AND below horizon
        self._in_bounds = (
            (bev_row_f >= 0)
            & (bev_row_f < GRID_SIZE)
            & (bev_col_f >= 0)
            & (bev_col_f < GRID_SIZE)
            & (vv > horizon_y)
        )

        self._pixel_to_bev_row = np.clip(bev_row_f, 0, GRID_SIZE - 1).astype(np.int32)
        self._pixel_to_bev_col = np.clip(bev_col_f, 0, GRID_SIZE - 1).astype(np.int32)
        self._last_shape = (img_h, img_w)

    def project(self, drivable: DrivableSpaceMask, ego_pose: EgoPose) -> np.ndarray:
        """Project drivable mask to ego-relative BEV grid.

        Returns a uint8 array of shape (GRID_SIZE, GRID_SIZE) where 255 = fully
        drivable and 0 = not drivable.  The grid is ego-relative (no world
        rotation applied) so the dashboard can place it oriented by ego yaw.
        """
        mask = drivable.mask  # (H, W) bool
        probs = drivable.class_probabilities  # (H, W, C)
        img_h, img_w = mask.shape[:2]

        if (img_h, img_w) != self._last_shape:
            self._build_lookup(img_h, img_w)

        # Confidence values: prefer probability channel, fall back to binary mask
        if probs is not None and probs.ndim == 3 and probs.shape[2] >= 2:
            values = probs[:, :, 1].astype(np.float32)
        else:
            values = mask.astype(np.float32)

        # Only scatter drivable pixels that land inside the grid
        keep = self._in_bounds & mask  # type: ignore[operator]
        rows = self._pixel_to_bev_row[keep]  # type: ignore[index]
        cols = self._pixel_to_bev_col[keep]  # type: ignore[index]
        vals = values[keep] * 255.0

        # Mean-pool into the BEV grid
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        np.add.at(grid, (rows, cols), vals)
        np.add.at(counts, (rows, cols), 1.0)

        valid = counts > 0
        grid[valid] /= counts[valid]

        return np.clip(grid, 0, 255).astype(np.uint8)
