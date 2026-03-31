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

# Camera heuristic constants (matching lane_extraction.py assumptions)
_HORIZON_RATIO = 0.42
_MAX_FORWARD_M = 46.0  # 4.0 + 42.0
_MIN_FORWARD_M = 4.0


class BEVDrivableProjector:
    """Projects a camera-space drivable mask into a top-down BEV occupancy grid.

    Uses the same simplified IPM heuristic as lane_extraction.py:
    pixel y → depth, pixel x → lateral offset.  The output is a 100×100
    uint8 grid (0–255 confidence) centred on the ego vehicle, 0.5 m/cell.

    Row 0 = 50 m ahead, row 99 = ego position.
    Col 0 = 25 m left, col 99 = 25 m right.
    """

    def __init__(self) -> None:
        self._lut_built = False
        self._pixel_to_bev_row: np.ndarray | None = None
        self._pixel_to_bev_col: np.ndarray | None = None
        self._last_shape: tuple[int, int] = (0, 0)

    def _build_lookup(self, img_h: int, img_w: int) -> None:
        """Pre-compute pixel → BEV cell mapping for the given image size."""
        horizon_y = img_h * _HORIZON_RATIO

        # For each pixel (v, u), compute forward distance and lateral offset
        vs = np.arange(img_h, dtype=np.float32)
        us = np.arange(img_w, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)  # (H, W)

        # Clamp pixels above horizon — they map to max forward distance
        clamped_v = np.clip(vv, horizon_y, img_h - 1)
        depth_ratio = (clamped_v - horizon_y) / max(img_h - horizon_y, 1.0)

        # Forward distance: near bottom of image → close, near horizon → far
        forward_m = _MIN_FORWARD_M + ((1.0 - depth_ratio) ** 2) * (_MAX_FORWARD_M - _MIN_FORWARD_M)
        # Lateral offset: image centre → 0, edges → ±lateral
        normalised_x = (uu / max(img_w, 1.0) - 0.5) * 2.0
        lateral_m = normalised_x * 2.6  # same spread factor as lane_extraction

        # Convert to BEV grid coords
        # Row: 0 = max forward (50 m), 99 = ego (0 m)
        bev_row = (FORWARD_RANGE_M - forward_m) / CELL_SIZE_M
        # Col: 0 = 25 m left, 99 = 25 m right
        bev_col = (lateral_m + LATERAL_RANGE_M) / CELL_SIZE_M

        self._pixel_to_bev_row = np.clip(bev_row, 0, GRID_SIZE - 1).astype(np.int32)
        self._pixel_to_bev_col = np.clip(bev_col, 0, GRID_SIZE - 1).astype(np.int32)
        self._last_shape = (img_h, img_w)
        self._lut_built = True

    def project(self, drivable: DrivableSpaceMask, ego_pose: EgoPose) -> np.ndarray:
        """Project drivable mask to ego-relative BEV grid.

        Returns a uint8 array of shape (GRID_SIZE, GRID_SIZE) where 255 = fully
        drivable and 0 = not drivable.  The grid is ego-relative (no world
        rotation applied) so the dashboard can place it oriented by ego yaw.
        """
        mask = drivable.mask  # (H, W) bool
        probs = drivable.class_probabilities  # (H, W, 2) — [:,:,1] is drivable prob
        img_h, img_w = mask.shape[:2]

        if (img_h, img_w) != self._last_shape or not self._lut_built:
            self._build_lookup(img_h, img_w)

        # Use probability channel if available, else binary mask
        if probs is not None and probs.ndim == 3 and probs.shape[2] >= 2:
            values = (probs[:, :, 1] * 255.0).astype(np.float32)
        else:
            values = mask.astype(np.float32) * 255.0

        # Scatter image pixels into BEV grid (max-pooling per cell)
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        rows = self._pixel_to_bev_row.ravel()  # type: ignore[union-attr]
        cols = self._pixel_to_bev_col.ravel()  # type: ignore[union-attr]
        vals = values.ravel()

        # Use np.add.at for accumulation (mean-pooling for smoother result)
        np.add.at(grid, (rows, cols), vals)
        np.add.at(counts, (rows, cols), 1.0)

        valid = counts > 0
        grid[valid] /= counts[valid]

        return np.clip(grid, 0, 255).astype(np.uint8)
