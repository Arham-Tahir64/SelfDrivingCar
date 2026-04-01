from __future__ import annotations

from typing import Any

import numpy as np

from autonomy_demo.interfaces.types import DrivableSpaceMask, EgoPose

# BEV grid parameters
GRID_SIZE = 100
CELL_SIZE_M = 0.5
FORWARD_RANGE_M = GRID_SIZE * CELL_SIZE_M
LATERAL_RANGE_M = (GRID_SIZE // 2) * CELL_SIZE_M

_DEFAULT_FRONT_CAMERA_CALIBRATION = {
    "fov_deg": 90.0,
    "mount_xyz": [2.3, 0.0, 0.8],
    "mount_rpy_deg": [0.0, 0.0, 0.0],  # roll, pitch, yaw
}
_ROAD_PROB_THRESHOLD = 0.35
_GRID_CONFIDENCE_THRESHOLD = 40.0


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class BEVDrivableProjector:
    """Projects a front-camera road mask into an ego-relative BEV grid."""

    def __init__(self) -> None:
        self._ray_directions_ego: np.ndarray | None = None
        self._camera_origin_ego = np.zeros(3, dtype=np.float32)
        self._cache_key: tuple[Any, ...] | None = None

    def _normalise_calibration(
        self,
        calibration: dict[str, Any] | None,
        *,
        image_height: int,
        image_width: int,
    ) -> dict[str, Any]:
        merged = dict(_DEFAULT_FRONT_CAMERA_CALIBRATION)
        if calibration:
            merged.update(calibration)
        merged["image_width"] = int(image_width)
        merged["image_height"] = int(image_height)
        merged["fov_deg"] = float(merged.get("fov_deg", 90.0))
        merged["mount_xyz"] = list(merged.get("mount_xyz", [2.3, 0.0, 0.8]))
        merged["mount_rpy_deg"] = list(merged.get("mount_rpy_deg", [0.0, 0.0, 0.0]))
        return merged

    def _build_lookup(self, calibration: dict[str, Any]) -> None:
        image_height = int(calibration["image_height"])
        image_width = int(calibration["image_width"])
        hfov_rad = np.deg2rad(float(calibration["fov_deg"]))
        vfov_rad = 2.0 * np.arctan(np.tan(hfov_rad * 0.5) * (image_height / max(image_width, 1)))

        fx = image_width / max(2.0 * np.tan(hfov_rad * 0.5), 1e-6)
        fy = image_height / max(2.0 * np.tan(vfov_rad * 0.5), 1e-6)
        cx = image_width * 0.5
        cy = image_height * 0.5

        us = np.arange(image_width, dtype=np.float32)
        vs = np.arange(image_height, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        # Camera ray in ego-style coordinates: x forward, y right, z up.
        rays_camera = np.stack(
            [
                np.ones_like(uu, dtype=np.float32),
                (uu - cx) / max(fx, 1e-6),
                -(vv - cy) / max(fy, 1e-6),
            ],
            axis=-1,
        )

        roll_deg, pitch_deg, yaw_deg = calibration["mount_rpy_deg"]
        rotation = (
            _rotation_z(np.deg2rad(float(yaw_deg)))
            @ _rotation_y(np.deg2rad(float(pitch_deg)))
            @ _rotation_x(np.deg2rad(float(roll_deg)))
        ).astype(np.float32)

        self._ray_directions_ego = rays_camera.reshape(-1, 3) @ rotation.T
        self._camera_origin_ego = np.asarray(calibration["mount_xyz"], dtype=np.float32)
        self._cache_key = (
            image_height,
            image_width,
            float(calibration["fov_deg"]),
            *self._camera_origin_ego.tolist(),
            *[float(v) for v in calibration["mount_rpy_deg"]],
        )

    def _connected_component_from_ego_anchor(self, candidate: np.ndarray) -> np.ndarray:
        if not candidate.any():
            return candidate.astype(np.bool_)
        try:
            import cv2  # type: ignore

            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                candidate.astype(np.uint8),
                connectivity=8,
            )
            if num_labels <= 1:
                return candidate.astype(np.bool_)
            height, width = candidate.shape
            seed = np.zeros_like(candidate, dtype=np.bool_)
            seed[int(height * 0.72) :, int(width * 0.35) : int(width * 0.65)] = True
            bottom_seed = np.zeros_like(candidate, dtype=np.bool_)
            bottom_seed[int(height * 0.86) :, :] = True

            best_label = 1
            best_score = -1
            for label in range(1, num_labels):
                component = labels == label
                area = int(stats[label, 4])
                if np.any(component & seed):
                    score = area + 10_000
                elif np.any(component & bottom_seed):
                    score = area + 1_000
                else:
                    score = area
                if score > best_score:
                    best_label = label
                    best_score = score
            return (labels == best_label).astype(np.bool_)
        except Exception:
            return candidate.astype(np.bool_)

    def _projection_mask(self, drivable: DrivableSpaceMask) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(drivable.mask, dtype=np.bool_)
        if drivable.class_probabilities.ndim == 3 and drivable.class_probabilities.shape[2] >= 2:
            values = np.asarray(drivable.class_probabilities[..., 1], dtype=np.float32)
        else:
            values = mask.astype(np.float32)
        candidate = mask & (values >= _ROAD_PROB_THRESHOLD)
        if not candidate.any():
            candidate = mask
        candidate = self._connected_component_from_ego_anchor(candidate)
        return candidate, values

    def _binary_dilate(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.bool_), 1, mode="constant", constant_values=False)
        result = np.zeros_like(mask, dtype=np.bool_)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return result

    def _binary_erode(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.bool_), 1, mode="constant", constant_values=True)
        result = np.ones_like(mask, dtype=np.bool_)
        for dy in range(3):
            for dx in range(3):
                result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return result

    def _cleanup_grid(self, grid: np.ndarray) -> np.ndarray:
        binary = grid >= _GRID_CONFIDENCE_THRESHOLD
        if not binary.any():
            return np.zeros_like(grid, dtype=np.uint8)
        closed = self._binary_erode(self._binary_dilate(binary))
        cleaned = np.where(closed, np.maximum(grid, _GRID_CONFIDENCE_THRESHOLD), 0.0)
        return np.clip(cleaned, 0, 255).astype(np.uint8)

    def project(
        self,
        drivable: DrivableSpaceMask,
        ego_pose: EgoPose,
        *,
        camera_calibration: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del ego_pose
        image_height, image_width = drivable.mask.shape[:2]
        calibration = self._normalise_calibration(
            camera_calibration,
            image_height=image_height,
            image_width=image_width,
        )
        cache_key = (
            image_height,
            image_width,
            float(calibration["fov_deg"]),
            *[float(v) for v in calibration["mount_xyz"]],
            *[float(v) for v in calibration["mount_rpy_deg"]],
        )
        if cache_key != self._cache_key or self._ray_directions_ego is None:
            self._build_lookup(calibration)

        candidate_mask, values = self._projection_mask(drivable)
        keep = candidate_mask.reshape(-1)
        if not keep.any():
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        directions = self._ray_directions_ego[keep]
        confidences = values.reshape(-1)[keep] * 255.0

        downward = directions[:, 2] < -1e-4
        if not np.any(downward):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        directions = directions[downward]
        confidences = confidences[downward]

        distances = -self._camera_origin_ego[2] / directions[:, 2]
        valid_distance = distances > 0.0
        if not np.any(valid_distance):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        directions = directions[valid_distance]
        confidences = confidences[valid_distance]
        distances = distances[valid_distance]

        points_ego = self._camera_origin_ego + (directions * distances[:, None])
        forward_m = points_ego[:, 0]
        lateral_m = points_ego[:, 1]
        in_bounds = (
            (forward_m >= 0.0)
            & (forward_m < FORWARD_RANGE_M)
            & (lateral_m >= -LATERAL_RANGE_M)
            & (lateral_m < LATERAL_RANGE_M)
        )
        if not np.any(in_bounds):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        forward_m = forward_m[in_bounds]
        lateral_m = lateral_m[in_bounds]
        confidences = confidences[in_bounds]

        rows = np.floor((FORWARD_RANGE_M - forward_m) / CELL_SIZE_M).astype(np.int32)
        cols = np.floor((lateral_m + LATERAL_RANGE_M) / CELL_SIZE_M).astype(np.int32)
        valid_indices = (
            (rows >= 0)
            & (rows < GRID_SIZE)
            & (cols >= 0)
            & (cols < GRID_SIZE)
        )
        if not np.any(valid_indices):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        rows = rows[valid_indices]
        cols = cols[valid_indices]
        confidences = confidences[valid_indices]

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        np.maximum.at(grid, (rows, cols), confidences)
        return self._cleanup_grid(grid)
