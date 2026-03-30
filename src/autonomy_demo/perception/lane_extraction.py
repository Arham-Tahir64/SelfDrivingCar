from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import LaneLine


def _sensor_mount(sensor_id: str) -> tuple[np.ndarray, float]:
    mounts = {
        "front_camera": (np.array([2.3, 0.0, 0.0], dtype=np.float32), 0.0),
        "rear_camera": (np.array([-2.6, 0.0, 0.0], dtype=np.float32), np.pi),
        "left_camera": (np.array([0.0, -0.8, 0.0], dtype=np.float32), -np.pi * 0.5),
        "right_camera": (np.array([0.0, 0.8, 0.0], dtype=np.float32), np.pi * 0.5),
    }
    return mounts.get(sensor_id, (np.zeros(3, dtype=np.float32), 0.0))


def _rotate_xy(vector_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    rotation = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rotation @ np.asarray(vector_xyz, dtype=np.float32)).astype(np.float32)


def _image_to_world_polyline(
    polyline_image: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    sensor_id: str,
    ego_world_xyz: np.ndarray,
    ego_yaw_rad: float,
) -> np.ndarray:
    image_height, image_width = image_shape[:2]
    horizon_y = image_height * 0.42
    sensor_offset, sensor_yaw_rad = _sensor_mount(sensor_id)
    world_points: list[np.ndarray] = []

    for pixel_x, pixel_y in polyline_image:
        clamped_y = float(np.clip(pixel_y, horizon_y, image_height - 1))
        depth_ratio = float((clamped_y - horizon_y) / max(image_height - horizon_y, 1.0))
        normalized_x = ((float(pixel_x) / max(image_width, 1.0)) - 0.5) * 2.0
        forward_m = 4.0 + ((1.0 - depth_ratio) ** 2.0) * 42.0
        lateral_m = float(normalized_x * 2.6)
        ego_relative_point = _rotate_xy(
            np.array([forward_m, lateral_m, 0.0], dtype=np.float32),
            sensor_yaw_rad,
        ) + sensor_offset
        world_point = _rotate_xy(ego_relative_point, ego_yaw_rad) + np.asarray(ego_world_xyz, dtype=np.float32)
        world_point[2] = float(ego_world_xyz[2])
        world_points.append(world_point.astype(np.float32))

    return np.asarray(world_points, dtype=np.float32)


class LaneExtractor:
    """Camera-first lane detector with simple image-space fitting and temporal smoothing."""

    def __init__(self, *, smoothing_alpha: float = 0.7) -> None:
        self.smoothing_alpha = smoothing_alpha
        self._previous_polylines: dict[str, np.ndarray] = {}

    def extract(
        self,
        frame: np.ndarray,
        *,
        sensor_id: str = "front_camera",
        ego_world_xyz: np.ndarray | None = None,
        ego_yaw_rad: float = 0.0,
    ) -> list[LaneLine]:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3:
            return []
        image_height, image_width = image.shape[:2]
        left_lane, right_lane = self._extract_with_opencv(image)
        left_lane, right_lane = self._sanitize_lane_pair(left_lane, right_lane, image_width)
        detected = [("lane_left", left_lane), ("lane_right", right_lane)]
        lanes: list[LaneLine] = []
        ego_xyz = np.asarray(
            np.zeros(3, dtype=np.float32) if ego_world_xyz is None else ego_world_xyz,
            dtype=np.float32,
        )

        for lane_id, polyline in detected:
            if polyline is None:
                self._previous_polylines.pop(lane_id, None)
                continue
            smoothed_polyline = self._smooth_polyline(lane_id, polyline.astype(np.float32))
            lanes.append(
                LaneLine(
                    lane_id=lane_id,
                    polyline_image=smoothed_polyline,
                    polyline_world=_image_to_world_polyline(
                        smoothed_polyline,
                        image.shape,
                        sensor_id=sensor_id,
                        ego_world_xyz=ego_xyz,
                        ego_yaw_rad=ego_yaw_rad,
                    ),
                    line_type=LaneLineType.SOLID,
                    confidence=self._lane_confidence(smoothed_polyline, image_width, image_height),
                    source_modality="camera",
                    source_sensor_ids=[sensor_id],
                    position_estimate_kind="camera_projection",
                )
            )
        if len(lanes) >= 2:
            lanes = self._stabilize_lane_pair(lanes, ego_xyz, ego_yaw_rad, sensor_id=sensor_id, image_shape=image.shape)
        return lanes

    def _stabilize_lane_pair(
        self,
        lanes: list[LaneLine],
        ego_world_xyz: np.ndarray,
        ego_yaw_rad: float,
        *,
        sensor_id: str,
        image_shape: tuple[int, int, int],
    ) -> list[LaneLine]:
        left_lane = next((lane for lane in lanes if lane.lane_id == "lane_left"), None)
        right_lane = next((lane for lane in lanes if lane.lane_id == "lane_right"), None)
        if left_lane is None or right_lane is None:
            return lanes

        center_image = ((left_lane.polyline_image + right_lane.polyline_image) * 0.5).astype(np.float32)
        center_world = _image_to_world_polyline(
            center_image,
            image_shape,
            sensor_id=sensor_id,
            ego_world_xyz=ego_world_xyz,
            ego_yaw_rad=ego_yaw_rad,
        )
        lane_half_width_m = float(np.clip(np.mean(np.abs(right_lane.polyline_world[:, 1] - left_lane.polyline_world[:, 1])) * 0.5, 1.35, 2.1))
        left_world, right_world = self._offset_world_boundaries(center_world, lane_half_width_m)
        left_image = self._smooth_polyline("lane_left_stabilized", center_image - np.array([18.0, 0.0], dtype=np.float32))
        right_image = self._smooth_polyline("lane_right_stabilized", center_image + np.array([18.0, 0.0], dtype=np.float32))

        return [
            LaneLine(
                lane_id="lane_left",
                polyline_image=left_image,
                polyline_world=left_world,
                line_type=LaneLineType.SOLID,
                confidence=float(np.clip((left_lane.confidence + right_lane.confidence) * 0.5, 0.0, 0.99)),
                source_modality="camera",
                source_sensor_ids=[sensor_id],
                position_estimate_kind="camera_projection",
            ),
            LaneLine(
                lane_id="lane_right",
                polyline_image=right_image,
                polyline_world=right_world,
                line_type=LaneLineType.SOLID,
                confidence=float(np.clip((left_lane.confidence + right_lane.confidence) * 0.5, 0.0, 0.99)),
                source_modality="camera",
                source_sensor_ids=[sensor_id],
                position_estimate_kind="camera_projection",
            ),
        ]

    def _offset_world_boundaries(
        self,
        center_world: np.ndarray,
        half_width_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        center_world = np.asarray(center_world, dtype=np.float32)
        if len(center_world) < 2:
            return center_world, center_world
        tangents = np.zeros((len(center_world), 3), dtype=np.float32)
        for index in range(len(center_world)):
            if index == 0:
                direction = center_world[1] - center_world[0]
            elif index == len(center_world) - 1:
                direction = center_world[-1] - center_world[-2]
            else:
                direction = center_world[index + 1] - center_world[index - 1]
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction[:2]))
            if norm <= 1e-6:
                tangents[index] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                continue
            tangents[index] = direction / norm
        normals = np.stack([-tangents[:, 1], tangents[:, 0], np.zeros(len(center_world), dtype=np.float32)], axis=1)
        left_world = center_world + (normals * half_width_m)
        right_world = center_world - (normals * half_width_m)
        return left_world.astype(np.float32), right_world.astype(np.float32)

    def _smooth_polyline(self, lane_id: str, polyline: np.ndarray) -> np.ndarray:
        previous = self._previous_polylines.get(lane_id)
        if previous is None or previous.shape != polyline.shape:
            self._previous_polylines[lane_id] = polyline.astype(np.float32)
            return polyline.astype(np.float32)
        smoothed = ((self.smoothing_alpha * previous) + ((1.0 - self.smoothing_alpha) * polyline)).astype(
            np.float32
        )
        self._previous_polylines[lane_id] = smoothed
        return smoothed

    def _lane_confidence(self, polyline: np.ndarray, image_width: int, image_height: int) -> float:
        span_y = float(np.max(polyline[:, 1]) - np.min(polyline[:, 1]))
        vertical_coverage = span_y / max(float(image_height), 1.0)
        lateral_centering = 1.0 - min(abs(float(np.mean(polyline[:, 0]) / max(image_width, 1.0) - 0.5)), 0.5) * 2.0
        return float(np.clip(0.45 + (vertical_coverage * 0.35) + (lateral_centering * 0.2), 0.0, 0.99))

    def _extract_with_opencv(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - optional dependency path
            return None, None

        binary = self._lane_binary_mask(image, cv2)
        return self._extract_with_sliding_windows(binary)

    def _lane_binary_mask(self, image: np.ndarray, cv2) -> np.ndarray:  # noqa: ANN001
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        white_mask = cv2.inRange(
            hls,
            np.array([0, 170, 0], dtype=np.uint8),
            np.array([255, 255, 120], dtype=np.uint8),
        )
        yellow_mask = cv2.inRange(
            hls,
            np.array([12, 80, 80], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        )
        bright_mask = cv2.inRange(gray, 180, 255)
        combined = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), bright_mask)
        blurred = cv2.GaussianBlur(combined, (5, 5), 0)
        height, width = blurred.shape
        mask = np.zeros_like(blurred)
        polygon = np.array(
            [[
                (int(width * 0.08), height),
                (int(width * 0.40), int(height * 0.50)),
                (int(width * 0.60), int(height * 0.50)),
                (int(width * 0.92), height),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        cropped = cv2.bitwise_and(blurred, mask)
        _, binary = cv2.threshold(cropped, 160, 255, cv2.THRESH_BINARY)
        return binary

    def _extract_with_sliding_windows(self, binary: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        height, width = binary.shape
        histogram = np.sum(binary[height // 2 :, :], axis=0)
        midpoint = width // 2
        left_base = int(np.argmax(histogram[:midpoint])) if np.any(histogram[:midpoint]) else -1
        right_base = (
            int(np.argmax(histogram[midpoint:]) + midpoint)
            if np.any(histogram[midpoint:])
            else -1
        )
        left_lane = self._fit_lane_from_windows(binary, left_base) if left_base >= 0 else None
        right_lane = self._fit_lane_from_windows(binary, right_base) if right_base >= 0 else None
        return left_lane, right_lane

    def _fit_lane_from_windows(self, binary: np.ndarray, start_x: int) -> np.ndarray | None:
        if start_x < 0:
            return None
        nonzero_y, nonzero_x = binary.nonzero()
        if len(nonzero_x) == 0:
            return None
        height, width = binary.shape
        num_windows = 8
        margin = max(30, width // 18)
        min_pixels = 12
        window_height = max(1, height // num_windows)
        current_x = int(start_x)
        lane_indices: list[np.ndarray] = []

        for window_index in range(num_windows):
            y_low = height - ((window_index + 1) * window_height)
            y_high = height - (window_index * window_height)
            x_low = max(current_x - margin, 0)
            x_high = min(current_x + margin, width)
            good = (
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            )
            good_indices = np.where(good)[0]
            if good_indices.size == 0:
                continue
            lane_indices.append(good_indices)
            if good_indices.size >= min_pixels:
                current_x = int(np.mean(nonzero_x[good_indices]))

        if not lane_indices:
            return None
        lane_indices_flat = np.concatenate(lane_indices)
        xs = np.asarray(nonzero_x[lane_indices_flat], dtype=np.float32)
        ys = np.asarray(nonzero_y[lane_indices_flat], dtype=np.float32)
        if xs.size < 30:
            return None
        fit_degree = 2 if xs.size >= 60 else 1
        fit = np.polyfit(ys, xs, deg=fit_degree)
        sample_ys = np.linspace(height * 0.96, height * 0.50, num=6, dtype=np.float32)
        sample_xs = np.polyval(fit, sample_ys)
        return np.stack([sample_xs, sample_ys], axis=1).astype(np.float32)

    def _sanitize_lane_pair(
        self,
        left_lane: np.ndarray | None,
        right_lane: np.ndarray | None,
        image_width: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if left_lane is None or right_lane is None:
            return left_lane, right_lane
        if float(left_lane[0, 0]) > float(right_lane[0, 0]):
            left_lane, right_lane = right_lane, left_lane
        lane_widths = right_lane[:, 0] - left_lane[:, 0]
        median_width = float(np.median(lane_widths))
        width_variation = float(np.max(lane_widths) - np.min(lane_widths))
        if np.any(lane_widths <= 8.0):
            return None, None
        if median_width < image_width * 0.18 or median_width > image_width * 0.75:
            return None, None
        if width_variation > image_width * 0.35:
            return None, None
        return left_lane, right_lane
