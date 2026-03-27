from __future__ import annotations

from typing import Any

import numpy as np

from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import LaneLine


def _image_to_world_polyline(polyline_image: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    image_height, image_width = image_shape[:2]
    world_points: list[list[float]] = []
    for pixel_x, pixel_y in polyline_image:
        normalized_x = ((float(pixel_x) / max(image_width, 1)) - 0.5) * 2.0
        normalized_y = 1.0 - (float(pixel_y) / max(image_height, 1))
        forward_m = max(3.0, normalized_y * 40.0)
        lateral_m = normalized_x * 4.0
        world_points.append([forward_m, lateral_m, 0.0])
    return np.asarray(world_points, dtype=np.float32)


class LaneExtractor:
    """TODO(PRD 3.2.3): replace these heuristics with a learned lane detector or BEV lane head."""

    def extract(self, frame: np.ndarray) -> list[LaneLine]:
        image = np.asarray(frame, dtype=np.uint8)
        image_height, image_width = image.shape[:2]
        left_lane, right_lane = self._extract_with_opencv(image)
        if left_lane is None or right_lane is None:
            left_lane = np.array(
                [
                    [image_width * 0.35, image_height * 0.95],
                    [image_width * 0.40, image_height * 0.70],
                    [image_width * 0.45, image_height * 0.45],
                ],
                dtype=np.float32,
            )
            right_lane = np.array(
                [
                    [image_width * 0.65, image_height * 0.95],
                    [image_width * 0.60, image_height * 0.70],
                    [image_width * 0.55, image_height * 0.45],
                ],
                dtype=np.float32,
            )
        lanes = []
        for lane_id, polyline in (("lane_left", left_lane), ("lane_right", right_lane)):
            lanes.append(
                LaneLine(
                    lane_id=lane_id,
                    polyline_image=polyline.astype(np.float32),
                    polyline_world=_image_to_world_polyline(polyline.astype(np.float32), image.shape),
                    line_type=LaneLineType.SOLID,
                    confidence=0.75,
                )
            )
        return lanes

    def _extract_with_opencv(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - optional dependency path
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array(
            [[(0, height), (width * 0.45, height * 0.55), (width * 0.55, height * 0.55), (width, height)]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        cropped = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(
            cropped,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=60,
            maxLineGap=100,
        )
        if lines is None:
            return None, None
        left_points: list[tuple[float, float]] = []
        right_points: list[tuple[float, float]] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.4:
                continue
            points = left_points if slope < 0 else right_points
            points.extend([(float(x1), float(y1)), (float(x2), float(y2))])
        return self._fit_lane(left_points, height), self._fit_lane(right_points, height)

    def _fit_lane(self, points: list[tuple[float, float]], image_height: int) -> np.ndarray | None:
        if len(points) < 4:
            return None
        xs = np.asarray([point[0] for point in points], dtype=np.float32)
        ys = np.asarray([point[1] for point in points], dtype=np.float32)
        fit = np.polyfit(ys, xs, deg=1)
        sample_ys = np.asarray(
            [image_height * 0.95, image_height * 0.72, image_height * 0.48],
            dtype=np.float32,
        )
        sample_xs = fit[0] * sample_ys + fit[1]
        return np.stack([sample_xs, sample_ys], axis=1).astype(np.float32)
