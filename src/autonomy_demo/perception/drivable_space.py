from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import DrivableSpaceMask


class DrivableSpaceExtractor:
    """TODO(PRD 3.2.3): replace this heuristic with a learned segmentation model or BEV occupancy head."""

    def extract(self, frame: np.ndarray, sensor_id: str) -> DrivableSpaceMask:
        mask = self._extract_with_opencv(frame)
        if mask is None:
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.bool_)
            mask[height // 2 :, :] = True
        class_probabilities = np.zeros(mask.shape + (2,), dtype=np.float32)
        class_probabilities[..., 1] = mask.astype(np.float32)
        class_probabilities[..., 0] = 1.0 - class_probabilities[..., 1]
        return DrivableSpaceMask(
            mask=mask,
            class_probabilities=class_probabilities,
            source_sensor_id=sensor_id,
        )

    def _extract_with_opencv(self, frame: np.ndarray) -> np.ndarray | None:
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - optional dependency path
            return None
        image = np.asarray(frame, dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 0, 40], dtype=np.uint8)
        upper = np.array([180, 80, 220], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper) > 0
        kernel = np.ones((7, 7), dtype=np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        height = mask.shape[0]
        mask[: int(height * 0.45), :] = False
        return mask.astype(np.bool_)
