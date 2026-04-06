"""DepthAnything V2 monocular depth estimator."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import DepthMap

logger = get_logger(__name__)


class DepthAnythingEstimator:
    """DepthAnything-V2-Small monocular depth estimator.

    Uses ``depth-anything/Depth-Anything-V2-Small-hf`` from HuggingFace.
    Runs on GPU when available, falls back to CPU.

    Throttled to ``run_every_n_ticks`` and offset by 2 ticks from SegFormer
    so both models don't spike the GPU on the same frame.
    Cached result is returned on skipped ticks.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        run_every_n_ticks: int = 5,
        max_input_long_edge_px: int | None = 518,
        tick_offset: int = 2,
    ) -> None:
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._load_attempted = False
        self._last_inference_ms: float = 0.0
        self._run_every_n_ticks = max(int(run_every_n_ticks), 1)
        self._max_input_long_edge_px = (
            None
            if max_input_long_edge_px is None or int(max_input_long_edge_px) <= 0
            else int(max_input_long_edge_px)
        )
        self._tick_offset = int(tick_offset)
        self._tick_counter: int = 0
        self._cached_result: DepthMap | None = None
        self._ran_inference_last_call = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            logger.info("Loading DepthAnything-V2-Small from %s on %s", model_name, self._device)
            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info("DepthAnything-V2-Small loaded successfully")
            return True
        except Exception:
            logger.warning("Failed to load DepthAnything-V2-Small, depth unavailable", exc_info=True)
            return False

    def extract(self, frame: np.ndarray, sensor_id: str) -> DepthMap | None:
        """Run depth estimation on a camera frame.

        Returns cached result on skipped ticks to stay within latency budget.
        Returns None only if model is unavailable.
        """
        self._tick_counter += 1
        self._ran_inference_last_call = False

        # Offset tick schedule so depth and segformer don't collide
        if (self._tick_counter + self._tick_offset) % self._run_every_n_ticks != 0:
            if self._cached_result is not None:
                return self._cached_result
            # No cache yet — fall through to run inference

        if not self._ensure_loaded():
            return None

        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3:
            return None

        height, width = image.shape[:2]
        model_image = self._downscale_image(image)

        t0 = time.perf_counter()
        try:
            inputs = self._processor(images=model_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # outputs.predicted_depth is (1, H_model, W_model)
            depth = outputs.predicted_depth.squeeze(0)  # (H_model, W_model)

            # Upsample to original image resolution
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()

            # Normalize to 0–1 (higher = closer)
            d_min = depth.min()
            d_max = depth.max()
            depth = (depth - d_min) / (d_max - d_min + 1e-6)

        except Exception:
            logger.warning("DepthAnything inference failed", exc_info=True)
            return None
        finally:
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        result = DepthMap(
            depth=depth.astype(np.float32),
            source_sensor_id=sensor_id,
        )
        self._cached_result = result
        self._ran_inference_last_call = True
        return result

    def _downscale_image(self, image: np.ndarray) -> np.ndarray:
        if self._max_input_long_edge_px is None:
            return image
        height, width = image.shape[:2]
        long_edge = max(height, width)
        if long_edge <= self._max_input_long_edge_px:
            return image
        scale = self._max_input_long_edge_px / float(long_edge)
        resized_width = max(int(round(width * scale)), 1)
        resized_height = max(int(round(height * scale)), 1)
        try:
            import cv2

            return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        except Exception:
            return image

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def ran_inference_last_call(self) -> bool:
        return self._ran_inference_last_call

    @property
    def last_depth_map(self) -> DepthMap | None:
        return self._cached_result
