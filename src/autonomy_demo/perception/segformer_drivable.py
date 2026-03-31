from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import DrivableSpaceMask

logger = get_logger(__name__)

# Cityscapes class indices considered "drivable"
# 0: road, 1: sidewalk (included for conservative drivable area)
DRIVABLE_CLASS_IDS = {0, 1}


class SegFormerDrivableExtractor:
    """SegFormer-B0 drivable space extractor using Cityscapes-finetuned weights.

    Uses ``nvidia/segformer-b0-finetuned-cityscapes-1024-1024`` from HuggingFace.
    Runs on GPU when available, falls back to CPU.

    Throttled to ``run_every_n_ticks`` to keep pipeline latency in budget.
    Cached result is returned on skipped ticks.
    """

    def __init__(self, *, device: str = "cuda", run_every_n_ticks: int = 5) -> None:
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._load_attempted = False
        self._last_inference_ms: float = 0.0
        self._run_every_n_ticks = run_every_n_ticks
        self._tick_counter: int = 0
        self._cached_result: DrivableSpaceMask | None = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
            logger.info("Loading SegFormer-B0 from %s on %s", model_name, self._device)
            self._processor = SegformerImageProcessor.from_pretrained(model_name)
            self._model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
            logger.info("SegFormer-B0 loaded successfully")
            return True
        except Exception:
            logger.warning("Failed to load SegFormer-B0, falling back to heuristic", exc_info=True)
            return False

    def extract(self, frame: np.ndarray, sensor_id: str) -> DrivableSpaceMask | None:
        """Run SegFormer inference on a camera frame.

        Returns cached result on skipped ticks to stay within latency budget.
        Returns None only if model is unavailable (caller falls back to heuristic).
        """
        self._tick_counter += 1

        # Return cached result on non-inference ticks
        if self._tick_counter % self._run_every_n_ticks != 1:
            if self._cached_result is not None:
                return self._cached_result
            # No cache yet — fall through to run inference on first tick

        if not self._ensure_loaded():
            return None

        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3:
            return None

        height, width = image.shape[:2]

        t0 = time.perf_counter()
        try:
            inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[misc]
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)  # type: ignore[misc]

            # Upsample logits to original image size
            logits = outputs.logits  # (1, num_classes, H/4, W/4)
            upsampled = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

            # Softmax probabilities
            probs = F.softmax(upsampled, dim=1)  # (1, 19, H, W)
            pred_classes = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

            # Build drivable mask: union of road + sidewalk classes
            mask = np.isin(pred_classes, list(DRIVABLE_CLASS_IDS))

            # Build 2-class probability: [not-drivable, drivable]
            probs_np = probs.squeeze(0).cpu().numpy()  # (19, H, W)
            drivable_prob = np.zeros((height, width), dtype=np.float32)
            for cid in DRIVABLE_CLASS_IDS:
                drivable_prob += probs_np[cid]
            drivable_prob = np.clip(drivable_prob, 0.0, 1.0)

            class_probabilities = np.stack(
                [1.0 - drivable_prob, drivable_prob], axis=-1
            ).astype(np.float32)

        except Exception:
            logger.warning("SegFormer inference failed", exc_info=True)
            return None
        finally:
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        result = DrivableSpaceMask(
            mask=mask.astype(np.bool_),
            class_probabilities=class_probabilities,
            source_sensor_id=sensor_id,
        )
        self._cached_result = result
        return result

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms
