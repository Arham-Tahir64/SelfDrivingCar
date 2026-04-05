from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import DrivableSpaceMask, SemanticSegMap

logger = get_logger(__name__)

# Cityscapes class index used for actual roadway.
ROAD_CLASS_ID = 0


class SegFormerDrivableExtractor:
    """SegFormer-B0 drivable space extractor using Cityscapes-finetuned weights.

    Uses ``nvidia/segformer-b0-finetuned-cityscapes-1024-1024`` from HuggingFace.
    Runs on GPU when available, falls back to CPU.

    Throttled to ``run_every_n_ticks`` to keep pipeline latency in budget.
    Cached result is returned on skipped ticks.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        run_every_n_ticks: int = 5,
        max_input_long_edge_px: int | None = None,
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
        self._tick_counter: int = 0
        self._cached_result: DrivableSpaceMask | None = None
        self._cached_seg_map: SemanticSegMap | None = None
        self._ran_inference_last_call = False

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
        self._ran_inference_last_call = False

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
        model_image = self._downscale_image(image)

        t0 = time.perf_counter()
        try:
            inputs = self._processor(images=model_image, return_tensors="pt")  # type: ignore[misc]
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

            road_prob = probs.squeeze(0)[ROAD_CLASS_ID].cpu().numpy().astype(np.float32)
            mask = self._road_mask(pred_classes, road_prob)

            # Build 2-class probability: [not-drivable, drivable]
            drivable_prob = np.where(mask, road_prob, 0.0).astype(np.float32)

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
        self._cached_seg_map = SemanticSegMap(
            label_map=pred_classes.astype(np.uint8),
            source_sensor_id=sensor_id,
        )
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
            import cv2  # type: ignore

            return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        except Exception:
            return image

    def _road_mask(self, pred_classes: np.ndarray, road_prob: np.ndarray) -> np.ndarray:
        candidate = (pred_classes == ROAD_CLASS_ID) & (road_prob >= 0.35)
        if not candidate.any():
            candidate = road_prob >= 0.5
        filtered = self._connected_component_from_ego_anchor(candidate)
        if filtered is None or not filtered.any():
            return candidate.astype(np.bool_)
        return filtered.astype(np.bool_)

    def _connected_component_from_ego_anchor(self, candidate: np.ndarray) -> np.ndarray | None:
        if candidate.ndim != 2 or not candidate.any():
            return None
        height, width = candidate.shape
        seed = np.zeros_like(candidate, dtype=np.bool_)
        seed[int(height * 0.72) :, int(width * 0.35) : int(width * 0.65)] = True
        bottom_seed = np.zeros_like(candidate, dtype=np.bool_)
        bottom_seed[int(height * 0.86) :, :] = True

        try:
            import cv2  # type: ignore

            labels, stats = self._connected_components_cv2(candidate, cv2)
            if labels is None or stats is None:
                return candidate.astype(np.bool_)
            selected = self._select_component_label(labels, stats, seed, bottom_seed)
            if selected is None:
                return candidate.astype(np.bool_)
            return labels == selected
        except Exception:
            return candidate.astype(np.bool_)

    def _connected_components_cv2(self, candidate: np.ndarray, cv2):
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
        )
        if num_labels <= 1:
            return None, None
        return labels, stats

    def _select_component_label(
        self,
        labels: np.ndarray,
        stats: np.ndarray,
        seed: np.ndarray,
        bottom_seed: np.ndarray,
    ) -> int | None:
        best_label: int | None = None
        best_score = -1
        for label in range(1, stats.shape[0]):
            component = labels == label
            if not component.any():
                continue
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
        return best_label

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def ran_inference_last_call(self) -> bool:
        return self._ran_inference_last_call

    @property
    def last_seg_map(self) -> SemanticSegMap | None:
        return self._cached_seg_map
