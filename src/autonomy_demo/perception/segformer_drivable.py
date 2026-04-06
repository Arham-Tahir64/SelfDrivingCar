from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import DrivableSpaceMask, SemanticSegMap
from autonomy_demo.perception.internal_types import CameraSegmentationResult
from autonomy_demo.perception.segmentation_tasks import (
    TASK_DRIVABLE,
    TASK_SIDEWALK_NON_DRIVABLE,
    TASK_VEHICLE,
    TASK_VRU,
    cityscapes_probs_to_task_probs,
    derive_boundary_targets,
    remap_cityscapes_to_task,
)

logger = get_logger(__name__)

# Cityscapes class index used for actual roadway.
ROAD_CLASS_ID = 0


class SegFormerDrivableExtractor:
    """Structured SegFormer-based segmentation with drivable output.

    The extractor still publishes ``DrivableSpaceMask`` to keep the public pipeline
    contract stable, but internally it now builds a richer task-space segmentation
    result with temporal confidence fusion, boundary maps, and uncertainty.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        run_every_n_ticks: int = 5,
        max_input_long_edge_px: int | None = None,
        temporal_alpha: float = 0.68,
        ego_prior_strength: float = 0.12,
    ) -> None:
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._model_name = str(model_name)
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
        self._cached_segmentation_result: CameraSegmentationResult | None = None
        self._ran_inference_last_call = False
        self._temporal_alpha = float(np.clip(temporal_alpha, 0.0, 0.95))
        self._ego_prior_strength = float(np.clip(ego_prior_strength, 0.0, 0.35))

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            logger.info("Loading SegFormer from %s on %s", self._model_name, self._device)
            self._processor = SegformerImageProcessor.from_pretrained(self._model_name)
            self._model = SegformerForSemanticSegmentation.from_pretrained(self._model_name)
            self._model.to(self._device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
            logger.info("SegFormer loaded successfully")
            return True
        except Exception:
            logger.warning("Failed to load SegFormer, falling back to heuristic", exc_info=True)
            return False

    def extract(self, frame: np.ndarray, sensor_id: str) -> DrivableSpaceMask | None:
        """Run SegFormer inference on a camera frame.

        Returns cached result on skipped ticks to stay within latency budget.
        Returns None only if model is unavailable (caller falls back to heuristic).
        """
        self._tick_counter += 1
        self._ran_inference_last_call = False

        if self._tick_counter % self._run_every_n_ticks != 1:
            if self._cached_result is not None:
                return self._cached_result

        if not self._ensure_loaded():
            return None

        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3:
            return None

        height, width = image.shape[:2]
        model_image = self._downscale_image(image)

        t0 = time.perf_counter()
        pred_classes: np.ndarray | None = None
        try:
            inputs = self._processor(images=model_image, return_tensors="pt")  # type: ignore[misc]
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)  # type: ignore[misc]

            logits = outputs.logits
            upsampled = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probs = F.softmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            semantic_probabilities = np.moveaxis(probs, 0, -1)
            pred_classes = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            segmentation = self._build_segmentation_result(
                image=image,
                semantic_probabilities=semantic_probabilities,
                pred_classes=pred_classes,
                sensor_id=sensor_id,
            )
            mask = self._refine_drivable_mask(
                semantic_pred_classes=pred_classes,
                road_prob=semantic_probabilities[..., ROAD_CLASS_ID],
                drivable_prob=segmentation.drivable_prob,
                task_probabilities=segmentation.task_probabilities,
                uncertainty=segmentation.uncertainty,
            )
            drivable_prob = np.where(mask, segmentation.drivable_prob, 0.0).astype(np.float32)
            class_probabilities = np.stack(
                [1.0 - drivable_prob, drivable_prob],
                axis=-1,
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
            label_map=pred_classes if pred_classes is not None else np.zeros((height, width), dtype=np.uint8),
            source_sensor_id=sensor_id,
        )
        self._cached_segmentation_result = segmentation
        self._ran_inference_last_call = True
        return result

    def _build_segmentation_result(
        self,
        *,
        image: np.ndarray,
        semantic_probabilities: np.ndarray,
        pred_classes: np.ndarray,
        sensor_id: str,
    ) -> CameraSegmentationResult:
        task_probabilities = cityscapes_probs_to_task_probs(semantic_probabilities)
        task_label_map = remap_cityscapes_to_task(pred_classes)
        drivable_prob = np.asarray(task_probabilities[..., TASK_DRIVABLE], dtype=np.float32)
        lane_boundary_target, curb_boundary_target = derive_boundary_targets(task_label_map)
        lane_marking_prior = self._lane_marking_prior(image, drivable_prob)
        curb_from_semantics = self._curb_boundary_prior(task_probabilities, curb_boundary_target)
        lane_boundary_prob = np.clip((0.55 * lane_marking_prior) + (0.45 * lane_boundary_target), 0.0, 1.0)
        curb_boundary_prob = np.clip(curb_from_semantics, 0.0, 1.0)
        drivable_prob = self._apply_ego_prior(drivable_prob)
        uncertainty = self._normalized_entropy(task_probabilities)

        previous = self._cached_segmentation_result
        if previous is not None and previous.drivable_prob.shape == drivable_prob.shape:
            fusion_strength = self._temporal_alpha * np.clip(1.0 - uncertainty, 0.0, 1.0)
            drivable_prob = self._fuse_probability(previous.drivable_prob, drivable_prob, fusion_strength)
            lane_boundary_prob = self._fuse_probability(
                previous.lane_boundary_prob,
                lane_boundary_prob,
                fusion_strength * 0.75,
            )
            curb_boundary_prob = self._fuse_probability(
                previous.curb_boundary_prob,
                curb_boundary_prob,
                fusion_strength * 0.85,
            )

        return CameraSegmentationResult(
            semantic_label_map=pred_classes.astype(np.uint8),
            task_label_map=task_label_map.astype(np.uint8),
            task_probabilities=task_probabilities.astype(np.float32),
            drivable_prob=drivable_prob.astype(np.float32),
            lane_boundary_prob=lane_boundary_prob.astype(np.float32),
            curb_boundary_prob=curb_boundary_prob.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            source_sensor_id=sensor_id,
            model_name=self._model_name,
            model_version=self._model_name.rsplit("/", maxsplit=1)[-1],
        )

    def _lane_marking_prior(self, image: np.ndarray, drivable_prob: np.ndarray) -> np.ndarray:
        try:
            import cv2  # type: ignore
        except Exception:
            return np.zeros(drivable_prob.shape, dtype=np.float32)
        rgb = np.asarray(image, dtype=np.uint8)
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        white_mask = cv2.inRange(
            hls,
            np.array([0, 165, 0], dtype=np.uint8),
            np.array([255, 255, 140], dtype=np.uint8),
        )
        yellow_mask = cv2.inRange(
            hls,
            np.array([12, 70, 70], dtype=np.uint8),
            np.array([45, 255, 255], dtype=np.uint8),
        )
        bright_mask = cv2.inRange(gray, 185, 255)
        combined = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), bright_mask)
        combined = cv2.GaussianBlur(combined, (5, 5), 0)
        lane_prior = (combined.astype(np.float32) / 255.0) * np.clip(drivable_prob * 1.15, 0.0, 1.0)
        lane_prior[: int(lane_prior.shape[0] * 0.42), :] = 0.0
        return lane_prior.astype(np.float32)

    def _curb_boundary_prior(
        self,
        task_probabilities: np.ndarray,
        curb_boundary_target: np.ndarray,
    ) -> np.ndarray:
        sidewalk_prob = task_probabilities[..., TASK_SIDEWALK_NON_DRIVABLE]
        drivable_prob = task_probabilities[..., TASK_DRIVABLE]
        object_prob = np.maximum(
            task_probabilities[..., TASK_VEHICLE],
            task_probabilities[..., TASK_VRU],
        )
        semantic_boundary = np.abs(drivable_prob - sidewalk_prob)
        semantic_boundary = np.maximum(semantic_boundary, object_prob * 0.35)
        return np.clip((0.5 * semantic_boundary) + (0.5 * curb_boundary_target), 0.0, 1.0)

    def _apply_ego_prior(self, drivable_prob: np.ndarray) -> np.ndarray:
        if self._ego_prior_strength <= 1e-6:
            return drivable_prob.astype(np.float32)
        height, width = drivable_prob.shape
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        vertical = np.clip((ys - 0.55) / 0.45, 0.0, 1.0)
        center = np.exp(-np.square(xs / 0.36)).astype(np.float32)
        prior = vertical * center
        fused = ((1.0 - self._ego_prior_strength) * drivable_prob) + (self._ego_prior_strength * prior)
        return np.clip(fused, 0.0, 1.0).astype(np.float32)

    def _normalized_entropy(self, task_probabilities: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(task_probabilities, dtype=np.float32), 1e-6, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        return (entropy / np.log(probs.shape[-1])).astype(np.float32)

    def _fuse_probability(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        previous = np.asarray(previous, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        return np.clip((alpha * previous) + ((1.0 - alpha) * current), 0.0, 1.0)

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

    def _refine_drivable_mask(
        self,
        *,
        semantic_pred_classes: np.ndarray,
        road_prob: np.ndarray,
        drivable_prob: np.ndarray,
        task_probabilities: np.ndarray,
        uncertainty: np.ndarray,
    ) -> np.ndarray:
        dynamic_threshold = np.clip(0.42 + (uncertainty * 0.12), 0.40, 0.56)
        vehicle_prob = task_probabilities[..., TASK_VEHICLE]
        vru_prob = task_probabilities[..., TASK_VRU]
        candidate = drivable_prob >= dynamic_threshold
        candidate &= vehicle_prob < 0.60
        candidate &= vru_prob < 0.55
        fallback = self._road_mask(semantic_pred_classes, road_prob)
        candidate |= fallback & (drivable_prob >= 0.30)
        candidate = self._morphological_cleanup(candidate)
        filtered = self._connected_component_from_ego_anchor(candidate)
        if filtered is None or not filtered.any():
            return candidate.astype(np.bool_)
        return filtered.astype(np.bool_)

    def _morphological_cleanup(self, candidate: np.ndarray) -> np.ndarray:
        try:
            import cv2  # type: ignore

            kernel = np.ones((5, 5), dtype=np.uint8)
            cleaned = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            return cleaned > 0
        except Exception:
            return candidate.astype(np.bool_)

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
    def model_name(self) -> str:
        return self._model_name

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def ran_inference_last_call(self) -> bool:
        return self._ran_inference_last_call

    @property
    def last_seg_map(self) -> SemanticSegMap | None:
        return self._cached_seg_map

    @property
    def last_segmentation_result(self) -> CameraSegmentationResult | None:
        return self._cached_segmentation_result
