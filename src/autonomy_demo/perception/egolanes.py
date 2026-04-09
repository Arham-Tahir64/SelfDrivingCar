from __future__ import annotations

import time
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import LaneLine
from autonomy_demo.perception.lane_extraction import _image_to_world_polyline

logger = get_logger(__name__)

_VALID_BACKENDS = {"heuristic", "online_train", "egolanes_onnx"}


class EgoLanesExtractor:
    """ONNX-backed ego-lane segmenter that emits the project's LaneLine contract."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_path: str,
        run_every_n_ticks: int = 1,
        confidence_threshold: float = 0.45,
        max_input_long_edge_px: int | None = None,
        smoothing_alpha: float = 0.55,
        turn_smoothing_disable_yaw_rate_rad_s: float = 0.20,
        turn_smoothing_resume_yaw_rate_rad_s: float = 0.12,
    ) -> None:
        self._device = str(device).lower()
        self._model_path = str(model_path or "")
        self._run_every_n_ticks = max(int(run_every_n_ticks), 1)
        self._confidence_threshold = float(np.clip(confidence_threshold, 0.05, 0.95))
        self._max_input_long_edge_px = (
            None
            if max_input_long_edge_px is None or int(max_input_long_edge_px) <= 0
            else int(max_input_long_edge_px)
        )
        self._smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 0.95))
        self._turn_smoothing_disable_yaw_rate_rad_s = float(turn_smoothing_disable_yaw_rate_rad_s)
        self._turn_smoothing_resume_yaw_rate_rad_s = float(turn_smoothing_resume_yaw_rate_rad_s)

        self._session = None
        self._session_load_attempted = False
        self._load_error: str | None = None
        self._input_name: str | None = None
        self._input_shape: list[Any] | tuple[Any, ...] | None = None
        self._output_names: list[str] = []

        self._tick_counter = 0
        self._last_inference_ms = 0.0
        self._ran_inference_last_call = False
        self._cached_lanes: list[LaneLine] | None = None
        self._previous_polylines: dict[str, np.ndarray] = {}
        self._turn_smoothing_suppressed = False

    def extract(
        self,
        frame: np.ndarray,
        *,
        sensor_id: str = "front_camera",
        ego_world_xyz: np.ndarray | None = None,
        ego_yaw_rad: float = 0.0,
        ego_yaw_rate_rad_s: float = 0.0,
    ) -> list[LaneLine] | None:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3:
            self._ran_inference_last_call = False
            return None

        self._tick_counter += 1
        self._ran_inference_last_call = False
        suppress_temporal_smoothing = self._update_turn_smoothing_state(ego_yaw_rate_rad_s)

        if self._tick_counter % self._run_every_n_ticks != 1 and self._cached_lanes is not None:
            return self._clone_lanes(self._cached_lanes)

        if not self._ensure_loaded():
            return None

        t0 = time.perf_counter()
        try:
            model_input = self._prepare_model_input(image)
            outputs = self._session.run(self._output_names or None, {self._input_name: model_input})
            probabilities = self._extract_probabilities(outputs)
            probabilities = self._resize_probabilities(probabilities, image.shape[:2])
            lanes = self._probabilities_to_lanes(
                probabilities,
                image=image,
                sensor_id=sensor_id,
                ego_world_xyz=ego_world_xyz,
                ego_yaw_rad=ego_yaw_rad,
                suppress_temporal_smoothing=suppress_temporal_smoothing,
            )
        except Exception:
            logger.warning("EgoLanes inference failed", exc_info=True)
            return None
        finally:
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        if len(lanes) < 2:
            return None
        self._cached_lanes = self._clone_lanes(lanes)
        self._ran_inference_last_call = True
        return lanes

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._session_load_attempted:
            return False
        self._session_load_attempted = True

        if not self._model_path:
            self._load_error = "missing model path"
            logger.warning("EgoLanes disabled: no ONNX model path configured")
            return False

        try:
            import onnxruntime as ort
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("EgoLanes disabled: onnxruntime unavailable", exc_info=True)
            return False

        try:
            available_providers = list(ort.get_available_providers())
            requested_providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device != "cpu"
                else ["CPUExecutionProvider"]
            )
            providers = [provider for provider in requested_providers if provider in available_providers]
            if not providers:
                providers = available_providers[:1]
            if self._device != "cpu" and providers and providers[0] != "CUDAExecutionProvider":
                logger.warning(
                    "EgoLanes requested %s but falling back to %s",
                    self._device,
                    providers[0],
                )
            self._session = ort.InferenceSession(self._model_path, providers=providers)
            input_meta = self._session.get_inputs()[0]
            self._input_name = str(input_meta.name)
            self._input_shape = list(getattr(input_meta, "shape", []) or [])
            self._output_names = [str(output.name) for output in self._session.get_outputs()]
            logger.info("EgoLanes loaded from %s", self._model_path)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Failed to load EgoLanes ONNX session", exc_info=True)
            return False

    def _prepare_model_input(self, image: np.ndarray) -> np.ndarray:
        resized = self._downscale_image(image)
        layout = self._infer_input_layout()
        target_height, target_width = self._infer_input_hw(resized.shape[:2])
        resized = self._resize_image(resized, (target_height, target_width))
        normalized = resized.astype(np.float32) / 255.0
        if layout == "nchw":
            return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)
        return normalized[None, ...].astype(np.float32)

    def _infer_input_layout(self) -> str:
        shape = list(self._input_shape or [])
        if len(shape) != 4:
            return "nchw"
        if shape[-1] == 3:
            return "nhwc"
        return "nchw"

    def _infer_input_hw(self, image_hw: tuple[int, int]) -> tuple[int, int]:
        shape = list(self._input_shape or [])
        if len(shape) == 4:
            if self._infer_input_layout() == "nchw":
                height = shape[2]
                width = shape[3]
            else:
                height = shape[1]
                width = shape[2]
            if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
                return int(height), int(width)
        return image_hw

    def _extract_probabilities(self, outputs: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        if not outputs:
            raise ValueError("EgoLanes produced no outputs")
        output = np.asarray(outputs[0], dtype=np.float32)
        if output.ndim == 4:
            output = output[0]
        if output.ndim != 3:
            raise ValueError(f"unexpected EgoLanes output shape: {output.shape}")
        if output.shape[0] == 3:
            logits = output
        elif output.shape[-1] == 3:
            logits = np.moveaxis(output, -1, 0)
        else:
            raise ValueError(f"unexpected EgoLanes channel dimension: {output.shape}")

        if self._looks_like_probabilities(logits):
            probabilities = logits / np.maximum(np.sum(logits, axis=0, keepdims=True), 1e-6)
        else:
            logits = logits - np.max(logits, axis=0, keepdims=True)
            exp_logits = np.exp(logits).astype(np.float32)
            probabilities = exp_logits / np.maximum(np.sum(exp_logits, axis=0, keepdims=True), 1e-6)
        return np.moveaxis(probabilities, 0, -1).astype(np.float32)

    def _looks_like_probabilities(self, logits: np.ndarray) -> bool:
        if np.any(logits < -1e-4) or np.any(logits > 1.05):
            return False
        channel_sum = np.sum(logits, axis=0)
        return bool(np.mean(np.abs(channel_sum - 1.0)) < 0.1)

    def _resize_probabilities(self, probabilities: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
        target_height, target_width = image_hw
        source_height, source_width = probabilities.shape[:2]
        if (source_height, source_width) == (target_height, target_width):
            return probabilities.astype(np.float32)

        try:
            import cv2  # type: ignore

            channels = [
                cv2.resize(
                    np.asarray(probabilities[..., channel_index], dtype=np.float32),
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                for channel_index in range(probabilities.shape[2])
            ]
            return np.stack(channels, axis=-1).astype(np.float32)
        except Exception:
            y_indices = np.linspace(0, source_height - 1, target_height).astype(np.int32)
            x_indices = np.linspace(0, source_width - 1, target_width).astype(np.int32)
            return probabilities[y_indices[:, None], x_indices[None, :], :].astype(np.float32)

    def _probabilities_to_lanes(
        self,
        probabilities: np.ndarray,
        *,
        image: np.ndarray,
        sensor_id: str,
        ego_world_xyz: np.ndarray | None,
        ego_yaw_rad: float,
        suppress_temporal_smoothing: bool,
    ) -> list[LaneLine]:
        ego_xyz = np.asarray(
            np.zeros(3, dtype=np.float32) if ego_world_xyz is None else ego_world_xyz,
            dtype=np.float32,
        )
        lanes: list[LaneLine] = []
        lane_specs = [
            ("lane_left", probabilities[..., 0], "left"),
            ("lane_right", probabilities[..., 1], "right"),
        ]
        for lane_id, lane_probability, side in lane_specs:
            polyline = self._mask_to_polyline(
                lane_probability,
                side=side,
                image_shape=image.shape,
            )
            if polyline is None:
                self._previous_polylines.pop(lane_id, None)
                continue
            polyline = self._maybe_smooth_polyline(
                lane_id,
                polyline,
                suppress_temporal_smoothing=suppress_temporal_smoothing,
            )
            lanes.append(
                LaneLine(
                    lane_id=lane_id,
                    polyline_image=polyline,
                    polyline_world=_image_to_world_polyline(
                        polyline,
                        image.shape,
                        sensor_id=sensor_id,
                        ego_world_xyz=ego_xyz,
                        ego_yaw_rad=ego_yaw_rad,
                    ),
                    line_type=LaneLineType.SOLID,
                    confidence=self._polyline_confidence(lane_probability, polyline),
                    source_modality="learned",
                    source_sensor_ids=[sensor_id],
                    position_estimate_kind="egolanes_segmentation",
                )
            )
        return lanes

    def _mask_to_polyline(
        self,
        lane_probability: np.ndarray,
        *,
        side: str,
        image_shape: tuple[int, int, int],
    ) -> np.ndarray | None:
        lane_probability = np.asarray(lane_probability, dtype=np.float32)
        height, width = image_shape[:2]
        if lane_probability.shape != (height, width):
            return None

        weighted = lane_probability * self._side_prior(width, height, side)
        binary = weighted >= self._confidence_threshold
        binary = self._clean_binary_mask(binary)
        if not np.any(binary):
            return None

        sample_ys = np.linspace(height * 0.96, height * 0.50, num=6, dtype=np.float32)
        points: list[list[float]] = []
        row_half_window = max(height // 60, 2)
        for anchor_y in sample_ys:
            center_y = int(np.clip(round(float(anchor_y)), 0, height - 1))
            y_low = max(center_y - row_half_window, 0)
            y_high = min(center_y + row_half_window + 1, height)
            window_mask = binary[y_low:y_high, :]
            if not np.any(window_mask):
                continue
            xs = np.where(window_mask)
            weights = weighted[y_low:y_high, :][window_mask]
            if weights.size == 0:
                continue
            x_value = float(np.average(xs[1], weights=weights))
            points.append([x_value, float(anchor_y)])

        if len(points) < 4:
            return None
        polyline = np.asarray(points, dtype=np.float32)
        fit_degree = 2 if len(polyline) >= 5 else 1
        fit = np.polyfit(polyline[:, 1], polyline[:, 0], deg=fit_degree)
        fitted_x = np.polyval(fit, sample_ys).astype(np.float32)
        return np.stack([fitted_x, sample_ys], axis=1).astype(np.float32)

    def _side_prior(self, width: int, height: int, side: str) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        target_x = 0.32 if side == "left" else 0.68
        lateral_prior = np.exp(-np.square((xs - target_x) / 0.22)).astype(np.float32)
        vertical_prior = np.clip((ys - 0.35) / 0.65, 0.0, 1.0).astype(np.float32)
        return (lateral_prior * vertical_prior).astype(np.float32)

    def _clean_binary_mask(self, binary: np.ndarray) -> np.ndarray:
        try:
            import cv2  # type: ignore

            cleaned = binary.astype(np.uint8) * 255
            kernel = np.ones((5, 5), dtype=np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            return self._select_main_component(cleaned > 0)
        except Exception:
            return binary.astype(np.bool_)

    def _select_main_component(self, binary: np.ndarray) -> np.ndarray:
        if not np.any(binary):
            return binary.astype(np.bool_)
        try:
            import cv2  # type: ignore

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary.astype(np.uint8),
                connectivity=8,
            )
            if num_labels <= 1:
                return binary.astype(np.bool_)
            best_label = 1
            best_score = -1
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                top = int(stats[label, cv2.CC_STAT_TOP])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                bottom = top + height
                score = area + (bottom * 4)
                if score > best_score:
                    best_label = label
                    best_score = score
            return labels == best_label
        except Exception:
            return binary.astype(np.bool_)

    def _polyline_confidence(self, lane_probability: np.ndarray, polyline: np.ndarray) -> float:
        sampled_values: list[float] = []
        height, width = lane_probability.shape
        for x_value, y_value in polyline:
            x_index = int(np.clip(round(float(x_value)), 0, width - 1))
            y_index = int(np.clip(round(float(y_value)), 0, height - 1))
            sampled_values.append(float(lane_probability[y_index, x_index]))
        return float(np.clip(np.mean(sampled_values) if sampled_values else 0.0, 0.0, 0.99))

    def _update_turn_smoothing_state(self, ego_yaw_rate_rad_s: float) -> bool:
        yaw_rate_abs = abs(float(ego_yaw_rate_rad_s))
        if self._turn_smoothing_suppressed:
            if yaw_rate_abs <= self._turn_smoothing_resume_yaw_rate_rad_s:
                self._turn_smoothing_suppressed = False
        elif yaw_rate_abs >= self._turn_smoothing_disable_yaw_rate_rad_s:
            self._turn_smoothing_suppressed = True
        return self._turn_smoothing_suppressed

    def _maybe_smooth_polyline(
        self,
        lane_id: str,
        polyline: np.ndarray,
        *,
        suppress_temporal_smoothing: bool,
    ) -> np.ndarray:
        polyline = np.asarray(polyline, dtype=np.float32)
        if suppress_temporal_smoothing:
            self._previous_polylines[lane_id] = polyline.copy()
            return polyline
        previous = self._previous_polylines.get(lane_id)
        if previous is None or previous.shape != polyline.shape:
            self._previous_polylines[lane_id] = polyline.copy()
            return polyline
        smoothed = (
            (self._smoothing_alpha * previous) + ((1.0 - self._smoothing_alpha) * polyline)
        ).astype(np.float32)
        self._previous_polylines[lane_id] = smoothed
        return smoothed

    def _clone_lanes(self, lanes: list[LaneLine]) -> list[LaneLine]:
        clones: list[LaneLine] = []
        for lane in lanes:
            clones.append(
                LaneLine(
                    lane_id=lane.lane_id,
                    polyline_image=np.asarray(lane.polyline_image, dtype=np.float32).copy(),
                    polyline_world=np.asarray(lane.polyline_world, dtype=np.float32).copy(),
                    line_type=lane.line_type,
                    confidence=float(lane.confidence),
                    source_modality=str(lane.source_modality),
                    source_sensor_ids=list(lane.source_sensor_ids),
                    position_estimate_kind=str(lane.position_estimate_kind),
                )
            )
        return clones

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
        return self._resize_image(image, (resized_height, resized_width))

    def _resize_image(self, image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        target_height, target_width = size_hw
        try:
            import cv2  # type: ignore

            return cv2.resize(
                image,
                (int(target_width), int(target_height)),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            source_height, source_width = image.shape[:2]
            y_indices = np.linspace(0, source_height - 1, target_height).astype(np.int32)
            x_indices = np.linspace(0, source_width - 1, target_width).astype(np.int32)
            return image[y_indices[:, None], x_indices[None, :], :]

    @property
    def model_name(self) -> str:
        return "EgoLanes"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def ran_inference_last_call(self) -> bool:
        return self._ran_inference_last_call


def normalize_lane_backend(value: str | None, *, default: str = "heuristic") -> str:
    backend = str(value or default).strip().lower()
    return backend if backend in _VALID_BACKENDS else default
