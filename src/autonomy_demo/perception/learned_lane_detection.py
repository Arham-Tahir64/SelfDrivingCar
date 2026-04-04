from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import LaneLine
from autonomy_demo.perception.lane_extraction import _image_to_world_polyline

logger = get_logger(__name__)

# Row anchors: y-coordinates (in normalized image space) where we predict lane x-positions.
# Inspired by UFLD — predicting x-location per row is faster than dense segmentation.
NUM_ROW_ANCHORS = 18
NUM_GRIDDING_CELLS = 100  # discretized x-positions per row
MAX_LANES = 4  # max lanes to detect (left-left, left, right, right-right)


class _LaneBackbone(nn.Module):
    """Lightweight encoder for lane feature extraction.

    Uses a small ResNet-style backbone to extract features, then predicts
    lane x-positions at each row anchor via classification over grid cells
    (the core UFLD idea: lane detection as row-wise classification).
    """

    def __init__(self) -> None:
        super().__init__()
        # Lightweight encoder
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((NUM_ROW_ANCHORS, 8)),
        )
        # Row-wise classification head: for each row anchor, predict grid cell probabilities per lane
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, MAX_LANES * (NUM_GRIDDING_CELLS + 1)),  # +1 for "no lane" class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        feat = self.features(x)  # (B, 256, NUM_ROW_ANCHORS, 8)
        # Reshape: each row anchor gets its own feature vector
        feat = feat.permute(0, 2, 1, 3).reshape(batch_size, NUM_ROW_ANCHORS, -1)
        logits = self.classifier(feat)  # (B, NUM_ROW_ANCHORS, MAX_LANES * (NUM_GRIDDING_CELLS + 1))
        logits = logits.reshape(batch_size, NUM_ROW_ANCHORS, MAX_LANES, NUM_GRIDDING_CELLS + 1)
        return logits


class LearnedLaneExtractor:
    """UFLD-style lane detector: row-wise classification over grid cells.

    This implements the Ultra-Fast-Lane-Detection concept where lane detection
    is formulated as selecting the x-position grid cell at each row anchor.
    Much faster than dense segmentation for lane-specific tasks.

    For this demo, we use a self-supervised initialization from the heuristic
    lane extractor's output to bootstrap the model on CARLA imagery.
    Until enough training data is collected, falls back to heuristic detection.
    """

    MIN_TRAINING_FRAMES = 200  # frames before switching from heuristic to learned
    TRAINING_INTERVAL = 5  # train every N frames during warmup

    def __init__(
        self,
        *,
        device: str = "cuda",
        run_every_n_ticks: int = 1,
        allow_online_training: bool = True,
    ) -> None:
        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = _LaneBackbone().to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        self._training_buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self._frames_seen = 0
        self._trained = False
        self._last_inference_ms: float = 0.0
        self._run_every_n_ticks = max(int(run_every_n_ticks), 1)
        self._allow_online_training = allow_online_training
        self._tick_counter = 0
        self._ran_inference_last_call = False
        self._row_anchors: np.ndarray | None = None
        logger.info("LearnedLaneExtractor initialized on %s", self._device)

    def _get_row_anchors(self, image_height: int) -> np.ndarray:
        """Row anchor y-coordinates in pixel space (bottom to top of lower half)."""
        if self._row_anchors is not None and len(self._row_anchors) == NUM_ROW_ANCHORS:
            return self._row_anchors
        start_y = int(image_height * 0.95)
        end_y = int(image_height * 0.45)
        self._row_anchors = np.linspace(start_y, end_y, NUM_ROW_ANCHORS).astype(np.float32)
        return self._row_anchors

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for the backbone: resize, normalize, to tensor."""
        import cv2  # type: ignore

        resized = cv2.resize(image, (288, 160))  # lightweight input size
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        # ImageNet-style normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self._device)

    def _heuristic_lanes_to_targets(
        self, lanes: list[LaneLine], image_width: int, image_height: int
    ) -> np.ndarray | None:
        """Convert heuristic lane polylines to grid-cell classification targets.

        Returns (NUM_ROW_ANCHORS, MAX_LANES) array of grid cell indices,
        or NUM_GRIDDING_CELLS for "no lane" at that row.
        """
        row_anchors = self._get_row_anchors(image_height)
        targets = np.full((NUM_ROW_ANCHORS, MAX_LANES), NUM_GRIDDING_CELLS, dtype=np.int64)

        for lane_idx, lane in enumerate(lanes[:MAX_LANES]):
            if lane.polyline_image is None or len(lane.polyline_image) < 2:
                continue
            # Interpolate lane x-position at each row anchor
            lane_ys = lane.polyline_image[:, 1]
            lane_xs = lane.polyline_image[:, 0]
            sort_idx = np.argsort(lane_ys)
            lane_ys = lane_ys[sort_idx]
            lane_xs = lane_xs[sort_idx]

            for row_idx, anchor_y in enumerate(row_anchors):
                if anchor_y < lane_ys[0] or anchor_y > lane_ys[-1]:
                    continue
                x_at_anchor = float(np.interp(anchor_y, lane_ys, lane_xs))
                grid_cell = int(np.clip(x_at_anchor / image_width * NUM_GRIDDING_CELLS, 0, NUM_GRIDDING_CELLS - 1))
                targets[row_idx, lane_idx] = grid_cell

        return targets

    def _collect_training_sample(
        self, image: np.ndarray, lanes: list[LaneLine], image_width: int, image_height: int
    ) -> None:
        """Store a (image, target) pair from heuristic output for self-supervised training."""
        if len(lanes) < 2:
            return
        targets = self._heuristic_lanes_to_targets(lanes, image_width, image_height)
        if targets is None:
            return
        import cv2  # type: ignore

        resized = cv2.resize(image, (288, 160))
        self._training_buffer.append((resized, targets))
        if len(self._training_buffer) > 1000:
            self._training_buffer = self._training_buffer[-1000:]

    def _train_step(self) -> float:
        """Run one training step on the collected buffer."""
        if len(self._training_buffer) < 32:
            return 0.0

        self._model.train()
        indices = np.random.choice(len(self._training_buffer), size=min(32, len(self._training_buffer)), replace=False)
        images = []
        targets = []
        for idx in indices:
            img, tgt = self._training_buffer[idx]
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            images.append(tensor)
            targets.append(torch.from_numpy(tgt))

        batch_images = torch.stack(images).to(self._device)
        batch_targets = torch.stack(targets).to(self._device)

        logits = self._model(batch_images)  # (B, NUM_ROW_ANCHORS, MAX_LANES, NUM_GRIDDING_CELLS+1)
        # Cross-entropy loss per row per lane
        loss = F.cross_entropy(
            logits.reshape(-1, NUM_GRIDDING_CELLS + 1),
            batch_targets.reshape(-1),
        )

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._model.eval()
        return float(loss.item())

    def _decode_predictions(
        self, logits: torch.Tensor, image_width: int, image_height: int
    ) -> list[np.ndarray]:
        """Decode model output to lane polylines in image space."""
        row_anchors = self._get_row_anchors(image_height)
        probs = F.softmax(logits.squeeze(0), dim=-1)  # (NUM_ROW_ANCHORS, MAX_LANES, NUM_GRIDDING_CELLS+1)
        pred_cells = probs[..., :NUM_GRIDDING_CELLS].argmax(dim=-1).cpu().numpy()  # (rows, lanes)
        no_lane_prob = probs[..., NUM_GRIDDING_CELLS].cpu().numpy()  # (rows, lanes)

        lanes: list[np.ndarray] = []
        for lane_idx in range(MAX_LANES):
            points: list[list[float]] = []
            for row_idx in range(NUM_ROW_ANCHORS):
                if no_lane_prob[row_idx, lane_idx] > 0.5:
                    continue
                x = float(pred_cells[row_idx, lane_idx]) / NUM_GRIDDING_CELLS * image_width
                y = float(row_anchors[row_idx])
                points.append([x, y])
            if len(points) >= 4:
                lanes.append(np.array(points, dtype=np.float32))
        return lanes

    def extract(
        self,
        frame: np.ndarray,
        *,
        sensor_id: str = "front_camera",
        ego_world_xyz: np.ndarray | None = None,
        ego_yaw_rad: float = 0.0,
        ego_yaw_rate_rad_s: float = 0.0,
        heuristic_lanes: list[LaneLine] | None = None,
    ) -> list[LaneLine] | None:
        """Run learned lane detection.

        During warmup (first MIN_TRAINING_FRAMES), collects heuristic lane
        output as self-supervised training data and returns None (caller uses heuristic).
        After warmup, runs the trained model for inference.
        """
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3:
            self._ran_inference_last_call = False
            return None
        image_height, image_width = image.shape[:2]

        self._tick_counter += 1
        self._frames_seen += 1
        self._ran_inference_last_call = False

        # Warmup phase: collect training data from heuristic extractor
        if not self._trained:
            if not self._allow_online_training:
                return None
            if heuristic_lanes is not None:
                self._collect_training_sample(image, heuristic_lanes, image_width, image_height)

            # Train periodically during warmup
            if self._frames_seen % self.TRAINING_INTERVAL == 0 and len(self._training_buffer) >= 32:
                loss = self._train_step()
                if self._frames_seen % 50 == 0:
                    logger.info(
                        "Lane model training: frame=%d, buffer=%d, loss=%.4f",
                        self._frames_seen, len(self._training_buffer), loss,
                    )

            if self._frames_seen >= self.MIN_TRAINING_FRAMES and len(self._training_buffer) >= 100:
                # Final training burst
                for _ in range(10):
                    self._train_step()
                self._trained = True
                logger.info(
                    "Lane model warmup complete after %d frames (%d training samples)",
                    self._frames_seen, len(self._training_buffer),
                )
            return None  # caller uses heuristic during warmup

        if self._tick_counter % self._run_every_n_ticks != 1:
            return None

        # Inference phase
        t0 = time.perf_counter()
        try:
            tensor = self._image_to_tensor(image)
            with torch.no_grad():
                logits = self._model(tensor)
            polylines = self._decode_predictions(logits, image_width, image_height)
        except Exception:
            logger.warning("Learned lane inference failed", exc_info=True)
            return None
        finally:
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        if len(polylines) < 2:
            return None  # not enough lanes detected, fall back

        ego_xyz = np.zeros(3, dtype=np.float32) if ego_world_xyz is None else np.asarray(ego_world_xyz, dtype=np.float32)

        lane_ids = ["lane_left", "lane_right", "lane_far_left", "lane_far_right"]
        lanes: list[LaneLine] = []
        for idx, polyline in enumerate(polylines[:MAX_LANES]):
            lane_id = lane_ids[idx] if idx < len(lane_ids) else f"lane_{idx}"
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
                    confidence=0.75,
                    source_modality="learned",
                    source_sensor_ids=[sensor_id],
                    position_estimate_kind="learned_lane_detection",
                )
            )
        self._ran_inference_last_call = True
        return lanes

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def ran_inference_last_call(self) -> bool:
        return self._ran_inference_last_call
