from __future__ import annotations

import numpy as np
import pytest

from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import DrivableSpaceMask, LaneLine


# ---------- SegFormer drivable space tests ----------

def test_segformer_extractor_loads_and_runs() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu")
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    result = extractor.extract(image, "front_camera")
    assert result is not None
    assert isinstance(result, DrivableSpaceMask)
    assert result.mask.shape == (256, 512)
    assert result.mask.dtype == np.bool_
    assert result.class_probabilities.shape == (256, 512, 2)
    assert extractor.last_inference_ms > 0


def test_segformer_extractor_returns_none_for_invalid_input() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu")
    # 2D grayscale image — should return None
    result = extractor.extract(np.zeros((100, 100), dtype=np.uint8), "front_camera")
    assert result is None


def test_segformer_drivable_mask_is_not_all_zeros() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu")
    # Road-like image: gray bottom half
    image = np.zeros((256, 512, 3), dtype=np.uint8)
    image[128:, :] = 128  # gray road-like area
    result = extractor.extract(image, "front_camera")
    assert result is not None
    # At least some pixels should be drivable
    assert result.mask.any()


# ---------- Learned lane detector tests ----------

def test_learned_lane_extractor_warmup_returns_none() -> None:
    from autonomy_demo.perception.learned_lane_detection import LearnedLaneExtractor

    extractor = LearnedLaneExtractor(device="cpu")
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    # During warmup, should return None (caller uses heuristic)
    result = extractor.extract(image, sensor_id="front_camera")
    assert result is None
    assert not extractor.is_trained


def test_learned_lane_extractor_collects_training_data() -> None:
    from autonomy_demo.perception.learned_lane_detection import LearnedLaneExtractor

    extractor = LearnedLaneExtractor(device="cpu")
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)

    # Provide heuristic lanes as training targets
    fake_lanes = [
        LaneLine(
            lane_id="lane_left",
            polyline_image=np.array([[200, 240], [210, 200], [220, 160], [230, 120]], dtype=np.float32),
            polyline_world=np.zeros((4, 3), dtype=np.float32),
            line_type=LaneLineType.SOLID,
            confidence=0.8,
        ),
        LaneLine(
            lane_id="lane_right",
            polyline_image=np.array([[300, 240], [310, 200], [320, 160], [330, 120]], dtype=np.float32),
            polyline_world=np.zeros((4, 3), dtype=np.float32),
            line_type=LaneLineType.SOLID,
            confidence=0.8,
        ),
    ]

    result = extractor.extract(image, sensor_id="front_camera", heuristic_lanes=fake_lanes)
    assert result is None  # still in warmup
    assert len(extractor._training_buffer) == 1


def test_learned_lane_extractor_invalid_input_returns_none() -> None:
    from autonomy_demo.perception.learned_lane_detection import LearnedLaneExtractor

    extractor = LearnedLaneExtractor(device="cpu")
    extractor._trained = True  # Force trained state
    # 2D grayscale
    result = extractor.extract(np.zeros((100, 100), dtype=np.uint8), sensor_id="front_camera")
    assert result is None


# ---------- Integration: perception stack with learned perception ----------

def test_perception_stack_creates_learned_extractors_when_enabled() -> None:
    from autonomy_demo.perception.module import PerceptionStack

    stack = PerceptionStack(device="cpu", model_variant="bootstrap", enable_learned_perception=True)
    assert stack.learned_drivable_extractor is not None
    assert stack.learned_lane_extractor is not None


def test_perception_stack_skips_learned_when_disabled() -> None:
    from autonomy_demo.perception.module import PerceptionStack

    stack = PerceptionStack(device="cpu", model_variant="bootstrap", enable_learned_perception=False)
    assert stack.learned_drivable_extractor is None
    assert stack.learned_lane_extractor is None
