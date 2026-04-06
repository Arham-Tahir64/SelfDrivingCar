from __future__ import annotations

import numpy as np
import pytest
import torch

from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import DrivableSpaceMask, LaneLine
from autonomy_demo.perception.internal_types import CameraSegmentationResult


# ---------- SegFormer drivable space tests ----------


class _FakeSegformerProcessor:
    def __init__(self) -> None:
        self.last_image_shape: tuple[int, int] | None = None

    def __call__(self, images, return_tensors: str):  # noqa: ANN001
        self.last_image_shape = tuple(images.shape[:2])
        return {"pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32)}


class _FakeSegformerOutputs:
    def __init__(self) -> None:
        logits = torch.zeros((1, 19, 2, 2), dtype=torch.float32)
        logits[:, 0, :, :] = 10.0
        self.logits = logits


class _FakeSegformerModel:
    def __call__(self, **inputs):  # noqa: ANN003
        return _FakeSegformerOutputs()


class _FakeLaneModel:
    def __call__(self, tensor):  # noqa: ANN001
        return torch.zeros((1, 1), dtype=torch.float32)

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
    assert extractor.last_segmentation_result is not None
    assert extractor.last_segmentation_result.drivable_prob.shape == (256, 512)
    assert extractor.last_segmentation_result.uncertainty.shape == (256, 512)


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


def test_segformer_extractor_returns_cached_result_on_skipped_ticks_without_reporting_inference() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu", run_every_n_ticks=2)
    fake_processor = _FakeSegformerProcessor()
    extractor._processor = fake_processor
    extractor._model = _FakeSegformerModel()
    extractor._ensure_loaded = lambda: True  # type: ignore[method-assign]
    image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)

    first = extractor.extract(image, "front_camera")
    second = extractor.extract(image, "front_camera")

    assert first is not None
    assert second is first
    assert extractor.ran_inference_last_call is False
    assert second.mask.shape == (128, 256)


def test_segformer_extractor_downscales_input_but_preserves_output_shape() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu", max_input_long_edge_px=128)
    fake_processor = _FakeSegformerProcessor()
    extractor._processor = fake_processor
    extractor._model = _FakeSegformerModel()
    extractor._ensure_loaded = lambda: True  # type: ignore[method-assign]
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)

    result = extractor.extract(image, "front_camera")

    assert result is not None
    assert fake_processor.last_image_shape is not None
    assert max(fake_processor.last_image_shape) <= 128
    assert result.mask.shape == (256, 512)
    assert result.class_probabilities.shape == (256, 512, 2)


def test_segformer_extractor_emits_structured_segmentation_result() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu")
    fake_processor = _FakeSegformerProcessor()
    extractor._processor = fake_processor
    extractor._model = _FakeSegformerModel()
    extractor._ensure_loaded = lambda: True  # type: ignore[method-assign]
    image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)

    result = extractor.extract(image, "front_camera")

    assert result is not None
    segmentation = extractor.last_segmentation_result
    assert segmentation is not None
    assert segmentation.semantic_label_map.shape == (128, 256)
    assert segmentation.task_label_map.shape == (128, 256)
    assert segmentation.task_probabilities.shape == (128, 256, 7)
    assert segmentation.model_name.endswith("cityscapes-1024-1024")


def test_segformer_extractor_reprojects_cached_segmentation_on_skipped_ticks() -> None:
    from autonomy_demo.perception.segformer_drivable import SegFormerDrivableExtractor

    extractor = SegFormerDrivableExtractor(device="cpu", run_every_n_ticks=2)
    task_probabilities = np.zeros((80, 120, 7), dtype=np.float32)
    task_probabilities[..., 0] = 0.9
    task_probabilities[42:78, 38:84, 0] = 0.05
    task_probabilities[42:78, 38:84, 1] = 0.95
    initial = CameraSegmentationResult(
        semantic_label_map=np.zeros((80, 120), dtype=np.uint8),
        task_label_map=np.argmax(task_probabilities, axis=-1).astype(np.uint8),
        task_probabilities=task_probabilities,
        drivable_prob=task_probabilities[..., 1].copy(),
        lane_boundary_prob=np.zeros((80, 120), dtype=np.float32),
        curb_boundary_prob=np.zeros((80, 120), dtype=np.float32),
        uncertainty=np.zeros((80, 120), dtype=np.float32),
        source_sensor_id="front_camera",
        model_name="test-model",
        model_version="test-model",
        source_frame_id=1,
        source_tick_id=1,
        source_sim_time_s=0.0,
        source_ego_world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        source_ego_yaw_rad=0.0,
        camera_calibration={
            "fov_deg": 90.0,
            "mount_xyz": [2.3, 0.0, 0.8],
            "mount_rpy_deg": [0.0, 0.0, 0.0],
            "image_width": 120,
            "image_height": 80,
        },
    )
    first = extractor._store_display_result(initial)
    extractor._latest_inference_segmentation_result = initial
    extractor._tick_counter = 1

    second = extractor.extract(
        np.zeros((80, 120, 3), dtype=np.uint8),
        "front_camera",
        frame_id=2,
        tick_id=2,
        sim_time_s=0.1,
        ego_world_xyz=np.array([0.6, 0.0, 0.0], dtype=np.float32),
        ego_yaw_rad=0.12,
        camera_calibration={
            "fov_deg": 90.0,
            "mount_xyz": [2.3, 0.0, 0.8],
            "mount_rpy_deg": [0.0, 0.0, 0.0],
            "image_width": 120,
            "image_height": 80,
        },
    )

    assert second is not None
    assert extractor.ran_inference_last_call is False
    assert extractor.last_segmentation_result is not None
    assert extractor.last_segmentation_result.reprojected is True
    assert extractor.last_segmentation_result.warped_from_tick_id == 1
    assert extractor.last_segmentation_result.source_tick_id == 2
    assert not np.array_equal(first.mask, second.mask)


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


def test_learned_lane_extractor_live_mode_disables_online_training() -> None:
    from autonomy_demo.perception.learned_lane_detection import LearnedLaneExtractor

    extractor = LearnedLaneExtractor(device="cpu", allow_online_training=False)
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
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

    assert result is None
    assert len(extractor._training_buffer) == 0
    assert extractor.is_trained is False


def test_learned_lane_extractor_respects_inference_cadence_when_trained() -> None:
    from autonomy_demo.perception.learned_lane_detection import LearnedLaneExtractor

    extractor = LearnedLaneExtractor(device="cpu", run_every_n_ticks=2, allow_online_training=False)
    extractor._trained = True
    extractor._model = _FakeLaneModel()
    extractor._image_to_tensor = lambda image: torch.zeros((1, 3, 160, 288), dtype=torch.float32)  # type: ignore[method-assign]
    extractor._decode_predictions = lambda logits, image_width, image_height: [  # type: ignore[method-assign]
        np.array([[100.0, 240.0], [110.0, 200.0], [120.0, 160.0], [130.0, 120.0]], dtype=np.float32),
        np.array([[300.0, 240.0], [310.0, 200.0], [320.0, 160.0], [330.0, 120.0]], dtype=np.float32),
    ]
    image = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)

    first = extractor.extract(image, sensor_id="front_camera")
    assert first is not None
    assert extractor.ran_inference_last_call is True
    second = extractor.extract(image, sensor_id="front_camera")

    assert extractor.ran_inference_last_call is False
    assert second is None


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


def test_build_perception_module_disables_aux_perception_for_live_cpu_by_default() -> None:
    from autonomy_demo.perception.module import build_perception_module

    runtime = type(
        "Runtime",
        (),
        {
            "backend": "carla",
            "perception_mode": "camera_v1",
            "perception_device": "cpu",
            "perception_model_variant": "bootstrap",
            "enable_learned_perception": True,
            "tuning": {},
        },
    )()
    module = build_perception_module(runtime)
    assert module.learned_drivable_extractor is None
    assert module.learned_lane_extractor is None


def test_build_perception_module_throttles_segformer_for_live_gpu_by_default() -> None:
    from autonomy_demo.perception.module import build_perception_module

    runtime = type(
        "Runtime",
        (),
        {
            "backend": "carla",
            "perception_mode": "camera_v1",
            "perception_device": "cuda",
            "perception_model_variant": "bootstrap",
            "enable_learned_perception": True,
            "tuning": {},
        },
    )()
    module = build_perception_module(runtime)
    assert module.learned_drivable_extractor is not None
    assert module.learned_drivable_extractor._run_every_n_ticks == 10
    assert module.learned_drivable_extractor._max_input_long_edge_px == 512
    assert module.learned_lane_extractor is None
