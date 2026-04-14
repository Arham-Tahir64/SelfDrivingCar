from __future__ import annotations

import numpy as np

from autonomy_demo.perception.internal_types import CameraSegmentationResult
from autonomy_demo.perception.segmentation_metrics import summarize_segmentation_metrics
from autonomy_demo.perception.segmentation_tasks import NUM_TASK_CLASSES, TASK_DRIVABLE


def test_summarize_segmentation_metrics_resizes_target_labels() -> None:
    height, width = 540, 960
    small_height, small_width = 360, 640

    task_probabilities = np.zeros((height, width, NUM_TASK_CLASSES), dtype=np.float32)
    task_probabilities[..., TASK_DRIVABLE] = 1.0
    segmentation = CameraSegmentationResult(
        semantic_label_map=np.zeros((height, width), dtype=np.uint8),
        task_label_map=np.full((height, width), TASK_DRIVABLE, dtype=np.uint8),
        task_probabilities=task_probabilities,
        drivable_prob=np.ones((height, width), dtype=np.float32),
        lane_boundary_prob=np.zeros((height, width), dtype=np.float32),
        curb_boundary_prob=np.zeros((height, width), dtype=np.float32),
        uncertainty=np.zeros((height, width), dtype=np.float32),
        source_sensor_id="front_camera",
        model_name="test-model",
        model_version="test-version",
    )

    target_task_labels = np.full((small_height, small_width), TASK_DRIVABLE, dtype=np.uint8)

    metrics = summarize_segmentation_metrics(segmentation, target_task_labels)

    assert metrics["segmentation_mean_iou"] == 1.0
    assert metrics["drivable_iou"] == 1.0
