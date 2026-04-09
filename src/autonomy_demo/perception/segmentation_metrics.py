from __future__ import annotations

import numpy as np

from autonomy_demo.perception.internal_types import CameraSegmentationResult
from autonomy_demo.perception.segmentation_tasks import (
    NUM_TASK_CLASSES,
    TASK_CLASS_NAMES,
    TASK_DRIVABLE,
    TASK_LANE_MARKING,
    derive_boundary_targets,
)


def binary_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=np.bool_)
    target = np.asarray(target_mask, dtype=np.bool_)
    union = np.count_nonzero(pred | target)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(pred & target)
    return float(intersection / union)


def boundary_f1(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=np.bool_)
    target = np.asarray(target_mask, dtype=np.bool_)
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 1e-6:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def mean_iou(pred_labels: np.ndarray, target_labels: np.ndarray) -> float:
    pred = np.asarray(pred_labels, dtype=np.uint8)
    target = np.asarray(target_labels, dtype=np.uint8)
    class_ious: list[float] = []
    for class_id in range(NUM_TASK_CLASSES):
        pred_mask = pred == class_id
        target_mask = target == class_id
        if not pred_mask.any() and not target_mask.any():
            continue
        class_ious.append(binary_iou(pred_mask, target_mask))
    if not class_ious:
        return 1.0
    return float(np.mean(class_ious))


def expected_calibration_error(
    confidence: np.ndarray,
    correctness: np.ndarray,
    *,
    num_bins: int = 10,
) -> float:
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float32).reshape(-1)
    if confidence.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)
    total = float(confidence.size)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper >= 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidence[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        error += abs(bin_acc - bin_conf) * (float(np.count_nonzero(mask)) / total)
    return float(error)


def summarize_segmentation_metrics(
    segmentation: CameraSegmentationResult,
    target_task_labels: np.ndarray,
) -> dict[str, float]:
    target_labels = np.asarray(target_task_labels, dtype=np.uint8)
    pred_labels = np.asarray(segmentation.task_label_map, dtype=np.uint8)
    pred_drivable = np.asarray(segmentation.drivable_prob >= 0.5, dtype=np.bool_)
    target_drivable = (target_labels == TASK_DRIVABLE) | (target_labels == TASK_LANE_MARKING)
    target_lane_boundary, target_curb_boundary = derive_boundary_targets(target_labels)
    confidence = np.max(segmentation.task_probabilities, axis=-1)
    correctness = pred_labels == target_labels
    metrics = {
        "segmentation_mean_iou": mean_iou(pred_labels, target_labels),
        "drivable_iou": binary_iou(pred_drivable, target_drivable),
        "lane_boundary_f1": boundary_f1(segmentation.lane_boundary_prob >= 0.5, target_lane_boundary >= 0.5),
        "curb_boundary_f1": boundary_f1(segmentation.curb_boundary_prob >= 0.5, target_curb_boundary >= 0.5),
        "task_expected_calibration_error": expected_calibration_error(confidence, correctness),
        "segmentation_uncertainty_mean": float(np.mean(segmentation.uncertainty)),
    }
    for class_id, class_name in enumerate(TASK_CLASS_NAMES):
        class_metrics = binary_iou(pred_labels == class_id, target_labels == class_id)
        metrics[f"class_iou_{class_name}"] = class_metrics
    return metrics
