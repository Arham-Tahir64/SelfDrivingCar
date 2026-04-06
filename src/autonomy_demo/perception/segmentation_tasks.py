from __future__ import annotations

import numpy as np

TASK_BACKGROUND = 0
TASK_DRIVABLE = 1
TASK_LANE_MARKING = 2
TASK_CURB_BOUNDARY = 3
TASK_SIDEWALK_NON_DRIVABLE = 4
TASK_VEHICLE = 5
TASK_VRU = 6

NUM_TASK_CLASSES = 7

TASK_CLASS_NAMES: list[str] = [
    "background",
    "drivable",
    "lane_marking",
    "curb_boundary",
    "sidewalk_non_drivable",
    "vehicle",
    "vulnerable_road_user",
]

TASK_CLASS_PALETTE: np.ndarray = np.array(
    [
        [16, 20, 28],
        [42, 210, 236],
        [255, 230, 90],
        [255, 123, 79],
        [136, 103, 202],
        [62, 161, 255],
        [255, 88, 124],
    ],
    dtype=np.uint8,
)

CITYSCAPES_TO_TASK = np.array(
    [
        TASK_DRIVABLE,
        TASK_SIDEWALK_NON_DRIVABLE,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_BACKGROUND,
        TASK_VRU,
        TASK_VRU,
        TASK_VEHICLE,
        TASK_VEHICLE,
        TASK_VEHICLE,
        TASK_VEHICLE,
        TASK_VRU,
        TASK_VRU,
    ],
    dtype=np.uint8,
)

CARLA_TO_TASK = np.zeros(256, dtype=np.uint8)
CARLA_TO_TASK[4] = TASK_VRU  # pedestrian
CARLA_TO_TASK[6] = TASK_LANE_MARKING  # road line
CARLA_TO_TASK[7] = TASK_DRIVABLE  # road
CARLA_TO_TASK[8] = TASK_SIDEWALK_NON_DRIVABLE  # sidewalk
CARLA_TO_TASK[10] = TASK_VEHICLE  # vehicle


def remap_cityscapes_to_task(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int64)
    labels = np.clip(labels, 0, len(CITYSCAPES_TO_TASK) - 1)
    return CITYSCAPES_TO_TASK[labels].astype(np.uint8)


def remap_carla_semantic_to_task(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int64)
    labels = np.clip(labels, 0, len(CARLA_TO_TASK) - 1)
    return CARLA_TO_TASK[labels].astype(np.uint8)


def semantic_camera_rgb_to_label_map(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim != 3 or image.shape[2] < 1:
        raise ValueError("semantic camera frame must be HxW or HxWxC")
    # CARLA semantic sensor raw output stores the object tag in the first RGB channel
    # once the BGRA buffer is converted to RGB in the sensor suite.
    return np.rint(image[..., 0]).astype(np.uint8)


def cityscapes_probs_to_task_probs(class_probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(class_probabilities, dtype=np.float32)
    if probs.ndim != 3 or probs.shape[2] != len(CITYSCAPES_TO_TASK):
        raise ValueError("expected Cityscapes probabilities with shape HxWx19")
    task_probs = np.zeros(probs.shape[:2] + (NUM_TASK_CLASSES,), dtype=np.float32)
    for class_index, task_index in enumerate(CITYSCAPES_TO_TASK.tolist()):
        task_probs[..., task_index] += probs[..., class_index]
    normalization = np.maximum(np.sum(task_probs, axis=-1, keepdims=True), 1e-6)
    return (task_probs / normalization).astype(np.float32)


def task_label_map_to_palette(task_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(task_labels, dtype=np.int64)
    labels = np.clip(labels, 0, len(TASK_CLASS_PALETTE) - 1)
    return TASK_CLASS_PALETTE[labels]


def derive_boundary_targets(
    task_labels: np.ndarray,
    *,
    lane_dilate_pixels: int = 1,
    curb_dilate_pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(task_labels, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError("task_labels must be HxW")
    lane_marking = labels == TASK_LANE_MARKING
    drivable = (labels == TASK_DRIVABLE) | lane_marking
    non_drivable = labels == TASK_SIDEWALK_NON_DRIVABLE
    curb_boundary = _boundary_between_masks(drivable, non_drivable)
    lane_boundary = lane_marking.copy()
    if lane_dilate_pixels > 0:
        lane_boundary = _dilate_mask(lane_boundary, lane_dilate_pixels)
    if curb_dilate_pixels > 0:
        curb_boundary = _dilate_mask(curb_boundary, curb_dilate_pixels)
    return lane_boundary.astype(np.float32), curb_boundary.astype(np.float32)


def _boundary_between_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = np.asarray(mask_a, dtype=np.bool_)
    mask_b = np.asarray(mask_b, dtype=np.bool_)
    boundary = np.zeros_like(mask_a, dtype=np.bool_)
    for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_b = np.roll(mask_b, shift=(delta_y, delta_x), axis=(0, 1))
        if delta_y > 0:
            shifted_b[:delta_y, :] = False
        elif delta_y < 0:
            shifted_b[delta_y:, :] = False
        if delta_x > 0:
            shifted_b[:, :delta_x] = False
        elif delta_x < 0:
            shifted_b[:, delta_x:] = False
        boundary |= mask_a & shifted_b
    return boundary


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.bool_)
    if radius <= 0 or not binary.any():
        return binary
    try:
        import cv2  # type: ignore

        kernel_size = (radius * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(binary.astype(np.uint8), kernel, iterations=1) > 0
    except Exception:
        dilated = binary.copy()
        for delta_y in range(-radius, radius + 1):
            for delta_x in range(-radius, radius + 1):
                shifted = np.roll(binary, shift=(delta_y, delta_x), axis=(0, 1))
                if delta_y > 0:
                    shifted[:delta_y, :] = False
                elif delta_y < 0:
                    shifted[delta_y:, :] = False
                if delta_x > 0:
                    shifted[:, :delta_x] = False
                elif delta_x < 0:
                    shifted[:, delta_x:] = False
                dilated |= shifted
        return dilated
