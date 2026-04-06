from __future__ import annotations

import numpy as np

from autonomy_demo.perception.segmentation_tasks import (
    TASK_DRIVABLE,
    TASK_LANE_MARKING,
    TASK_SIDEWALK_NON_DRIVABLE,
    TASK_VEHICLE,
    cityscapes_probs_to_task_probs,
    derive_boundary_targets,
    remap_carla_semantic_to_task,
    remap_cityscapes_to_task,
    semantic_camera_rgb_to_label_map,
)


def test_remap_cityscapes_to_task_groups_road_sidewalk_and_vehicles() -> None:
    cityscapes = np.array(
        [
            [0, 1, 13, 11],
            [14, 18, 8, 2],
        ],
        dtype=np.uint8,
    )

    remapped = remap_cityscapes_to_task(cityscapes)

    assert remapped[0, 0] == TASK_DRIVABLE
    assert remapped[0, 1] == TASK_SIDEWALK_NON_DRIVABLE
    assert remapped[0, 2] == TASK_VEHICLE
    assert remapped[0, 3] != TASK_DRIVABLE


def test_remap_carla_semantic_to_task_keeps_roadline_and_road() -> None:
    carla = np.array(
        [
            [6, 7, 8],
            [10, 4, 0],
        ],
        dtype=np.uint8,
    )

    remapped = remap_carla_semantic_to_task(carla)

    assert remapped[0, 0] == TASK_LANE_MARKING
    assert remapped[0, 1] == TASK_DRIVABLE
    assert remapped[0, 2] == TASK_SIDEWALK_NON_DRIVABLE
    assert remapped[1, 0] == TASK_VEHICLE


def test_semantic_camera_rgb_to_label_map_uses_first_channel() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.float32)
    frame[..., 0] = np.array([[7, 6, 8], [10, 4, 0]], dtype=np.float32)

    labels = semantic_camera_rgb_to_label_map(frame)

    assert labels.dtype == np.uint8
    assert labels[0, 0] == 7
    assert labels[1, 1] == 4


def test_cityscapes_probs_to_task_probs_normalizes_probabilities() -> None:
    probs = np.zeros((2, 2, 19), dtype=np.float32)
    probs[..., 0] = 0.7
    probs[..., 1] = 0.2
    probs[..., 13] = 0.1

    task_probs = cityscapes_probs_to_task_probs(probs)

    assert task_probs.shape == (2, 2, 7)
    assert np.allclose(np.sum(task_probs, axis=-1), 1.0)
    assert np.all(task_probs[..., TASK_DRIVABLE] >= 0.69)


def test_derive_boundary_targets_extracts_lane_and_curb_edges() -> None:
    task_labels = np.full((10, 12), TASK_SIDEWALK_NON_DRIVABLE, dtype=np.uint8)
    task_labels[:, 3:9] = TASK_DRIVABLE
    task_labels[5:7, 5:7] = TASK_LANE_MARKING

    lane_boundary, curb_boundary = derive_boundary_targets(task_labels)

    assert lane_boundary.shape == task_labels.shape
    assert curb_boundary.shape == task_labels.shape
    assert float(np.max(lane_boundary[5:7, 5:7])) > 0.5
    assert float(np.max(curb_boundary[:, 3])) > 0.5
