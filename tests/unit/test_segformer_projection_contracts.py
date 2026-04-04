from __future__ import annotations

import numpy as np

from autonomy_demo.perception.segformer_drivable import ROAD_CLASS_ID, SegFormerDrivableExtractor
from autonomy_demo.visualization.service import _has_image_bbox, _is_camera_grounded


def test_overlay_bbox_guard_accepts_only_camera_projected_boxes() -> None:
    bbox = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    assert _has_image_bbox(bbox)
    assert not _has_image_bbox(None)
    assert _is_camera_grounded("camera_projection")
    assert not _is_camera_grounded("truth_fallback")


def test_segformer_road_mask_keeps_bottom_center_road_and_excludes_sidewalk() -> None:
    extractor = SegFormerDrivableExtractor(device="cpu")
    pred_classes = np.full((80, 120), fill_value=1, dtype=np.int64)
    road_prob = np.zeros((80, 120), dtype=np.float32)

    pred_classes[45:, 40:80] = ROAD_CLASS_ID
    road_prob[45:, 40:80] = 0.92

    pred_classes[35:60, 92:116] = 1
    road_prob[35:60, 92:116] = 0.98

    mask = extractor._road_mask(pred_classes, road_prob)

    assert mask[70, 60]
    assert not mask[50, 100]


def test_segformer_road_mask_prefers_ego_anchored_component_over_isolated_blob() -> None:
    extractor = SegFormerDrivableExtractor(device="cpu")
    pred_classes = np.full((100, 160), fill_value=1, dtype=np.int64)
    road_prob = np.zeros((100, 160), dtype=np.float32)

    pred_classes[58:, 52:108] = ROAD_CLASS_ID
    road_prob[58:, 52:108] = 0.88

    pred_classes[10:28, 5:30] = ROAD_CLASS_ID
    road_prob[10:28, 5:30] = 0.95

    mask = extractor._road_mask(pred_classes, road_prob)

    assert mask[85, 80]
    assert not mask[18, 12]
