"""Cityscapes 19-class palette and label names for SegFormer visualization."""

from __future__ import annotations

import numpy as np

# Cityscapes training-ID order (0-18), matching SegFormer-B0 output indices.
CITYSCAPES_LABELS: list[str] = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# (19, 3) uint8 RGB palette — index with palette[label_map] for fast colorization.
CITYSCAPES_PALETTE: np.ndarray = np.array(
    [
        [128, 64, 128],   # 0  road
        [244, 35, 232],   # 1  sidewalk
        [70, 70, 70],     # 2  building
        [102, 102, 156],  # 3  wall
        [190, 153, 153],  # 4  fence
        [153, 153, 153],  # 5  pole
        [250, 170, 30],   # 6  traffic light
        [220, 220, 0],    # 7  traffic sign
        [107, 142, 35],   # 8  vegetation
        [152, 251, 152],  # 9  terrain
        [70, 130, 180],   # 10 sky
        [220, 20, 60],    # 11 person
        [255, 0, 0],      # 12 rider
        [0, 0, 142],      # 13 car
        [0, 0, 70],       # 14 truck
        [0, 60, 100],     # 15 bus
        [0, 80, 100],     # 16 train
        [0, 0, 230],      # 17 motorcycle
        [119, 11, 32],    # 18 bicycle
    ],
    dtype=np.uint8,
)

NUM_CLASSES: int = len(CITYSCAPES_LABELS)
