from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def distance_xy(point_a: ArrayLike, point_b: ArrayLike) -> float:
    a = np.asarray(point_a, dtype=np.float32)
    b = np.asarray(point_b, dtype=np.float32)
    return float(np.linalg.norm(a[:2] - b[:2]))


def distance_xyz(point_a: ArrayLike, point_b: ArrayLike) -> float:
    a = np.asarray(point_a, dtype=np.float32)
    b = np.asarray(point_b, dtype=np.float32)
    return float(np.linalg.norm(a[:3] - b[:3]))


def signed_lateral_error(
    origin_x: float,
    origin_y: float,
    origin_yaw_rad: float,
    target_x: float,
    target_y: float,
) -> float:
    dx = target_x - origin_x
    dy = target_y - origin_y
    return (-math.sin(origin_yaw_rad) * dx) + (math.cos(origin_yaw_rad) * dy)
