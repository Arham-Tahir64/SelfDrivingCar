from __future__ import annotations

from collections import defaultdict, deque
from math import cos, sin

import numpy as np

from autonomy_demo.interfaces.enums import ObjectClass
from autonomy_demo.interfaces.types import ConeDetection, SensorFrameBundle
from autonomy_demo.perception.internal_types import LidarClusterDetection


def _bbox_corners(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
        ],
        dtype=np.float32,
    )


class LidarObstacleDetector:
    """Deterministic LiDAR obstacle extraction via simple XY occupancy clustering."""

    def __init__(
        self,
        *,
        cell_size_m: float = 0.75,
        min_cluster_points: int = 3,
        min_range_m: float = 1.5,
        max_range_m: float = 60.0,
        min_height_m: float = 0.35,
        ground_z_threshold_m: float = -1.5,
    ) -> None:
        self.cell_size_m = cell_size_m
        self.min_cluster_points = min_cluster_points
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.min_height_m = min_height_m
        self.ground_z_threshold_m = ground_z_threshold_m

    def detect(
        self,
        bundle: SensorFrameBundle,
    ) -> tuple[list[LidarClusterDetection], list[ConeDetection]]:
        sensor_points = self._filtered_sensor_points(bundle.lidar.points_xyz)
        if len(sensor_points) == 0:
            return [], []

        world_points = self._sensor_to_world(sensor_points, bundle)
        detections: list[LidarClusterDetection] = []
        cones: list[ConeDetection] = []
        for indices in self._cluster_indices(sensor_points[:, :2]):
            if len(indices) < self.min_cluster_points:
                continue
            cluster_world = world_points[indices]
            min_xyz = np.min(cluster_world, axis=0)
            max_xyz = np.max(cluster_world, axis=0)
            size_xyz = max_xyz - min_xyz
            if float(size_xyz[2]) < self.min_height_m:
                continue
            centroid_xyz = np.mean(cluster_world, axis=0).astype(np.float32)
            cone = self._maybe_cone(centroid_xyz, size_xyz, len(indices))
            if cone is not None:
                cones.append(cone)
                continue
            detections.append(
                LidarClusterDetection(
                    centroid_xyz=centroid_xyz,
                    world_bbox_3d=_bbox_corners(min_xyz, max_xyz),
                    object_class=self._classify_cluster(size_xyz),
                    confidence=self._cluster_confidence(len(indices), size_xyz),
                    point_count=len(indices),
                )
            )
        return detections, cones

    def _filtered_sensor_points(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        ranges = np.linalg.norm(points[:, :2], axis=1)
        mask = (
            (ranges >= self.min_range_m)
            & (ranges <= self.max_range_m)
            & (points[:, 2] >= self.ground_z_threshold_m)
            & (points[:, 2] <= 3.5)
        )
        return points[mask]

    def _sensor_to_world(self, sensor_points: np.ndarray, bundle: SensorFrameBundle) -> np.ndarray:
        yaw_rad = float(bundle.metadata.get("ego_yaw_rad", 0.0))
        ego_xyz = np.asarray(bundle.gnss.world_xyz, dtype=np.float32)
        rotation = np.array(
            [
                [cos(yaw_rad), -sin(yaw_rad), 0.0],
                [sin(yaw_rad), cos(yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return (sensor_points @ rotation.T) + ego_xyz

    def _cluster_indices(self, points_xy: np.ndarray) -> list[np.ndarray]:
        if len(points_xy) == 0:
            return []
        cell_points: dict[tuple[int, int], list[int]] = defaultdict(list)
        cell_coords = np.floor(points_xy / self.cell_size_m).astype(np.int32)
        for index, coord in enumerate(cell_coords):
            cell_points[(int(coord[0]), int(coord[1]))].append(index)

        visited: set[tuple[int, int]] = set()
        clusters: list[np.ndarray] = []
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        for start in cell_points:
            if start in visited:
                continue
            queue = deque([start])
            visited.add(start)
            cluster_indices: list[int] = []
            while queue:
                cell = queue.popleft()
                cluster_indices.extend(cell_points[cell])
                for dx, dy in neighbors:
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in cell_points and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(np.asarray(cluster_indices, dtype=np.int32))
        return clusters

    def _classify_cluster(self, size_xyz: np.ndarray) -> ObjectClass:
        length_m = float(size_xyz[0])
        width_m = float(size_xyz[1])
        height_m = float(size_xyz[2])
        footprint_m = max(length_m, width_m)
        if footprint_m < 1.2 and height_m > 1.2:
            return ObjectClass.PEDESTRIAN
        if footprint_m < 1.5:
            return ObjectClass.CYCLIST
        return ObjectClass.VEHICLE

    def _maybe_cone(
        self,
        centroid_xyz: np.ndarray,
        size_xyz: np.ndarray,
        point_count: int,
    ) -> ConeDetection | None:
        footprint_m = max(float(size_xyz[0]), float(size_xyz[1]))
        height_m = float(size_xyz[2])
        if footprint_m <= 0.9 and 0.2 <= height_m <= 1.2 and point_count <= 12:
            return ConeDetection(world_xyz=centroid_xyz.astype(np.float32), confidence=0.8)
        return None

    def _cluster_confidence(self, point_count: int, size_xyz: np.ndarray) -> float:
        confidence = 0.35 + (min(point_count, 20) / 30.0) + min(float(size_xyz[2]), 2.0) * 0.1
        return float(np.clip(confidence, 0.3, 0.99))
