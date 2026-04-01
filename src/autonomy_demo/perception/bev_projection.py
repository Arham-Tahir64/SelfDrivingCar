from __future__ import annotations

import math
from typing import Any

import numpy as np

from autonomy_demo.interfaces.types import DrivableSpaceMask, EgoPose, LocalMap, RoutePlan, StaticLaneSegment

# BEV grid parameters
GRID_SIZE = 100
CELL_SIZE_M = 0.5
FORWARD_RANGE_M = GRID_SIZE * CELL_SIZE_M
LATERAL_RANGE_M = (GRID_SIZE // 2) * CELL_SIZE_M
_HISTORY_CELL_SIZE_M = 0.5
_CROP_X_MIN_M = -20.0
_CROP_X_MAX_M = 40.0
_CROP_Y_MIN_M = -15.0
_CROP_Y_MAX_M = 15.0
_HISTORY_DECAY_S = 4.0
_HISTORY_PRUNE_DISTANCE_M = 90.0
_MAX_HISTORY_SAMPLES = 2200

_DEFAULT_FRONT_CAMERA_CALIBRATION = {
    "fov_deg": 90.0,
    "mount_xyz": [2.3, 0.0, 0.8],
    "mount_rpy_deg": [0.0, 0.0, 0.0],  # roll, pitch, yaw
}
_ROAD_PROB_THRESHOLD = 0.35
_GRID_CONFIDENCE_THRESHOLD = 40.0
_GRID_KEEP_THRESHOLD = 28.0
_GRID_CLOSE_ITERATIONS = 3


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class BEVDrivableProjector:
    """Projects a front-camera road mask into an ego-relative BEV grid."""

    def __init__(self) -> None:
        self._ray_directions_ego: np.ndarray | None = None
        self._camera_origin_ego = np.zeros(3, dtype=np.float32)
        self._cache_key: tuple[Any, ...] | None = None
        self._history: dict[tuple[int, int], tuple[float, float]] = {}
        self._corridor_cache_key: tuple[str, ...] | None = None
        self._corridor_cache_payload: dict[str, Any] | None = None

    def _normalise_calibration(
        self,
        calibration: dict[str, Any] | None,
        *,
        image_height: int,
        image_width: int,
    ) -> dict[str, Any]:
        merged = dict(_DEFAULT_FRONT_CAMERA_CALIBRATION)
        if calibration:
            merged.update(calibration)
        merged["image_width"] = int(image_width)
        merged["image_height"] = int(image_height)
        merged["fov_deg"] = float(merged.get("fov_deg", 90.0))
        merged["mount_xyz"] = list(merged.get("mount_xyz", [2.3, 0.0, 0.8]))
        merged["mount_rpy_deg"] = list(merged.get("mount_rpy_deg", [0.0, 0.0, 0.0]))
        return merged

    def _build_lookup(self, calibration: dict[str, Any]) -> None:
        image_height = int(calibration["image_height"])
        image_width = int(calibration["image_width"])
        hfov_rad = np.deg2rad(float(calibration["fov_deg"]))
        vfov_rad = 2.0 * np.arctan(np.tan(hfov_rad * 0.5) * (image_height / max(image_width, 1)))

        fx = image_width / max(2.0 * np.tan(hfov_rad * 0.5), 1e-6)
        fy = image_height / max(2.0 * np.tan(vfov_rad * 0.5), 1e-6)
        cx = image_width * 0.5
        cy = image_height * 0.5

        us = np.arange(image_width, dtype=np.float32)
        vs = np.arange(image_height, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        # Camera ray in ego-style coordinates: x forward, y right, z up.
        rays_camera = np.stack(
            [
                np.ones_like(uu, dtype=np.float32),
                (uu - cx) / max(fx, 1e-6),
                -(vv - cy) / max(fy, 1e-6),
            ],
            axis=-1,
        )

        roll_deg, pitch_deg, yaw_deg = calibration["mount_rpy_deg"]
        rotation = (
            _rotation_z(np.deg2rad(float(yaw_deg)))
            @ _rotation_y(np.deg2rad(float(pitch_deg)))
            @ _rotation_x(np.deg2rad(float(roll_deg)))
        ).astype(np.float32)

        self._ray_directions_ego = rays_camera.reshape(-1, 3) @ rotation.T
        self._camera_origin_ego = np.asarray(calibration["mount_xyz"], dtype=np.float32)
        self._cache_key = (
            image_height,
            image_width,
            float(calibration["fov_deg"]),
            *self._camera_origin_ego.tolist(),
            *[float(v) for v in calibration["mount_rpy_deg"]],
        )

    def _connected_component_from_ego_anchor(self, candidate: np.ndarray) -> np.ndarray:
        if not candidate.any():
            return candidate.astype(np.bool_)
        try:
            import cv2  # type: ignore

            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                candidate.astype(np.uint8),
                connectivity=8,
            )
            if num_labels <= 1:
                return candidate.astype(np.bool_)
            height, width = candidate.shape
            seed = np.zeros_like(candidate, dtype=np.bool_)
            seed[int(height * 0.72) :, int(width * 0.35) : int(width * 0.65)] = True
            bottom_seed = np.zeros_like(candidate, dtype=np.bool_)
            bottom_seed[int(height * 0.86) :, :] = True

            best_label = 1
            best_score = -1
            for label in range(1, num_labels):
                component = labels == label
                area = int(stats[label, 4])
                if np.any(component & seed):
                    score = area + 10_000
                elif np.any(component & bottom_seed):
                    score = area + 1_000
                else:
                    score = area
                if score > best_score:
                    best_label = label
                    best_score = score
            return (labels == best_label).astype(np.bool_)
        except Exception:
            return candidate.astype(np.bool_)

    def _projection_mask(self, drivable: DrivableSpaceMask) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(drivable.mask, dtype=np.bool_)
        if drivable.class_probabilities.ndim == 3 and drivable.class_probabilities.shape[2] >= 2:
            values = np.asarray(drivable.class_probabilities[..., 1], dtype=np.float32)
        else:
            values = mask.astype(np.float32)
        candidate = mask & (values >= _ROAD_PROB_THRESHOLD)
        if not candidate.any():
            candidate = mask
        candidate = self._connected_component_from_ego_anchor(candidate)
        return candidate, values

    def _binary_dilate(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.bool_), 1, mode="constant", constant_values=False)
        result = np.zeros_like(mask, dtype=np.bool_)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return result

    def _binary_erode(self, mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask.astype(np.bool_), 1, mode="constant", constant_values=True)
        result = np.ones_like(mask, dtype=np.bool_)
        for dy in range(3):
            for dx in range(3):
                result &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return result

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.bool_)
        if not mask.any():
            return mask
        inverse = ~mask
        visited = np.zeros_like(mask, dtype=np.bool_)
        stack: list[tuple[int, int]] = []
        rows, cols = mask.shape

        def _push(r: int, c: int) -> None:
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return
            if visited[r, c] or not inverse[r, c]:
                return
            visited[r, c] = True
            stack.append((r, c))

        for r in range(rows):
            _push(r, 0)
            _push(r, cols - 1)
        for c in range(cols):
            _push(0, c)
            _push(rows - 1, c)

        while stack:
            r, c = stack.pop()
            _push(r - 1, c)
            _push(r + 1, c)
            _push(r, c - 1)
            _push(r, c + 1)

        holes = inverse & ~visited
        return mask | holes

    def _keep_ego_connected_component(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return mask.astype(np.bool_)
        try:
            import cv2  # type: ignore

            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8),
                connectivity=8,
            )
            if num_labels <= 1:
                return mask.astype(np.bool_)
            rows, cols = mask.shape
            near_ego_seed = np.zeros_like(mask, dtype=np.bool_)
            near_ego_seed[max(rows - 8, 0) :, max((cols // 2) - 8, 0) : min((cols // 2) + 8, cols)] = True
            fallback_seed = np.zeros_like(mask, dtype=np.bool_)
            fallback_seed[max(rows - 8, 0) :, :] = True

            best_label = 1
            best_score = -1
            for label in range(1, num_labels):
                component = labels == label
                area = int(stats[label, 4])
                if np.any(component & near_ego_seed):
                    score = area + 10_000
                elif np.any(component & fallback_seed):
                    score = area + 1_000
                else:
                    score = area
                if score > best_score:
                    best_score = score
                    best_label = label
            return (labels == best_label).astype(np.bool_)
        except Exception:
            return mask.astype(np.bool_)

    def _cleanup_grid(self, grid: np.ndarray) -> np.ndarray:
        binary = grid >= _GRID_KEEP_THRESHOLD
        if not binary.any():
            return np.zeros_like(grid, dtype=np.uint8)
        closed = binary
        for _ in range(_GRID_CLOSE_ITERATIONS):
            closed = self._binary_dilate(closed)
        for _ in range(_GRID_CLOSE_ITERATIONS):
            closed = self._binary_erode(closed)
        connected = self._keep_ego_connected_component(closed)
        filled = self._fill_holes(connected)
        smoothed = self._binary_dilate(filled)
        smoothed = self._keep_ego_connected_component(smoothed)
        cleaned = np.where(smoothed, np.maximum(grid, _GRID_CONFIDENCE_THRESHOLD), 0.0)
        return np.clip(cleaned, 0, 255).astype(np.uint8)

    def _decayed_confidence(
        self,
        confidence: float,
        updated_at_s: float,
        current_time_s: float,
    ) -> float:
        dt = max(float(current_time_s - updated_at_s), 0.0)
        if dt <= 0.0:
            return float(confidence)
        return float(confidence) * math.exp(-dt / _HISTORY_DECAY_S)

    def _lane_boundaries_from_centerline(self, segment: StaticLaneSegment) -> tuple[np.ndarray, np.ndarray]:
        centerline = np.asarray(segment.centerline_world, dtype=np.float32)
        if len(centerline) < 2:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        headings: list[float] = []
        for index in range(len(centerline)):
            if index == len(centerline) - 1:
                delta = centerline[index] - centerline[index - 1]
            else:
                delta = centerline[index + 1] - centerline[index]
            headings.append(math.atan2(float(delta[1]), float(delta[0])))
        half_width = 1.75
        left_points: list[np.ndarray] = []
        right_points: list[np.ndarray] = []
        for point, heading in zip(centerline, headings, strict=False):
            normal = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=np.float32)
            left_points.append(point + (normal * half_width))
            right_points.append(point - (normal * half_width))
        return np.asarray(left_points, dtype=np.float32), np.asarray(right_points, dtype=np.float32)

    def _lane_boundaries(self, segment: StaticLaneSegment) -> tuple[np.ndarray, np.ndarray]:
        left = np.asarray(segment.left_boundary_world, dtype=np.float32)
        right = np.asarray(segment.right_boundary_world, dtype=np.float32)
        if len(left) >= 2 and len(right) >= 2:
            return left, right
        return self._lane_boundaries_from_centerline(segment)

    def _route_plan_window(
        self,
        route_plan: RoutePlan | None,
        ego_pose: EgoPose,
        *,
        behind_distance_m: float = 20.0,
        ahead_distance_m: float = 60.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if route_plan is None or not route_plan.waypoints:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        ego_xy = np.asarray(ego_pose.world_xyz[:2], dtype=np.float32)
        nearest_index = min(
            range(len(route_plan.waypoints)),
            key=lambda index: float(
                np.linalg.norm(
                    np.array([route_plan.waypoints[index].x, route_plan.waypoints[index].y], dtype=np.float32) - ego_xy
                )
            ),
        )
        current_s = float(route_plan.waypoints[nearest_index].cumulative_distance_m)
        behind: list[list[float]] = []
        ahead: list[list[float]] = []
        for waypoint in route_plan.waypoints:
            point_xy = [float(waypoint.x), float(waypoint.y)]
            if waypoint.cumulative_distance_m < current_s - behind_distance_m:
                continue
            if waypoint.cumulative_distance_m <= current_s:
                behind.append(point_xy)
                continue
            if waypoint.cumulative_distance_m <= current_s + ahead_distance_m:
                ahead.append(point_xy)
        return (
            np.asarray(behind, dtype=np.float32),
            np.asarray(ahead, dtype=np.float32),
        )

    def _lane_reference_distance(
        self,
        segment: StaticLaneSegment,
        reference_points_xy: np.ndarray,
    ) -> float:
        centerline = np.asarray(segment.centerline_world, dtype=np.float32)
        if len(centerline) == 0 or reference_points_xy.size == 0:
            return float("inf")
        centerline_xy = centerline[:, :2]
        deltas = centerline_xy[:, None, :] - reference_points_xy[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        return float(np.min(distances))

    def _successor_chain(
        self,
        start_lane_id: str,
        lane_lookup: dict[str, StaticLaneSegment],
        future_route_xy: np.ndarray,
        *,
        max_hops: int = 5,
    ) -> list[str]:
        selected: list[str] = []
        current_lane_id = start_lane_id
        visited = {start_lane_id}
        for _ in range(max_hops):
            current = lane_lookup.get(current_lane_id)
            if current is None or not current.successor_lane_ids:
                break
            candidates = [
                lane_lookup[lane_id]
                for lane_id in current.successor_lane_ids
                if lane_id in lane_lookup and lane_id not in visited
            ]
            if not candidates:
                break
            best = min(
                candidates,
                key=lambda segment: (
                    self._lane_reference_distance(segment, future_route_xy),
                    float(np.linalg.norm(np.mean(np.asarray(segment.centerline_world, dtype=np.float32)[:, :2], axis=0))),
                ),
            )
            selected.append(best.lane_id)
            visited.add(best.lane_id)
            current_lane_id = best.lane_id
        return selected

    def _predecessor_chain(
        self,
        start_lane_id: str,
        lane_lookup: dict[str, StaticLaneSegment],
        previous_route_xy: np.ndarray,
        *,
        max_hops: int = 2,
    ) -> list[str]:
        selected: list[str] = []
        current_lane_id = start_lane_id
        visited = {start_lane_id}
        for _ in range(max_hops):
            current = lane_lookup.get(current_lane_id)
            if current is None or not current.predecessor_lane_ids:
                break
            candidates = [
                lane_lookup[lane_id]
                for lane_id in current.predecessor_lane_ids
                if lane_id in lane_lookup and lane_id not in visited
            ]
            if not candidates:
                break
            best = min(
                candidates,
                key=lambda segment: (
                    self._lane_reference_distance(segment, previous_route_xy),
                    float(np.linalg.norm(np.mean(np.asarray(segment.centerline_world, dtype=np.float32)[:, :2], axis=0))),
                ),
            )
            selected.append(best.lane_id)
            visited.add(best.lane_id)
            current_lane_id = best.lane_id
        selected.reverse()
        return selected

    def _current_lane_id(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
    ) -> str | None:
        lane_lookup = {lane.lane_id: lane for lane in local_map.static_lanes}
        if ego_pose.current_lane_id in lane_lookup:
            return ego_pose.current_lane_id
        ego_xy = np.asarray(ego_pose.world_xyz[:2], dtype=np.float32)
        best_lane_id: str | None = None
        best_distance = float("inf")
        for segment in local_map.static_lanes:
            centerline_xy = np.asarray(segment.centerline_world, dtype=np.float32)[:, :2]
            if len(centerline_xy) == 0:
                continue
            distance = float(np.min(np.linalg.norm(centerline_xy - ego_xy[None, :], axis=1)))
            if distance < best_distance:
                best_distance = distance
                best_lane_id = segment.lane_id
        return best_lane_id

    def build_route_corridor(
        self,
        local_map: LocalMap | None,
        ego_pose: EgoPose | None,
        *,
        route_plan: RoutePlan | None = None,
    ) -> dict[str, Any]:
        if local_map is None or ego_pose is None or not local_map.static_lanes:
            return {"strips": [], "polygons_xy": []}
        lane_lookup = {lane.lane_id: lane for lane in local_map.static_lanes}
        current_lane_id = self._current_lane_id(local_map, ego_pose)
        if current_lane_id is None:
            return {"strips": [], "polygons_xy": []}
        previous_route_xy, future_route_xy = self._route_plan_window(route_plan, ego_pose)
        selected_lane_ids = (
            self._predecessor_chain(current_lane_id, lane_lookup, previous_route_xy)
            + [current_lane_id]
            + self._successor_chain(current_lane_id, lane_lookup, future_route_xy)
        )
        unique_lane_ids: list[str] = []
        seen: set[str] = set()
        for lane_id in selected_lane_ids:
            if lane_id in seen or lane_id not in lane_lookup:
                continue
            seen.add(lane_id)
            unique_lane_ids.append(lane_id)
        corridor_cache_key = tuple(unique_lane_ids)
        if corridor_cache_key == self._corridor_cache_key and self._corridor_cache_payload is not None:
            return self._corridor_cache_payload

        strips: list[dict[str, Any]] = []
        polygons_xy: list[np.ndarray] = []
        for lane_id in unique_lane_ids:
            segment = lane_lookup[lane_id]
            left_boundary, right_boundary = self._lane_boundaries(segment)
            if len(left_boundary) < 2 or len(right_boundary) < 2:
                continue
            polygon_world = np.vstack([left_boundary, right_boundary[::-1]])
            strips.append(
                {
                    "lane_id": lane_id,
                    "left_boundary_world": left_boundary.tolist(),
                    "right_boundary_world": right_boundary.tolist(),
                    "polygon_world": polygon_world.tolist(),
                    "is_junction": bool(segment.is_junction),
                }
            )
            polygons_xy.append(np.asarray(polygon_world[:, :2], dtype=np.float32))
        payload = {"strips": strips, "polygons_xy": polygons_xy}
        self._corridor_cache_key = corridor_cache_key
        self._corridor_cache_payload = payload
        return payload

    def _points_in_polygon(self, points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0 or len(polygon_xy) < 3:
            return np.zeros(points_xy.shape[0], dtype=np.bool_)
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        poly_x = polygon_xy[:, 0]
        poly_y = polygon_xy[:, 1]
        inside = np.zeros(points_xy.shape[0], dtype=np.bool_)
        j = len(polygon_xy) - 1
        for i in range(len(polygon_xy)):
            intersects = ((poly_y[i] > y) != (poly_y[j] > y)) & (
                x < ((poly_x[j] - poly_x[i]) * (y - poly_y[i]) / ((poly_y[j] - poly_y[i]) + 1e-6)) + poly_x[i]
            )
            inside ^= intersects
            j = i
        return inside

    def _clip_points_to_corridor(
        self,
        points_world_xy: np.ndarray,
        confidences: np.ndarray,
        polygons_xy: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if points_world_xy.size == 0 or not polygons_xy:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        inside_any = np.zeros(points_world_xy.shape[0], dtype=np.bool_)
        for polygon_xy in polygons_xy:
            inside_any |= self._points_in_polygon(points_world_xy, polygon_xy)
        return points_world_xy[inside_any], confidences[inside_any]

    def _downsample_history_points(
        self,
        points_world_xy: np.ndarray,
        confidences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if points_world_xy.shape[0] <= _MAX_HISTORY_SAMPLES:
            return points_world_xy, confidences
        indices = np.linspace(
            0,
            points_world_xy.shape[0] - 1,
            _MAX_HISTORY_SAMPLES,
            dtype=np.int32,
        )
        return points_world_xy[indices], confidences[indices]

    def _ego_to_world_xy(self, points_ego: np.ndarray, ego_pose: EgoPose) -> np.ndarray:
        yaw = float(ego_pose.yaw_rad)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation = np.array(
            [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
            dtype=np.float32,
        )
        ego_xy = np.asarray(ego_pose.world_xyz[:2], dtype=np.float32)
        return (np.asarray(points_ego[:, :2], dtype=np.float32) @ rotation.T) + ego_xy

    def _project_ground_points_ego(
        self,
        drivable: DrivableSpaceMask,
        *,
        camera_calibration: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_height, image_width = drivable.mask.shape[:2]
        calibration = self._normalise_calibration(
            camera_calibration,
            image_height=image_height,
            image_width=image_width,
        )
        cache_key = (
            image_height,
            image_width,
            float(calibration["fov_deg"]),
            *[float(v) for v in calibration["mount_xyz"]],
            *[float(v) for v in calibration["mount_rpy_deg"]],
        )
        if cache_key != self._cache_key or self._ray_directions_ego is None:
            self._build_lookup(calibration)

        candidate_mask, values = self._projection_mask(drivable)
        keep = candidate_mask.reshape(-1)
        if not keep.any():
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

        directions = self._ray_directions_ego[keep]
        confidences = values.reshape(-1)[keep] * 255.0

        downward = directions[:, 2] < -1e-4
        if not np.any(downward):
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        directions = directions[downward]
        confidences = confidences[downward]

        distances = -self._camera_origin_ego[2] / directions[:, 2]
        valid_distance = distances > 0.0
        if not np.any(valid_distance):
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        directions = directions[valid_distance]
        confidences = confidences[valid_distance]
        distances = distances[valid_distance]

        points_ego = self._camera_origin_ego + (directions * distances[:, None])
        forward_m = points_ego[:, 0]
        lateral_m = points_ego[:, 1]
        in_bounds = (
            (forward_m >= _CROP_X_MIN_M)
            & (forward_m < _CROP_X_MAX_M + 20.0)
            & (lateral_m >= _CROP_Y_MIN_M - 10.0)
            & (lateral_m < _CROP_Y_MAX_M + 10.0)
        )
        if not np.any(in_bounds):
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return points_ego[in_bounds], confidences[in_bounds]

    def project_world_points(
        self,
        drivable: DrivableSpaceMask,
        ego_pose: EgoPose,
        *,
        camera_calibration: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points_ego, confidences = self._project_ground_points_ego(
            drivable,
            camera_calibration=camera_calibration,
        )
        if points_ego.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return self._ego_to_world_xy(points_ego, ego_pose), confidences

    def _update_history_cell(
        self,
        key: tuple[int, int],
        confidence: float,
        sim_time_s: float,
    ) -> None:
        previous = self._history.get(key)
        if previous is not None:
            decayed_previous = self._decayed_confidence(previous[0], previous[1], sim_time_s)
            confidence = max(float(confidence), decayed_previous)
        self._history[key] = (float(confidence), float(sim_time_s))

    def update_world_history(
        self,
        points_world_xy: np.ndarray,
        confidences: np.ndarray,
        *,
        sim_time_s: float,
        corridor_polygons_xy: list[np.ndarray],
    ) -> None:
        if points_world_xy.size == 0:
            self._prune_history(sim_time_s=sim_time_s)
            return
        sampled_points, sampled_confidences = self._downsample_history_points(
            np.asarray(points_world_xy, dtype=np.float32),
            np.asarray(confidences, dtype=np.float32),
        )
        clipped_points, clipped_confidences = self._clip_points_to_corridor(
            sampled_points,
            sampled_confidences,
            corridor_polygons_xy,
        )
        if clipped_points.size == 0:
            self._prune_history(sim_time_s=sim_time_s)
            return

        grid_pos = (clipped_points / _HISTORY_CELL_SIZE_M) - 0.5
        grid_floor = np.floor(grid_pos).astype(np.int32)
        grid_frac = grid_pos - grid_floor

        candidates = (
            (grid_floor[:, 0], grid_floor[:, 1], (1.0 - grid_frac[:, 0]) * (1.0 - grid_frac[:, 1])),
            (grid_floor[:, 0] + 1, grid_floor[:, 1], grid_frac[:, 0] * (1.0 - grid_frac[:, 1])),
            (grid_floor[:, 0], grid_floor[:, 1] + 1, (1.0 - grid_frac[:, 0]) * grid_frac[:, 1]),
            (grid_floor[:, 0] + 1, grid_floor[:, 1] + 1, grid_frac[:, 0] * grid_frac[:, 1]),
        )

        for ix, iy, weights in candidates:
            valid = weights > 1e-6
            if not np.any(valid):
                continue
            weighted = clipped_confidences[valid] * weights[valid]
            for cell_x, cell_y, confidence in zip(ix[valid], iy[valid], weighted, strict=False):
                self._update_history_cell((int(cell_x), int(cell_y)), float(confidence), sim_time_s)

        self._prune_history(sim_time_s=sim_time_s)

    def _prune_history(
        self,
        *,
        sim_time_s: float,
        ego_world_xy: np.ndarray | None = None,
    ) -> None:
        stale_keys: list[tuple[int, int]] = []
        for key, (confidence, updated_at) in self._history.items():
            decayed = self._decayed_confidence(confidence, updated_at, sim_time_s)
            if decayed < 12.0:
                stale_keys.append(key)
                continue
            if ego_world_xy is not None:
                world_xy = (np.array(key, dtype=np.float32) + 0.5) * _HISTORY_CELL_SIZE_M
                if float(np.linalg.norm(world_xy - ego_world_xy)) > _HISTORY_PRUNE_DISTANCE_M:
                    stale_keys.append(key)
        for key in stale_keys:
            self._history.pop(key, None)

    def render_local_crop(
        self,
        ego_pose: EgoPose,
        *,
        sim_time_s: float,
    ) -> dict[str, Any]:
        rows = int(round((_CROP_X_MAX_M - _CROP_X_MIN_M) / _HISTORY_CELL_SIZE_M))
        cols = int(round((_CROP_Y_MAX_M - _CROP_Y_MIN_M) / _HISTORY_CELL_SIZE_M))
        grid = np.zeros((rows, cols), dtype=np.uint8)
        ego_xy = np.asarray(ego_pose.world_xyz[:2], dtype=np.float32)
        self._prune_history(sim_time_s=sim_time_s, ego_world_xy=ego_xy)

        yaw = float(ego_pose.yaw_rad)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation_world_to_ego = np.array(
            [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
            dtype=np.float32,
        )

        for key, (confidence, updated_at_s) in self._history.items():
            decayed = self._decayed_confidence(confidence, updated_at_s, sim_time_s)
            if decayed < 20.0:
                continue
            world_xy = (np.array(key, dtype=np.float32) + 0.5) * _HISTORY_CELL_SIZE_M
            local_xy = (world_xy - ego_xy) @ rotation_world_to_ego.T
            local_x = float(local_xy[0])
            local_y = float(local_xy[1])
            if not (_CROP_X_MIN_M <= local_x < _CROP_X_MAX_M and _CROP_Y_MIN_M <= local_y < _CROP_Y_MAX_M):
                continue
            row = int(np.floor((_CROP_X_MAX_M - local_x) / _HISTORY_CELL_SIZE_M))
            col = int(np.floor((local_y - _CROP_Y_MIN_M) / _HISTORY_CELL_SIZE_M))
            if 0 <= row < rows and 0 <= col < cols:
                grid[row, col] = max(grid[row, col], int(np.clip(decayed, 0.0, 255.0)))

        return {
            "grid": grid,
            "rows": rows,
            "cols": cols,
            "cell_size_m": _HISTORY_CELL_SIZE_M,
            "x_min_m": _CROP_X_MIN_M,
            "x_max_m": _CROP_X_MAX_M,
            "y_min_m": _CROP_Y_MIN_M,
            "y_max_m": _CROP_Y_MAX_M,
        }

    def project(
        self,
        drivable: DrivableSpaceMask,
        ego_pose: EgoPose,
        *,
        camera_calibration: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del ego_pose
        image_height, image_width = drivable.mask.shape[:2]
        calibration = self._normalise_calibration(
            camera_calibration,
            image_height=image_height,
            image_width=image_width,
        )
        cache_key = (
            image_height,
            image_width,
            float(calibration["fov_deg"]),
            *[float(v) for v in calibration["mount_xyz"]],
            *[float(v) for v in calibration["mount_rpy_deg"]],
        )
        if cache_key != self._cache_key or self._ray_directions_ego is None:
            self._build_lookup(calibration)

        candidate_mask, values = self._projection_mask(drivable)
        keep = candidate_mask.reshape(-1)
        if not keep.any():
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        directions = self._ray_directions_ego[keep]
        confidences = values.reshape(-1)[keep] * 255.0

        downward = directions[:, 2] < -1e-4
        if not np.any(downward):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        directions = directions[downward]
        confidences = confidences[downward]

        distances = -self._camera_origin_ego[2] / directions[:, 2]
        valid_distance = distances > 0.0
        if not np.any(valid_distance):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        directions = directions[valid_distance]
        confidences = confidences[valid_distance]
        distances = distances[valid_distance]

        points_ego = self._camera_origin_ego + (directions * distances[:, None])
        forward_m = points_ego[:, 0]
        lateral_m = points_ego[:, 1]
        in_bounds = (
            (forward_m >= 0.0)
            & (forward_m < FORWARD_RANGE_M)
            & (lateral_m >= -LATERAL_RANGE_M)
            & (lateral_m < LATERAL_RANGE_M)
        )
        if not np.any(in_bounds):
            return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        forward_m = forward_m[in_bounds]
        lateral_m = lateral_m[in_bounds]
        confidences = confidences[in_bounds]

        row_pos = ((FORWARD_RANGE_M - forward_m) / CELL_SIZE_M) - 0.5
        col_pos = ((lateral_m + LATERAL_RANGE_M) / CELL_SIZE_M) - 0.5
        row_floor = np.floor(row_pos).astype(np.int32)
        col_floor = np.floor(col_pos).astype(np.int32)
        row_frac = row_pos - row_floor
        col_frac = col_pos - col_floor

        row_candidates = (
            (row_floor, col_floor, (1.0 - row_frac) * (1.0 - col_frac)),
            (row_floor + 1, col_floor, row_frac * (1.0 - col_frac)),
            (row_floor, col_floor + 1, (1.0 - row_frac) * col_frac),
            (row_floor + 1, col_floor + 1, row_frac * col_frac),
        )

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for rows, cols, weights in row_candidates:
            valid_indices = (
                (rows >= 0)
                & (rows < GRID_SIZE)
                & (cols >= 0)
                & (cols < GRID_SIZE)
                & (weights > 1e-6)
            )
            if not np.any(valid_indices):
                continue
            weighted_confidences = confidences[valid_indices] * weights[valid_indices]
            np.maximum.at(
                grid,
                (rows[valid_indices], cols[valid_indices]),
                weighted_confidences,
            )
        return self._cleanup_grid(grid)
