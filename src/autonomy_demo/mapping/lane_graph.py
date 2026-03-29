from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autonomy_demo.common.geometry import distance_xy, normalize_angle, signed_lateral_error
from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import StaticLaneSegment


def lane_id_from_components(road_id: int, section_id: int, lane_id: int) -> str:
    return f"road_{road_id}:section_{section_id}:lane_{lane_id}"


def lane_id_from_waypoint(waypoint) -> str:
    return lane_id_from_components(
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
    )


def parse_lane_id(lane_id: str) -> tuple[int, int, int] | None:
    try:
        road_text, section_text, lane_text = lane_id.split(":")
        return (
            int(road_text.split("_", 1)[1]),
            int(section_text.split("_", 1)[1]),
            int(lane_text.split("_", 1)[1]),
        )
    except Exception:
        return None


@dataclass(slots=True)
class FrenetProjection:
    lane_id: str
    s: float
    d: float
    heading_rad: float
    heading_error_rad: float
    nearest_xyz: np.ndarray
    distance_m: float


@dataclass(slots=True)
class LaneGraph:
    segments: dict[str, StaticLaneSegment]

    def nearest_projection(
        self,
        world_xyz: np.ndarray,
        *,
        candidate_lane_ids: list[str] | None = None,
        max_height_delta_m: float = 4.0,
    ) -> FrenetProjection | None:
        lane_ids = candidate_lane_ids or list(self.segments.keys())
        filtered: list[tuple[float, FrenetProjection]] = []
        fallback: list[tuple[float, float, FrenetProjection]] = []
        for lane_id in lane_ids:
            segment = self.segments.get(lane_id)
            if segment is None or len(segment.centerline_world) < 2:
                continue
            projection = project_point_to_centerline(segment.centerline_world, world_xyz)
            projection.lane_id = segment.lane_id
            height_delta_m = float(abs(float(projection.nearest_xyz[2]) - float(world_xyz[2])))
            fallback.append((height_delta_m, projection.distance_m, projection))
            if height_delta_m <= max_height_delta_m:
                filtered.append((projection.distance_m, projection))
        if filtered:
            filtered.sort(key=lambda item: item[0])
            return filtered[0][1]
        if not fallback:
            return None
        fallback.sort(key=lambda item: (item[0], item[1]))
        return fallback[0][2]

    def nearby_lanes(
        self,
        world_xyz: np.ndarray,
        *,
        radius_m: float = 60.0,
        limit: int = 12,
        max_height_delta_m: float = 4.0,
    ) -> list[StaticLaneSegment]:
        filtered: list[tuple[float, StaticLaneSegment]] = []
        fallback: list[tuple[float, float, StaticLaneSegment]] = []
        for segment in self.segments.values():
            projection = project_point_to_centerline(segment.centerline_world, world_xyz)
            if projection.distance_m <= radius_m:
                height_delta_m = float(abs(float(projection.nearest_xyz[2]) - float(world_xyz[2])))
                fallback.append((height_delta_m, projection.distance_m, segment))
                if height_delta_m <= max_height_delta_m:
                    filtered.append((projection.distance_m, segment))
        if filtered:
            filtered.sort(key=lambda item: item[0])
            return [segment for _, segment in filtered[:limit]]
        fallback.sort(key=lambda item: (item[0], item[1]))
        return [segment for _, _, segment in fallback[:limit]]


class LaneGraphProvider:
    def __init__(self, *, sample_step_m: float = 4.0) -> None:
        self.sample_step_m = sample_step_m
        self.lane_graph: LaneGraph | None = None
        self.logger = get_logger(__name__, component="lane_graph")

    def prepare_from_simulation(self, simulation) -> LaneGraph | None:
        if self.lane_graph is not None:
            return self.lane_graph
        state = getattr(simulation, "state", None)
        world = getattr(state, "world", None)
        if world is None:
            self.logger.debug("No live world available; lane graph stays unavailable.")
            return None
        self.lane_graph = build_lane_graph_from_world(world, state.carla, step_m=self.sample_step_m)
        self.logger.info("Prepared lane graph with %s typed lane segments", len(self.lane_graph.segments))
        return self.lane_graph


def project_point_to_centerline(centerline_world: np.ndarray, world_xyz: np.ndarray) -> FrenetProjection:
    point = np.asarray(world_xyz, dtype=np.float32)
    centerline = np.asarray(centerline_world, dtype=np.float32)
    if len(centerline) < 2:
        nearest = centerline[0] if len(centerline) else np.zeros(3, dtype=np.float32)
        return FrenetProjection(
            lane_id="",
            s=0.0,
            d=0.0,
            heading_rad=0.0,
            heading_error_rad=0.0,
            nearest_xyz=np.asarray(nearest, dtype=np.float32),
            distance_m=float(distance_xy(point[:2], nearest[:2])),
        )

    best_distance = float("inf")
    best_projection: FrenetProjection | None = None
    cumulative_s = 0.0
    for index in range(len(centerline) - 1):
        start = centerline[index]
        end = centerline[index + 1]
        segment = end[:2] - start[:2]
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-6:
            continue
        point_delta = point[:2] - start[:2]
        progress = float(np.clip(np.dot(point_delta, segment) / (segment_length ** 2), 0.0, 1.0))
        nearest_xy = start[:2] + (segment * progress)
        nearest_z = float(start[2] + ((end[2] - start[2]) * progress))
        heading_rad = math.atan2(float(segment[1]), float(segment[0]))
        lateral_d = float(
            signed_lateral_error(
                float(start[0]),
                float(start[1]),
                heading_rad,
                float(point[0]),
                float(point[1]),
            )
        )
        distance_m = float(np.linalg.norm(point[:2] - nearest_xy))
        if distance_m < best_distance:
            best_distance = distance_m
            best_projection = FrenetProjection(
                lane_id="",
                s=float(cumulative_s + (progress * segment_length)),
                d=lateral_d,
                heading_rad=heading_rad,
                heading_error_rad=0.0,
                nearest_xyz=np.array([nearest_xy[0], nearest_xy[1], nearest_z], dtype=np.float32),
                distance_m=distance_m,
            )
        cumulative_s += segment_length

    if best_projection is None:
        nearest = centerline[0]
        return FrenetProjection(
            lane_id="",
            s=0.0,
            d=0.0,
            heading_rad=0.0,
            heading_error_rad=0.0,
            nearest_xyz=np.asarray(nearest, dtype=np.float32),
            distance_m=float(distance_xy(point[:2], nearest[:2])),
        )
    return best_projection


def sample_centerline_at_s(centerline_world: np.ndarray, s_m: float) -> tuple[np.ndarray, float]:
    centerline = np.asarray(centerline_world, dtype=np.float32)
    if len(centerline) == 0:
        return np.zeros(3, dtype=np.float32), 0.0
    if len(centerline) == 1:
        return centerline[0], 0.0
    remaining = max(float(s_m), 0.0)
    for index in range(len(centerline) - 1):
        start = centerline[index]
        end = centerline[index + 1]
        segment = end - start
        segment_length = float(np.linalg.norm(segment[:2]))
        if segment_length <= 1e-6:
            continue
        if remaining <= segment_length:
            ratio = remaining / segment_length
            point = start + (segment * ratio)
            heading = math.atan2(float(segment[1]), float(segment[0]))
            return np.asarray(point, dtype=np.float32), float(heading)
        remaining -= segment_length
    tail = centerline[-1]
    last_segment = centerline[-1] - centerline[-2]
    heading = math.atan2(float(last_segment[1]), float(last_segment[0]))
    return np.asarray(tail, dtype=np.float32), float(heading)


def _lane_boundaries(centerline_points: np.ndarray, headings_rad: np.ndarray, lane_width_m: float) -> tuple[np.ndarray, np.ndarray]:
    half_width = max(float(lane_width_m) * 0.5, 1.5)
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    for point, heading in zip(centerline_points, headings_rad, strict=False):
        normal = np.array([-math.sin(float(heading)), math.cos(float(heading)), 0.0], dtype=np.float32)
        left_points.append(point + (normal * half_width))
        right_points.append(point - (normal * half_width))
    return (
        np.asarray(left_points, dtype=np.float32),
        np.asarray(right_points, dtype=np.float32),
    )


def build_lane_graph_from_world(world, carla_module, *, step_m: float = 4.0) -> LaneGraph:
    carla_map = world.get_map()
    grouped: dict[str, list] = {}
    for waypoint in carla_map.generate_waypoints(step_m):
        if waypoint.lane_type != carla_module.LaneType.Driving:
            continue
        grouped.setdefault(lane_id_from_waypoint(waypoint), []).append(waypoint)

    segments: dict[str, StaticLaneSegment] = {}
    for lane_id, waypoints in grouped.items():
        ordered = sorted(waypoints, key=lambda waypoint: float(getattr(waypoint, "s", 0.0)))
        deduped: list = []
        seen_keys: set[tuple[float, float, float]] = set()
        for waypoint in ordered:
            location = waypoint.transform.location
            key = (round(float(location.x), 2), round(float(location.y), 2), round(float(location.z), 2))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(waypoint)
        if len(deduped) < 2:
            continue

        centerline_points = np.asarray(
            [
                [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z]
                for waypoint in deduped
            ],
            dtype=np.float32,
        )
        headings_rad = np.asarray(
            [math.radians(float(waypoint.transform.rotation.yaw)) for waypoint in deduped],
            dtype=np.float32,
        )
        left_boundary, right_boundary = _lane_boundaries(
            centerline_points,
            headings_rad,
            float(getattr(deduped[0], "lane_width", 3.5)),
        )
        predecessor_lane_ids = sorted(
            {
                lane_id_from_waypoint(prev)
                for prev in deduped[0].previous(step_m)
                if lane_id_from_waypoint(prev) != lane_id
            }
        )
        successor_lane_ids = sorted(
            {
                lane_id_from_waypoint(nxt)
                for nxt in deduped[-1].next(step_m)
                if lane_id_from_waypoint(nxt) != lane_id
            }
        )
        speed_limit_mps = 22.35
        if hasattr(deduped[0], "get_landmarks"):
            try:
                speed_landmarks = deduped[0].get_landmarks(20.0, stop_at_junction=False)
                for landmark in speed_landmarks:
                    if "speed_limit" in str(getattr(landmark, "type", "")):
                        speed_limit_mps = float(getattr(landmark, "value", speed_limit_mps * 3.6)) / 3.6
                        break
            except Exception:
                pass
        segments[lane_id] = StaticLaneSegment(
            lane_id=lane_id,
            centerline_world=centerline_points,
            speed_limit_mps=float(speed_limit_mps),
            left_boundary_world=left_boundary,
            right_boundary_world=right_boundary,
            predecessor_lane_ids=predecessor_lane_ids,
            successor_lane_ids=successor_lane_ids,
            is_junction=bool(any(bool(getattr(waypoint, "is_junction", False)) for waypoint in deduped)),
        )
    return LaneGraph(segments=segments)


def heading_error_to_lane(yaw_rad: float, projection: FrenetProjection) -> float:
    return float(normalize_angle(yaw_rad - projection.heading_rad))
