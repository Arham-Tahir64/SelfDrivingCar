from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autonomy_demo.common.geometry import distance_xy
from autonomy_demo.control.controller import RouteFollowerController
from autonomy_demo.interfaces.enums import BehaviorState, TrafficLightState
from autonomy_demo.interfaces.types import AgentPrediction, EgoPose, EgoTrajectory, LocalMap, Waypoint
from autonomy_demo.mapping.lane_graph import parse_lane_id, project_point_to_centerline, sample_centerline_at_s
from autonomy_demo.planning.route_following import RouteFollowerMotionPlanner

_PLANNING_HORIZON_S = 5.0
_PLANNING_DT_S = 0.1
_PLANNING_STEPS = int(_PLANNING_HORIZON_S / _PLANNING_DT_S)
_EGO_LENGTH_M = 4.6
_EGO_WIDTH_M = 2.0
_BASE_SAFETY_BUFFER_M = 0.5
_LEAD_LANE_TOLERANCE_M = 2.25


@dataclass(slots=True)
class PlannerCostBreakdown:
    collision: float = 0.0
    cone_proximity: float = 0.0
    lane_deviation: float = 0.0
    jerk: float = 0.0
    speed_error: float = 0.0
    traffic_violation: float = 0.0
    route_progress: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class PlannerCandidate:
    trajectory: EgoTrajectory
    lane_id: str
    target_speed_mps: float
    score: float = 0.0
    feasible: bool = True
    reject_reason: str | None = None
    reference_lane_id: str = ""
    target_lane_id: str = ""
    target_d_m: float = 0.0
    terminal_time_s: float = 0.0
    cost_breakdown: PlannerCostBreakdown = field(default_factory=PlannerCostBreakdown)


@dataclass(slots=True)
class _ReferencePath:
    lane_id: str
    centerline_world: np.ndarray
    speed_limit_mps: float
    source_kind: str
    cumulative_s: np.ndarray
    segment_lengths: np.ndarray
    segment_vectors: np.ndarray
    segment_headings: np.ndarray
    left_boundary_world: np.ndarray | None = None
    right_boundary_world: np.ndarray | None = None


@dataclass(slots=True)
class _MotionTarget:
    target_lane_id: str
    target_d_m: float
    target_speed_mps: float
    terminal_time_s: float
    target_s_m: float | None = None


@dataclass(slots=True)
class _TrajectorySamples:
    s: np.ndarray
    s_dot: np.ndarray
    s_ddot: np.ndarray
    s_jerk: np.ndarray
    d: np.ndarray
    d_dot: np.ndarray
    d_ddot: np.ndarray
    d_jerk: np.ndarray
    relative_progress_m: np.ndarray
    world_points_xy: np.ndarray
    world_waypoints: list[Waypoint]
    max_curvature: float


@dataclass(slots=True)
class _DynamicClearance:
    min_clearance_m: float
    min_boundary_clearance_m: float


@dataclass(slots=True)
class _Polynomial:
    coefficients: np.ndarray

    def value(self, t: float) -> float:
        return float(np.polyval(self.coefficients[::-1], t))

    def derivative(self, t: float, order: int) -> float:
        return float(self.derivative_values(np.asarray([t], dtype=np.float64), order)[0])

    def values(self, t_values: np.ndarray) -> np.ndarray:
        values = np.asarray(t_values, dtype=np.float64)
        result = np.zeros_like(values, dtype=np.float64)
        for coefficient in self.coefficients[::-1]:
            result = (result * values) + float(coefficient)
        return result

    def derivative_values(self, t_values: np.ndarray, order: int) -> np.ndarray:
        values = np.asarray(t_values, dtype=np.float64)
        coeffs = self.coefficients.astype(np.float64, copy=True)
        for _ in range(order):
            coeffs = np.array([index * coeffs[index] for index in range(1, len(coeffs))], dtype=np.float64)
            if len(coeffs) == 0:
                return np.zeros_like(values, dtype=np.float64)
        result = np.zeros_like(values, dtype=np.float64)
        for coefficient in coeffs[::-1]:
            result = (result * values) + float(coefficient)
        return result


@dataclass(slots=True)
class _PlannerRunContext:
    reference_projection: any
    closed_lanes: set[str]
    safety_margins: dict[int, float]
    dynamic_obstacles_by_step: list[list[tuple[int, np.ndarray, float]]]


def _solve_quintic(
    *,
    x0: float,
    x_dot0: float,
    x_ddot0: float,
    xT: float,
    x_dotT: float,
    x_ddotT: float,
    T: float,
) -> _Polynomial:
    a0 = x0
    a1 = x_dot0
    a2 = x_ddot0 * 0.5
    matrix = np.array(
        [
            [T**3, T**4, T**5],
            [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
            [6.0 * T, 12.0 * T**2, 20.0 * T**3],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            xT - (a0 + (a1 * T) + (a2 * T**2)),
            x_dotT - (a1 + (2.0 * a2 * T)),
            x_ddotT - (2.0 * a2),
        ],
        dtype=np.float64,
    )
    a3, a4, a5 = np.linalg.solve(matrix, rhs)
    return _Polynomial(np.array([a0, a1, a2, a3, a4, a5], dtype=np.float64))


def _solve_quartic(
    *,
    x0: float,
    x_dot0: float,
    x_ddot0: float,
    x_dotT: float,
    x_ddotT: float,
    T: float,
) -> _Polynomial:
    a0 = x0
    a1 = x_dot0
    a2 = x_ddot0 * 0.5
    matrix = np.array(
        [
            [3.0 * T**2, 4.0 * T**3],
            [6.0 * T, 12.0 * T**2],
        ],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            x_dotT - (a1 + (2.0 * a2 * T)),
            x_ddotT - (2.0 * a2),
        ],
        dtype=np.float64,
    )
    a3, a4 = np.linalg.solve(matrix, rhs)
    return _Polynomial(np.array([a0, a1, a2, a3, a4], dtype=np.float64))


class StubMotionPlanner:
    """Fallback planner for stub mode and degraded cases."""

    def run(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
    ) -> EgoTrajectory:
        waypoints = [
            Waypoint(
                x=float(ego_pose.world_xyz[0] + step * 2.0),
                y=float(ego_pose.world_xyz[1]),
                yaw=ego_pose.yaw_rad,
                velocity=ego_pose.speed_mps,
                timestamp=step * 0.1,
            )
            for step in range(5)
        ]
        return EgoTrajectory(waypoints=waypoints, cost=0.1, behavior_state=behavior_state)


class FrenetMotionPlanner:
    """PRD-style Frenet lattice planner with feasibility filtering and weighted costs."""

    def __init__(
        self,
        *,
        horizon_steps: int = _PLANNING_STEPS,
        dt_s: float = _PLANNING_DT_S,
        cruise_speed_mps: float = 12.0,
        terminal_times_s: tuple[float, ...] = (3.5, 5.0),
        max_candidate_count: int = 50,
        max_curvature_rad_per_m: float = 0.35,
        max_lateral_jerk_mps3: float = 6.0,
        max_longitudinal_jerk_mps3: float = 12.0,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.dt_s = dt_s
        self.planning_horizon_s = float(horizon_steps * dt_s)
        self.cruise_speed_mps = cruise_speed_mps
        self.terminal_times_s = tuple(sorted(float(value) for value in terminal_times_s))
        self.max_candidate_count = max_candidate_count
        self.max_curvature_rad_per_m = max_curvature_rad_per_m
        self.max_lateral_jerk_mps3 = max_lateral_jerk_mps3
        self.max_longitudinal_jerk_mps3 = max_longitudinal_jerk_mps3
        self.route_plan = None
        self._fallback = RouteFollowerMotionPlanner(
            target_speed_mps=cruise_speed_mps,
            horizon_waypoints=max(horizon_steps, 6),
        )
        self.last_candidates: list[PlannerCandidate] = []

    def _build_reference_path(
        self,
        *,
        lane_id: str,
        centerline_world: np.ndarray,
        speed_limit_mps: float,
        source_kind: str,
        left_boundary_world: np.ndarray | None = None,
        right_boundary_world: np.ndarray | None = None,
    ) -> _ReferencePath:
        centerline_world = np.asarray(centerline_world, dtype=np.float32)
        segment_vectors = np.diff(centerline_world, axis=0)
        if len(segment_vectors) == 0:
            segment_vectors = np.zeros((1, 3), dtype=np.float32)
            segment_lengths = np.ones(1, dtype=np.float32)
            segment_headings = np.zeros(1, dtype=np.float32)
            cumulative_s = np.array([0.0, 1.0], dtype=np.float32)
        else:
            segment_lengths = np.linalg.norm(segment_vectors[:, :2], axis=1).astype(np.float32)
            segment_lengths = np.where(segment_lengths > 1e-6, segment_lengths, 1e-6).astype(np.float32)
            segment_headings = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0]).astype(np.float32)
            cumulative_s = np.concatenate(
                [np.array([0.0], dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
            )
        return _ReferencePath(
            lane_id=lane_id,
            centerline_world=centerline_world,
            speed_limit_mps=float(speed_limit_mps),
            source_kind=source_kind,
            cumulative_s=cumulative_s,
            segment_lengths=segment_lengths,
            segment_vectors=segment_vectors,
            segment_headings=segment_headings,
            left_boundary_world=left_boundary_world,
            right_boundary_world=right_boundary_world,
        )

    def _sample_reference_points(
        self,
        reference_path: _ReferencePath,
        s_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        clamped_s = np.clip(np.asarray(s_values, dtype=np.float64), 0.0, float(reference_path.cumulative_s[-1]))
        segment_indices = np.searchsorted(reference_path.cumulative_s[1:], clamped_s, side="right")
        segment_indices = np.clip(segment_indices, 0, len(reference_path.segment_lengths) - 1)
        segment_start_s = reference_path.cumulative_s[segment_indices]
        ratios = ((clamped_s - segment_start_s) / reference_path.segment_lengths[segment_indices]).astype(np.float32)
        points = (
            reference_path.centerline_world[segment_indices]
            + (reference_path.segment_vectors[segment_indices] * ratios[:, None])
        ).astype(np.float32)
        headings = reference_path.segment_headings[segment_indices].astype(np.float64)
        return points, headings

    def _effective_terminal_times(self) -> list[float]:
        if self.planning_horizon_s <= 0.0:
            return [0.1]
        effective = sorted(
            {
                round(min(max(float(value), self.dt_s), self.planning_horizon_s), 4)
                for value in self.terminal_times_s
            }
        )
        return effective or [round(self.planning_horizon_s, 4)]

    def prepare_route(self, simulation, scenario) -> None:
        self._fallback.prepare_route(simulation, scenario)
        self.route_plan = self._fallback.route_plan

    def run(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
    ) -> EgoTrajectory:
        self.last_candidates = []
        reference_path = self._select_reference_path(local_map, ego_pose, behavior_state)
        if reference_path is None:
            return self._fallback.run(local_map, ego_pose, predictions, behavior_state)

        motion_targets = self._build_motion_targets(local_map, ego_pose, behavior_state, reference_path)
        if not motion_targets:
            return self._fallback.run(local_map, ego_pose, predictions, behavior_state)

        safety_margins = self._prediction_safety_map(local_map, predictions)
        run_context = _PlannerRunContext(
            reference_projection=project_point_to_centerline(reference_path.centerline_world, ego_pose.world_xyz),
            closed_lanes=set(local_map.closed_lanes),
            safety_margins=safety_margins,
            dynamic_obstacles_by_step=self._prepare_dynamic_obstacles(local_map, predictions, safety_margins),
        )

        candidates = [
            self._evaluate_candidate(
                local_map=local_map,
                ego_pose=ego_pose,
                predictions=predictions,
                behavior_state=behavior_state,
                reference_path=reference_path,
                target=target,
                run_context=run_context,
            )
            for target in motion_targets
        ]
        self.last_candidates = candidates

        feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
        if feasible_candidates:
            best = min(feasible_candidates, key=lambda candidate: candidate.score)
            best.trajectory.cost = float(best.score)
            return best.trajectory

        stop_candidate = self._synthesize_stop_candidate(
            local_map=local_map,
            ego_pose=ego_pose,
            predictions=predictions,
            behavior_state=behavior_state,
            reference_path=reference_path,
            run_context=run_context,
        )
        if stop_candidate is not None:
            self.last_candidates.append(stop_candidate)
            stop_candidate.trajectory.cost = float(stop_candidate.score)
            return stop_candidate.trajectory

        return self._fallback.run(local_map, ego_pose, predictions, behavior_state)

    def _select_reference_path(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
    ) -> _ReferencePath | None:
        if behavior_state not in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING}:
            route_reference = self._route_reference(local_map, ego_pose)
            if route_reference is not None:
                return route_reference
        lane = self._reference_lane(local_map, ego_pose)
        if lane is None:
            return None
        return self._build_reference_path(
            lane_id=lane.lane_id,
            centerline_world=np.asarray(lane.centerline_world, dtype=np.float32),
            left_boundary_world=np.asarray(lane.left_boundary_world, dtype=np.float32),
            right_boundary_world=np.asarray(lane.right_boundary_world, dtype=np.float32),
            speed_limit_mps=float(lane.speed_limit_mps),
            source_kind="lane",
        )

    def _route_reference(self, local_map: LocalMap, ego_pose: EgoPose) -> _ReferencePath | None:
        if self.route_plan is None or not self.route_plan.waypoints:
            return None
        centerline_world = np.asarray(
            [[waypoint.x, waypoint.y, waypoint.z] for waypoint in self.route_plan.waypoints],
            dtype=np.float32,
        )
        nearest_lane = self._reference_lane(local_map, ego_pose)
        speed_limit = (
            float(nearest_lane.speed_limit_mps)
            if nearest_lane is not None and nearest_lane.speed_limit_mps > 0.0
            else self.cruise_speed_mps
        )
        return self._build_reference_path(
            lane_id="route_reference",
            centerline_world=centerline_world,
            left_boundary_world=(
                None
                if nearest_lane is None
                else np.asarray(nearest_lane.left_boundary_world, dtype=np.float32)
            ),
            right_boundary_world=(
                None
                if nearest_lane is None
                else np.asarray(nearest_lane.right_boundary_world, dtype=np.float32)
            ),
            speed_limit_mps=speed_limit,
            source_kind="route",
        )

    def _reference_lane(self, local_map: LocalMap, ego_pose: EgoPose):
        if not local_map.static_lanes:
            return None
        for lane in local_map.static_lanes:
            if lane.lane_id == ego_pose.current_lane_id:
                return lane
        ego_xy = np.asarray(ego_pose.world_xyz, dtype=np.float32)[:2]
        return min(
            local_map.static_lanes,
            key=lambda lane: project_point_to_centerline(
                np.asarray(lane.centerline_world, dtype=np.float32),
                np.asarray([ego_xy[0], ego_xy[1], 0.0], dtype=np.float32),
            ).distance_m,
        )

    def _build_motion_targets(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
        reference_path: _ReferencePath,
    ) -> list[_MotionTarget]:
        target_speed_values = self._target_speed_values(ego_pose, behavior_state, reference_path)
        terminal_times = self._effective_terminal_times()
        current_lane = self._reference_lane(local_map, ego_pose)
        stop_progress_m = self._behavior_stop_progress(local_map, ego_pose, behavior_state, current_lane)
        target_s_m = (
            None
            if stop_progress_m is None
            else float(ego_pose.frenet_s + stop_progress_m)
        )

        targets: list[_MotionTarget] = []
        if behavior_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING, BehaviorState.CONSTRUCTION_NAVIGATE}:
            merge_target_lane = None if current_lane is None else self._adjacent_lane(
                local_map,
                current_lane.lane_id,
                closed_lanes=set(local_map.closed_lanes),
            )
            merge_terminal_offset = 0.0
            if current_lane is not None and merge_target_lane is not None:
                merge_terminal_offset = self._target_lane_offset_m(
                    reference_centerline=np.asarray(current_lane.centerline_world, dtype=np.float32),
                    target_centerline=np.asarray(merge_target_lane.centerline_world, dtype=np.float32),
                    ego_pose=ego_pose,
                )
            target_offsets = (
                np.linspace(0.0, merge_terminal_offset, 5, dtype=np.float64)
                if abs(merge_terminal_offset) > 1e-3
                else np.zeros(5, dtype=np.float64)
            )
            for terminal_time_s in terminal_times:
                for target_speed_mps in target_speed_values:
                    for target_d_m in target_offsets:
                        target_lane_id = current_lane.lane_id if merge_target_lane is None or abs(target_d_m) < 1e-3 else merge_target_lane.lane_id
                        targets.append(
                            _MotionTarget(
                                target_lane_id=target_lane_id,
                                target_d_m=float(target_d_m),
                                target_speed_mps=float(target_speed_mps),
                                terminal_time_s=float(terminal_time_s),
                                target_s_m=target_s_m,
                            )
                        )
        else:
            lateral_targets = (
                [0.0]
                if behavior_state in {
                    BehaviorState.STOPPING_FOR_RED,
                    BehaviorState.PEDESTRIAN_YIELD,
                    BehaviorState.EMERGENCY_YIELD,
                    BehaviorState.GOAL_REACHED,
                }
                else [-1.0, -0.5, 0.0, 0.5, 1.0]
            )
            target_lane_id = reference_path.lane_id
            for terminal_time_s in terminal_times:
                for target_speed_mps in target_speed_values:
                    for target_d_m in lateral_targets:
                        targets.append(
                            _MotionTarget(
                                target_lane_id=target_lane_id,
                                target_d_m=float(target_d_m),
                                target_speed_mps=float(target_speed_mps),
                                terminal_time_s=float(terminal_time_s),
                                target_s_m=target_s_m,
                            )
                        )

        if len(targets) <= self.max_candidate_count:
            return targets
        ranked = sorted(
            targets,
            key=lambda item: (
                abs(item.target_d_m),
                abs(item.target_speed_mps - ego_pose.speed_mps),
                item.terminal_time_s,
            ),
        )
        return ranked[: self.max_candidate_count]

    def _target_speed_values(
        self,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
        reference_path: _ReferencePath,
    ) -> list[float]:
        speed_limit_mps = max(float(reference_path.speed_limit_mps), 0.0)
        cruise_target_mps = (
            speed_limit_mps
            if self.cruise_speed_mps <= 0.0
            else min(speed_limit_mps or self.cruise_speed_mps, self.cruise_speed_mps)
        )
        if behavior_state == BehaviorState.GOAL_REACHED:
            values = [0.0]
        elif behavior_state in {BehaviorState.STOPPING_FOR_RED, BehaviorState.PEDESTRIAN_YIELD}:
            values = np.linspace(0.0, max(min(ego_pose.speed_mps, cruise_target_mps * 0.4), 2.0), 5).tolist()
        elif behavior_state == BehaviorState.EMERGENCY_YIELD:
            values = np.linspace(0.0, max(min(ego_pose.speed_mps, cruise_target_mps * 0.35), 2.0), 5).tolist()
        elif behavior_state == BehaviorState.INTERSECTION_APPROACH:
            values = np.linspace(max(cruise_target_mps * 0.25, 2.0), max(cruise_target_mps * 0.6, 4.0), 5).tolist()
        elif behavior_state == BehaviorState.CONSTRUCTION_NAVIGATE:
            values = np.linspace(max(cruise_target_mps * 0.2, 2.0), max(cruise_target_mps * 0.45, 4.0), 5).tolist()
        elif behavior_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING}:
            values = np.linspace(
                max(cruise_target_mps * 0.45, 5.0),
                min(cruise_target_mps, max(ego_pose.speed_mps + 2.0, 6.0)),
                5,
            ).tolist()
        else:
            lower = max(min(ego_pose.speed_mps, cruise_target_mps * 0.8), 4.0)
            upper = max(cruise_target_mps, lower)
            values = np.linspace(lower, upper, 5).tolist()
        unique_sorted = sorted({round(max(float(value), 0.0), 4) for value in values})
        return unique_sorted[:5]

    def _behavior_stop_progress(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
        current_lane,
    ) -> float | None:
        if behavior_state in {BehaviorState.INTERSECTION_APPROACH, BehaviorState.STOPPING_FOR_RED}:
            return self._stop_distance(local_map, behavior_state)
        if behavior_state in {BehaviorState.PEDESTRIAN_YIELD, BehaviorState.EMERGENCY_YIELD, BehaviorState.GOAL_REACHED}:
            return self._dynamic_stop_progress(local_map, ego_pose, current_lane)
        return None

    def _dynamic_stop_progress(self, local_map: LocalMap, ego_pose: EgoPose, current_lane) -> float | None:
        if current_lane is None:
            return None
        ego_projection = project_point_to_centerline(
            np.asarray(current_lane.centerline_world, dtype=np.float32),
            ego_pose.world_xyz,
        )
        candidates: list[float] = []
        for detection in local_map.dynamic_agents:
            world_bbox_3d = np.asarray(detection.world_bbox_3d, dtype=np.float32)
            center_xyz = np.mean(world_bbox_3d, axis=0)
            projection = project_point_to_centerline(
                np.asarray(current_lane.centerline_world, dtype=np.float32),
                center_xyz,
            )
            longitudinal_gap = float(projection.s - ego_projection.s)
            lateral_gap = float(abs(projection.d))
            if longitudinal_gap <= 0.0 or lateral_gap > _LEAD_LANE_TOLERANCE_M:
                continue
            stopping_buffer_m = (
                self._ego_radius_m()
                + self._agent_radius_m(world_bbox_3d)
                + _BASE_SAFETY_BUFFER_M
                + 0.25
            )
            candidates.append(max(longitudinal_gap - stopping_buffer_m, 0.0))
        if not candidates:
            return None
        return float(min(candidates))

    def _target_lane_offset_m(
        self,
        *,
        reference_centerline: np.ndarray,
        target_centerline: np.ndarray,
        ego_pose: EgoPose,
    ) -> float:
        reference_projection = project_point_to_centerline(reference_centerline, ego_pose.world_xyz)
        target_projection = project_point_to_centerline(target_centerline, ego_pose.world_xyz)
        target_point, _ = sample_centerline_at_s(target_centerline, max(target_projection.s, 0.0))
        reference_point, reference_heading = sample_centerline_at_s(reference_centerline, max(reference_projection.s, 0.0))
        normal = np.array([-np.sin(reference_heading), np.cos(reference_heading)], dtype=np.float32)
        delta_xy = np.asarray(target_point[:2] - reference_point[:2], dtype=np.float32)
        return float(np.dot(delta_xy, normal))

    def _evaluate_candidate(
        self,
        *,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
        reference_path: _ReferencePath,
        target: _MotionTarget,
        run_context: _PlannerRunContext,
    ) -> PlannerCandidate:
        samples = self._sample_candidate(reference_path, ego_pose, target, run_context.reference_projection)
        trajectory = EgoTrajectory(
            waypoints=samples.world_waypoints,
            cost=0.0,
            behavior_state=behavior_state,
        )
        cost_breakdown = PlannerCostBreakdown()
        candidate = PlannerCandidate(
            trajectory=trajectory,
            lane_id=target.target_lane_id,
            target_speed_mps=target.target_speed_mps,
            feasible=True,
            reject_reason=None,
            reference_lane_id=reference_path.lane_id,
            target_lane_id=target.target_lane_id,
            target_d_m=target.target_d_m,
            terminal_time_s=target.terminal_time_s,
            cost_breakdown=cost_breakdown,
        )

        reject_reason = self._reject_reason(
            local_map=local_map,
            ego_pose=ego_pose,
            predictions=predictions,
            behavior_state=behavior_state,
            reference_path=reference_path,
            target=target,
            samples=samples,
            run_context=run_context,
        )
        if reject_reason is not None:
            candidate.feasible = False
            candidate.reject_reason = reject_reason
            candidate.score = 1_000_000.0 + float(abs(target.target_d_m) * 100.0) + target.target_speed_mps
            candidate.cost_breakdown.traffic_violation = 1.0 if reject_reason in {"closed_lane", "stop_line_violation"} else 0.0
            candidate.cost_breakdown.total = float(candidate.score)
            return candidate

        clearance = self._dynamic_clearance(local_map, samples, run_context)
        desired_lateral_path = (
            np.full_like(samples.d, target.target_d_m)
            if behavior_state in {
                BehaviorState.PREPARE_MERGE,
                BehaviorState.MERGING,
                BehaviorState.CONSTRUCTION_NAVIGATE,
            }
            else np.zeros_like(samples.d)
        )
        lane_deviation = float(np.mean(np.abs(samples.d - desired_lateral_path)))
        jerk_term = float(
            np.mean(np.abs(samples.d_jerk)) + (0.5 * np.mean(np.abs(samples.s_jerk)))
        )
        collision_term = max(0.0, min(1.0, (6.0 - clearance.min_clearance_m) / 6.0))
        boundary_term = max(0.0, min(1.0, (5.0 - clearance.min_boundary_clearance_m) / 5.0))
        speed_error_term = abs(float(samples.world_waypoints[-1].velocity) - target.target_speed_mps)
        route_progress_term = -0.05 * float(samples.relative_progress_m[-1])
        if behavior_state in {
            BehaviorState.PREPARE_MERGE,
            BehaviorState.MERGING,
            BehaviorState.CONSTRUCTION_NAVIGATE,
        }:
            route_progress_term -= 2.5 * abs(float(target.target_d_m))
        total = (
            (collision_term * 100.0)
            + (boundary_term * 40.0)
            + (lane_deviation * 5.0)
            + (jerk_term * 2.0)
            + (speed_error_term * 3.0)
            + route_progress_term
        )

        candidate.cost_breakdown.collision = float(collision_term)
        candidate.cost_breakdown.cone_proximity = float(boundary_term)
        candidate.cost_breakdown.lane_deviation = float(lane_deviation)
        candidate.cost_breakdown.jerk = float(jerk_term)
        candidate.cost_breakdown.speed_error = float(speed_error_term)
        candidate.cost_breakdown.traffic_violation = 0.0
        candidate.cost_breakdown.route_progress = float(route_progress_term)
        candidate.cost_breakdown.total = float(total)
        candidate.score = float(total)
        candidate.trajectory.cost = float(total)
        return candidate

    def _sample_candidate(
        self,
        reference_path: _ReferencePath,
        ego_pose: EgoPose,
        target: _MotionTarget,
        reference_projection,
    ) -> _TrajectorySamples:
        lateral_poly = _solve_quintic(
            x0=float(reference_projection.d),
            x_dot0=0.0,
            x_ddot0=0.0,
            xT=float(target.target_d_m),
            x_dotT=0.0,
            x_ddotT=0.0,
            T=target.terminal_time_s,
        )
        if target.target_s_m is None:
            longitudinal_poly = _solve_quartic(
                x0=float(reference_projection.s),
                x_dot0=max(float(ego_pose.speed_mps), 0.0),
                x_ddot0=float(ego_pose.acceleration_mps2),
                x_dotT=float(target.target_speed_mps),
                x_ddotT=0.0,
                T=target.terminal_time_s,
            )
        else:
            longitudinal_poly = _solve_quintic(
                x0=float(reference_projection.s),
                x_dot0=max(float(ego_pose.speed_mps), 0.0),
                x_ddot0=float(ego_pose.acceleration_mps2),
                xT=float(target.target_s_m),
                x_dotT=float(target.target_speed_mps),
                x_ddotT=0.0,
                T=target.terminal_time_s,
            )

        time_s = np.arange(1, self.horizon_steps + 1, dtype=np.float64) * float(self.dt_s)
        eval_time_s = np.minimum(time_s, float(target.terminal_time_s))

        s_values = longitudinal_poly.values(eval_time_s)
        s_dot_values = np.maximum(longitudinal_poly.derivative_values(eval_time_s, 1), 0.0)
        s_ddot_values = longitudinal_poly.derivative_values(eval_time_s, 2)
        s_jerk_values = longitudinal_poly.derivative_values(eval_time_s, 3)

        past_terminal_mask = time_s > float(target.terminal_time_s)
        if np.any(past_terminal_mask):
            delta_t = time_s[past_terminal_mask] - float(target.terminal_time_s)
            s_terminal = float(longitudinal_poly.value(float(target.terminal_time_s)))
            s_dot_terminal = max(float(longitudinal_poly.derivative(float(target.terminal_time_s), 1)), 0.0)
            if target.target_s_m is not None and target.target_speed_mps <= 0.1:
                s_values[past_terminal_mask] = s_terminal
            else:
                s_values[past_terminal_mask] = s_terminal + (s_dot_terminal * delta_t)
            s_dot_values[past_terminal_mask] = s_dot_terminal
            s_ddot_values[past_terminal_mask] = 0.0
            s_jerk_values[past_terminal_mask] = 0.0

        d_values = lateral_poly.values(eval_time_s)
        d_dot_values = lateral_poly.derivative_values(eval_time_s, 1)
        d_ddot_values = lateral_poly.derivative_values(eval_time_s, 2)
        d_jerk_values = lateral_poly.derivative_values(eval_time_s, 3)
        if np.any(past_terminal_mask):
            d_values[past_terminal_mask] = float(target.target_d_m)
            d_dot_values[past_terminal_mask] = 0.0
            d_ddot_values[past_terminal_mask] = 0.0
            d_jerk_values[past_terminal_mask] = 0.0

        center_points, center_headings = self._sample_reference_points(reference_path, s_values)
        normals = np.column_stack(
            [
                -np.sin(center_headings),
                np.cos(center_headings),
                np.zeros_like(center_headings),
            ]
        ).astype(np.float32)
        world_points = center_points + (normals * d_values[:, None].astype(np.float32))
        world_headings = center_headings + np.arctan2(d_dot_values, np.maximum(s_dot_values, 1e-3))
        world_velocities = np.maximum(s_dot_values, 0.0)

        if len(world_points) > 1:
            delta_xy = np.diff(world_points[:, :2], axis=0)
            segment_distances = np.maximum(np.linalg.norm(delta_xy, axis=1), 1e-3)
            heading_deltas = np.abs(np.diff(np.unwrap(world_headings)))
            max_curvature = float(np.max(heading_deltas / segment_distances))
        else:
            max_curvature = 0.0

        waypoints = [
            Waypoint(
                x=float(world_point[0]),
                y=float(world_point[1]),
                yaw=float(world_heading),
                velocity=float(world_velocity),
                timestamp=float(timestamp_s),
            )
            for world_point, world_heading, world_velocity, timestamp_s in zip(
                world_points,
                world_headings,
                world_velocities,
                time_s,
                strict=False,
            )
        ]

        relative_progress = np.asarray(s_values, dtype=np.float64) - float(reference_projection.s)
        return _TrajectorySamples(
            s=np.asarray(s_values, dtype=np.float64),
            s_dot=np.asarray(s_dot_values, dtype=np.float64),
            s_ddot=np.asarray(s_ddot_values, dtype=np.float64),
            s_jerk=np.asarray(s_jerk_values, dtype=np.float64),
            d=np.asarray(d_values, dtype=np.float64),
            d_dot=np.asarray(d_dot_values, dtype=np.float64),
            d_ddot=np.asarray(d_ddot_values, dtype=np.float64),
            d_jerk=np.asarray(d_jerk_values, dtype=np.float64),
            relative_progress_m=relative_progress,
            world_points_xy=np.asarray(world_points[:, :2], dtype=np.float32),
            world_waypoints=waypoints,
            max_curvature=float(max_curvature),
        )

    def _reject_reason(
        self,
        *,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
        reference_path: _ReferencePath,
        target: _MotionTarget,
        samples: _TrajectorySamples,
        run_context: _PlannerRunContext,
    ) -> str | None:
        if target.target_lane_id in run_context.closed_lanes:
            return "closed_lane"
        if samples.max_curvature > self.max_curvature_rad_per_m:
            return "curvature_limit"
        if np.max(np.abs(samples.d_jerk)) > self.max_lateral_jerk_mps3:
            return "lateral_jerk_limit"
        if np.max(np.abs(samples.s_jerk)) > self.max_longitudinal_jerk_mps3:
            return "longitudinal_jerk_limit"
        if self._corridor_violation(reference_path, target, samples):
            return "corridor_violation"
        if self._temporary_boundary_violation(local_map, samples):
            return "temporary_boundary"
        if self._stop_line_violation(local_map, behavior_state, samples):
            return "stop_line_violation"
        if self._dynamic_collision_violation(samples, run_context):
            return "dynamic_collision"
        if np.any(samples.relative_progress_m < -0.1):
            return "reverse_progress"
        return None

    def _corridor_violation(
        self,
        reference_path: _ReferencePath,
        target: _MotionTarget,
        samples: _TrajectorySamples,
    ) -> bool:
        if reference_path.source_kind == "route":
            corridor_limit = max(abs(target.target_d_m) + 1.5, 2.75)
        else:
            lane_half_width = self._reference_lane_half_width(reference_path)
            corridor_limit = max(lane_half_width + max(abs(target.target_d_m), 0.0), lane_half_width + 0.35)
        return bool(np.any(np.abs(samples.d) > corridor_limit))

    def _stop_line_violation(
        self,
        local_map: LocalMap,
        behavior_state: BehaviorState,
        samples: _TrajectorySamples,
    ) -> bool:
        stop_distance_m = self._stop_distance(local_map, behavior_state)
        if stop_distance_m is None:
            return False
        safe_stop_progress = max(stop_distance_m - 1.0, 0.0)
        if float(np.max(samples.relative_progress_m)) <= safe_stop_progress:
            return False
        final_speed = float(samples.world_waypoints[-1].velocity)
        return final_speed > 0.25 or float(np.max(samples.relative_progress_m)) > (stop_distance_m + 0.25)

    def _temporary_boundary_violation(self, local_map: LocalMap, samples: _TrajectorySamples) -> bool:
        if not local_map.temporary_boundaries:
            return False
        for lane in local_map.temporary_boundaries:
            polyline = np.asarray(lane.polyline_world, dtype=np.float32)
            if len(polyline) < 2:
                continue
            for point_xy in samples.world_points_xy:
                if self._distance_to_polyline(point_xy, polyline[:, :2]) < 0.35:
                    return True
        return False

    def _dynamic_collision_violation(
        self,
        samples: _TrajectorySamples,
        run_context: _PlannerRunContext,
    ) -> bool:
        for ego_xy, obstacles in zip(samples.world_points_xy, run_context.dynamic_obstacles_by_step, strict=False):
            for _track_id, agent_xy, safety_margin_m in obstacles:
                if distance_xy(ego_xy, agent_xy) < safety_margin_m:
                    return True
        return False

    def _dynamic_clearance(
        self,
        local_map: LocalMap,
        samples: _TrajectorySamples,
        run_context: _PlannerRunContext,
    ) -> _DynamicClearance:
        min_clearance_m = float("inf")
        for ego_xy, obstacles in zip(samples.world_points_xy, run_context.dynamic_obstacles_by_step, strict=False):
            for _track_id, agent_xy, safety_margin_m in obstacles:
                clearance_m = distance_xy(ego_xy, agent_xy) - safety_margin_m
                min_clearance_m = min(min_clearance_m, float(clearance_m))

        boundary_points: list[np.ndarray] = []
        for cone in local_map.cone_instances:
            boundary_points.append(np.asarray(cone.world_xyz[:2], dtype=np.float32))
        min_boundary_clearance_m = float("inf")
        if boundary_points:
            for waypoint_xy in samples.world_points_xy:
                for cone_xy in boundary_points:
                    min_boundary_clearance_m = min(min_boundary_clearance_m, float(distance_xy(waypoint_xy, cone_xy)))
        elif local_map.temporary_boundaries:
            for lane in local_map.temporary_boundaries:
                polyline = np.asarray(lane.polyline_world, dtype=np.float32)
                if len(polyline) < 2:
                    continue
                for waypoint_xy in samples.world_points_xy:
                    min_boundary_clearance_m = min(
                        min_boundary_clearance_m,
                        self._distance_to_polyline(waypoint_xy, polyline[:, :2]),
                    )

        if not np.isfinite(min_clearance_m):
            min_clearance_m = 20.0
        if not np.isfinite(min_boundary_clearance_m):
            min_boundary_clearance_m = 20.0
        return _DynamicClearance(
            min_clearance_m=float(min_clearance_m),
            min_boundary_clearance_m=float(min_boundary_clearance_m),
        )

    def _prediction_safety_map(
        self,
        local_map: LocalMap,
        predictions: list[AgentPrediction],
    ) -> dict[int, float]:
        detections_by_track = {int(agent.track_id): agent for agent in local_map.dynamic_agents}
        safety_by_track: dict[int, float] = {}
        ego_radius = self._ego_radius_m()
        for track_id, detection in detections_by_track.items():
            safety_by_track[track_id] = (
                ego_radius
                + self._agent_radius_m(np.asarray(detection.world_bbox_3d, dtype=np.float32))
                + _BASE_SAFETY_BUFFER_M
            )
        for prediction in predictions:
            covariance_padding = 0.0
            if prediction.covariance_by_step:
                cov = np.asarray(prediction.covariance_by_step[0], dtype=np.float32)
                covariance_padding = float(np.sqrt(max(float(cov[0, 0]), float(cov[1, 1]))))
            safety_by_track[int(prediction.track_id)] = max(
                safety_by_track.get(int(prediction.track_id), ego_radius + 1.2 + _BASE_SAFETY_BUFFER_M),
                ego_radius + 1.2 + _BASE_SAFETY_BUFFER_M + covariance_padding,
            )
        return safety_by_track

    def _prepare_dynamic_obstacles(
        self,
        local_map: LocalMap,
        predictions: list[AgentPrediction],
        safety_margins: dict[int, float],
    ) -> list[list[tuple[int, np.ndarray, float]]]:
        static_obstacles: list[tuple[int, np.ndarray, float]] = []
        for detection in local_map.dynamic_agents:
            track_id = int(detection.track_id)
            centroid_xy = np.mean(np.asarray(detection.world_bbox_3d, dtype=np.float32)[:, :2], axis=0).astype(np.float32)
            static_obstacles.append(
                (
                    track_id,
                    centroid_xy,
                    float(safety_margins.get(track_id, self._ego_radius_m() + 1.2)),
                )
            )

        obstacles_by_step: list[list[tuple[int, np.ndarray, float]]] = []
        for index in range(self.horizon_steps):
            yielded_tracks: set[int] = set()
            step_obstacles: list[tuple[int, np.ndarray, float]] = []
            for prediction in predictions:
                if not prediction.predicted_trajectory:
                    continue
                sample_index = min(index, len(prediction.predicted_trajectory) - 1)
                predicted_waypoint = prediction.predicted_trajectory[sample_index]
                track_id = int(prediction.track_id)
                yielded_tracks.add(track_id)
                step_obstacles.append(
                    (
                        track_id,
                        np.array([predicted_waypoint.x, predicted_waypoint.y], dtype=np.float32),
                        float(safety_margins.get(track_id, self._ego_radius_m() + 1.2)),
                    )
                )
            for track_id, centroid_xy, safety_margin_m in static_obstacles:
                if track_id in yielded_tracks:
                    continue
                step_obstacles.append((track_id, centroid_xy, safety_margin_m))
            obstacles_by_step.append(step_obstacles)
        return obstacles_by_step

    def _synthesize_stop_candidate(
        self,
        *,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
        reference_path: _ReferencePath,
        run_context: _PlannerRunContext,
    ) -> PlannerCandidate | None:
        stop_progress_m = self._behavior_stop_progress(
            local_map,
            ego_pose,
            behavior_state,
            self._reference_lane(local_map, ego_pose),
        )
        stop_target = _MotionTarget(
            target_lane_id=reference_path.lane_id,
            target_d_m=0.0,
            target_speed_mps=0.0,
            terminal_time_s=max(self._effective_terminal_times()),
            target_s_m=None if stop_progress_m is None else float(ego_pose.frenet_s + stop_progress_m),
        )
        candidate = self._evaluate_candidate(
            local_map=local_map,
            ego_pose=ego_pose,
            predictions=predictions,
            behavior_state=behavior_state,
            reference_path=reference_path,
            target=stop_target,
            run_context=run_context,
        )
        if candidate.feasible:
            return candidate
        return None

    def _stop_distance(self, local_map: LocalMap, behavior_state: BehaviorState) -> float | None:
        if behavior_state not in {BehaviorState.INTERSECTION_APPROACH, BehaviorState.STOPPING_FOR_RED}:
            return None
        red_like_states = {TrafficLightState.RED, TrafficLightState.AMBER}
        candidates = [
            light.stop_line_distance_m
            for light in local_map.traffic_signal_states
            if light.state in red_like_states
        ]
        if not candidates:
            return None
        return max(min(candidates) - 3.0, 0.0)

    def _adjacent_lane(self, local_map: LocalMap, lane_id: str, *, closed_lanes: set[str]):
        parsed = parse_lane_id(lane_id)
        if parsed is None:
            return None
        road_id, section_id, lane_index = parsed
        candidates = []
        for segment in local_map.static_lanes:
            if segment.lane_id in closed_lanes:
                continue
            parts = parse_lane_id(segment.lane_id)
            if parts is None:
                continue
            if parts[0] == road_id and parts[1] == section_id and abs(parts[2] - lane_index) == 1:
                candidates.append(segment)
        if not candidates:
            return None
        return min(candidates, key=lambda segment: abs(parse_lane_id(segment.lane_id)[2] - lane_index))

    def _reference_lane_half_width(self, reference_path: _ReferencePath) -> float:
        if reference_path.left_boundary_world is None or reference_path.right_boundary_world is None:
            return 1.75
        left = np.asarray(reference_path.left_boundary_world, dtype=np.float32)
        right = np.asarray(reference_path.right_boundary_world, dtype=np.float32)
        if len(left) == 0 or len(right) == 0:
            return 1.75
        return max(float(np.mean(np.linalg.norm(left[:, :2] - right[:, :2], axis=1)) * 0.5), 1.5)

    def _ego_radius_m(self) -> float:
        return float(np.hypot(_EGO_LENGTH_M, _EGO_WIDTH_M)) * 0.5

    def _agent_radius_m(self, world_bbox_3d: np.ndarray) -> float:
        if world_bbox_3d.shape != (8, 3):
            return 1.2
        size_xy = np.max(world_bbox_3d[:, :2], axis=0) - np.min(world_bbox_3d[:, :2], axis=0)
        return float(np.hypot(float(size_xy[0]), float(size_xy[1]))) * 0.5

    def _distance_to_polyline(self, point_xy: np.ndarray, polyline_xy: np.ndarray) -> float:
        if len(polyline_xy) < 2:
            return float("inf")
        best_distance = float("inf")
        for start, end in zip(polyline_xy[:-1], polyline_xy[1:], strict=False):
            segment = end - start
            segment_length_sq = float(np.dot(segment, segment))
            if segment_length_sq <= 1e-6:
                best_distance = min(best_distance, float(distance_xy(point_xy, start)))
                continue
            t = float(np.clip(np.dot(point_xy - start, segment) / segment_length_sq, 0.0, 1.0))
            projection = start + (segment * t)
            best_distance = min(best_distance, float(distance_xy(point_xy, projection)))
        return float(best_distance)


StanleyPidController = RouteFollowerController
