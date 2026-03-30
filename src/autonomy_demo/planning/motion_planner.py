from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autonomy_demo.common.geometry import distance_xy
from autonomy_demo.interfaces.enums import BehaviorState, TrafficLightState
from autonomy_demo.interfaces.types import AgentPrediction, EgoPose, EgoTrajectory, LocalMap, Waypoint
from autonomy_demo.mapping.lane_graph import parse_lane_id, project_point_to_centerline, sample_centerline_at_s
from autonomy_demo.planning.route_following import RouteFollowerMotionPlanner


@dataclass(slots=True)
class PlannerCandidate:
    trajectory: EgoTrajectory
    lane_id: str
    target_speed_mps: float
    score: float = 0.0


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
    """Lane-aware local planner with route-following fallback."""

    def __init__(
        self,
        *,
        horizon_steps: int = 12,
        dt_s: float = 0.2,
        cruise_speed_mps: float = 12.0,
    ) -> None:
        self.horizon_steps = horizon_steps
        self.dt_s = dt_s
        self.cruise_speed_mps = cruise_speed_mps
        self.route_plan = None
        self._fallback = RouteFollowerMotionPlanner(
            target_speed_mps=cruise_speed_mps,
            horizon_waypoints=horizon_steps,
        )

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
        route_guided = self._route_guided_trajectory(local_map, ego_pose, behavior_state)
        if route_guided is not None:
            return route_guided
        reference_lane = self._reference_lane(local_map, ego_pose)
        if reference_lane is None:
            return self._fallback.run(local_map, ego_pose, predictions, behavior_state)

        candidates = self._generate_candidates(local_map, ego_pose, predictions, behavior_state, reference_lane)
        if not candidates:
            return self._fallback.run(local_map, ego_pose, predictions, behavior_state)
        best = min(candidates, key=lambda candidate: candidate.score)
        return best.trajectory

    def _route_guided_trajectory(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
    ) -> EgoTrajectory | None:
        if self.route_plan is None or behavior_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING}:
            return None
        base = self._fallback.run(local_map, ego_pose, [], behavior_state)
        if not base.waypoints:
            return None
        target_speed_mps = self._speed_profiles(ego_pose, behavior_state)[0]
        stop_distance_m = self._stop_distance(local_map, behavior_state)
        traveled_m = 0.0
        waypoints: list[Waypoint] = []
        for index, waypoint in enumerate(base.waypoints):
            commanded_speed = target_speed_mps
            if stop_distance_m is not None:
                remaining = max(stop_distance_m - traveled_m, 0.0)
                commanded_speed = min(
                    commanded_speed,
                    max(remaining / max(self.dt_s * max(len(base.waypoints) - index, 1), 1e-3), 0.0),
                )
            waypoints.append(
                Waypoint(
                    x=float(waypoint.x),
                    y=float(waypoint.y),
                    yaw=float(waypoint.yaw),
                    velocity=float(commanded_speed),
                    timestamp=float(index * self.dt_s),
                )
            )
            traveled_m += float(commanded_speed * self.dt_s)
        return EgoTrajectory(
            waypoints=waypoints,
            cost=float(base.cost),
            behavior_state=base.behavior_state,
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

    def _generate_candidates(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
        reference_lane,
    ) -> list[PlannerCandidate]:
        lanes_to_try = [reference_lane]
        if behavior_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING}:
            adjacent = self._adjacent_lane(local_map, reference_lane.lane_id, closed_lanes=set(local_map.closed_lanes))
            if adjacent is not None:
                lanes_to_try.append(adjacent)

        stop_distance = self._stop_distance(local_map, behavior_state)
        speed_profiles = self._speed_profiles(ego_pose, behavior_state)
        candidates: list[PlannerCandidate] = []
        for lane in lanes_to_try:
            for target_speed in speed_profiles:
                trajectory = self._trajectory_for_lane(
                    current_lane=reference_lane,
                    target_lane=lane,
                    ego_pose=ego_pose,
                    behavior_state=behavior_state,
                    target_speed_mps=target_speed,
                    stop_distance_m=stop_distance,
                )
                score = self._score_candidate(
                    trajectory=trajectory,
                    local_map=local_map,
                    predictions=predictions,
                    target_lane_id=lane.lane_id,
                    reference_lane_id=reference_lane.lane_id,
                    behavior_state=behavior_state,
                    stop_distance_m=stop_distance,
                )
                candidates.append(
                    PlannerCandidate(
                        trajectory=trajectory,
                        lane_id=lane.lane_id,
                        target_speed_mps=target_speed,
                        score=score,
                    )
                )
        return candidates

    def _speed_profiles(self, ego_pose: EgoPose, behavior_state: BehaviorState) -> list[float]:
        if behavior_state == BehaviorState.GOAL_REACHED:
            return [0.0]
        if behavior_state == BehaviorState.STOPPING_FOR_RED:
            return [0.0, max(ego_pose.speed_mps * 0.35, 1.0)]
        if behavior_state == BehaviorState.INTERSECTION_APPROACH:
            return [max(self.cruise_speed_mps * 0.5, 4.0), max(ego_pose.speed_mps * 0.7, 4.0)]
        if behavior_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING}:
            return [max(self.cruise_speed_mps * 0.85, 6.0), max(ego_pose.speed_mps, 5.0)]
        return [self.cruise_speed_mps, max(ego_pose.speed_mps, self.cruise_speed_mps * 0.8)]

    def _trajectory_for_lane(
        self,
        *,
        current_lane,
        target_lane,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
        target_speed_mps: float,
        stop_distance_m: float | None,
    ) -> EgoTrajectory:
        current_projection = project_point_to_centerline(current_lane.centerline_world, ego_pose.world_xyz)
        target_projection = project_point_to_centerline(target_lane.centerline_world, ego_pose.world_xyz)
        if target_lane.lane_id == current_lane.lane_id:
            merge_alpha_max = 0.0
        elif behavior_state == BehaviorState.MERGING:
            merge_alpha_max = 1.0
        else:
            merge_alpha_max = 0.75
        waypoints: list[Waypoint] = []
        accumulated_s = current_projection.s
        for step in range(self.horizon_steps):
            time_s = step * self.dt_s
            blended_speed = target_speed_mps
            if stop_distance_m is not None:
                remaining = max(stop_distance_m - max(accumulated_s - current_projection.s, 0.0), 0.0)
                blended_speed = min(blended_speed, max(remaining / max(self.dt_s * max(self.horizon_steps - step, 1), 1e-3), 0.0))
            accumulated_s += blended_speed * self.dt_s
            current_point, current_heading = sample_centerline_at_s(current_lane.centerline_world, accumulated_s)
            target_point, target_heading = sample_centerline_at_s(
                target_lane.centerline_world,
                max(target_projection.s, 0.0) + max(accumulated_s - current_projection.s, 0.0),
            )
            if merge_alpha_max > 0.0:
                alpha = min(merge_alpha_max, (step + 1) / max(self.horizon_steps + 2, 1))
                point = current_point + ((target_point - current_point) * alpha)
                heading = current_heading + ((target_heading - current_heading) * alpha)
            else:
                point = current_point
                heading = current_heading
            waypoints.append(
                Waypoint(
                    x=float(point[0]),
                    y=float(point[1]),
                    yaw=float(heading),
                    velocity=float(blended_speed),
                    timestamp=float(time_s),
                )
            )
        return EgoTrajectory(
            waypoints=waypoints,
            cost=0.0,
            behavior_state=behavior_state,
        )

    def _score_candidate(
        self,
        *,
        trajectory: EgoTrajectory,
        local_map: LocalMap,
        predictions: list[AgentPrediction],
        target_lane_id: str,
        reference_lane_id: str,
        behavior_state: BehaviorState,
        stop_distance_m: float | None,
    ) -> float:
        if not trajectory.waypoints:
            return float("inf")
        target_lane = next((lane for lane in local_map.static_lanes if lane.lane_id == target_lane_id), None)
        if target_lane is None:
            return float("inf")

        score = 0.0
        if target_lane_id in local_map.closed_lanes:
            score += 1000.0
        if behavior_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING} and target_lane_id == reference_lane_id:
            score += 20.0

        lane_offset_sum = 0.0
        progress_reward = 0.0
        min_clearance = float("inf")
        previous_yaw = trajectory.waypoints[0].yaw
        smoothness_penalty = 0.0
        for index, waypoint in enumerate(trajectory.waypoints):
            projection = project_point_to_centerline(
                target_lane.centerline_world,
                np.array([waypoint.x, waypoint.y, 0.0], dtype=np.float32),
            )
            lane_offset_sum += abs(projection.d)
            progress_reward += projection.s
            if index:
                smoothness_penalty += abs(waypoint.yaw - previous_yaw)
            previous_yaw = waypoint.yaw
            for prediction in predictions:
                if index >= len(prediction.predicted_trajectory):
                    continue
                predicted_waypoint = prediction.predicted_trajectory[index]
                clearance = distance_xy(
                    np.array([waypoint.x, waypoint.y], dtype=np.float32),
                    np.array([predicted_waypoint.x, predicted_waypoint.y], dtype=np.float32),
                )
                min_clearance = min(min_clearance, clearance)

        score += lane_offset_sum * 5.0
        score += smoothness_penalty * 2.0
        score -= progress_reward * 0.05
        if min_clearance < 4.0:
            score += (4.0 - min_clearance) * 250.0
        elif min_clearance < 8.0:
            score += (8.0 - min_clearance) * 20.0

        if stop_distance_m is not None:
            projected_progress = sum(waypoint.velocity for waypoint in trajectory.waypoints) * self.dt_s
            if projected_progress > stop_distance_m:
                score += (projected_progress - stop_distance_m) * 50.0
            if behavior_state == BehaviorState.STOPPING_FOR_RED:
                score += abs(trajectory.waypoints[-1].velocity) * 20.0

        trajectory.cost = float(max(score, 0.0))
        return trajectory.cost

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
