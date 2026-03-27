from __future__ import annotations

import math

import numpy as np

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.common.geometry import distance_xy, distance_xyz, normalize_angle
from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import BehaviorState
from autonomy_demo.interfaces.types import (
    AgentPrediction,
    EgoPose,
    EgoTrajectory,
    LocalMap,
    RoutePlan,
    RouteWaypoint,
    ScenarioConfig,
    Waypoint,
)


def route_progress_distance(route_plan: RoutePlan, world_xyz: np.ndarray) -> float:
    if not route_plan.waypoints:
        return 0.0
    nearest = min(
        route_plan.waypoints,
        key=lambda waypoint: distance_xyz(
            np.array([waypoint.x, waypoint.y, waypoint.z], dtype=np.float32), world_xyz
        ),
    )
    return float(nearest.cumulative_distance_m)


def build_route_plan_for_carla(
    simulation,
    scenario: ScenarioConfig,
    *,
    step_m: float = 4.0,
    target_speed_mps: float = 12.0,
    goal_tolerance_m: float = 6.0,
    max_steps: int = 400,
) -> RoutePlan:
    state = getattr(simulation, "state", None)
    world = getattr(state, "world", None)
    carla = getattr(state, "carla", None)
    ego_actor = getattr(state, "ego_actor", None)
    if world is None or carla is None or ego_actor is None:
        raise CarlaRuntimeError("CARLA route generation requires an initialized world and ego actor.")

    carla_map = world.get_map()
    def _waypoint_position_key(waypoint) -> tuple[float, float, float]:
        location = waypoint.transform.location
        return (round(float(location.x), 1), round(float(location.y), 1), round(float(location.z), 1))

    start_waypoint = carla_map.get_waypoint(
        ego_actor.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    goal_waypoint = carla_map.get_waypoint(
        carla.Location(
            x=float(scenario.ego_goal.x),
            y=float(scenario.ego_goal.y),
            z=float(scenario.ego_goal.z),
        ),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if start_waypoint is None or goal_waypoint is None:
        raise CarlaRuntimeError("Unable to resolve a drivable CARLA route for the ego vehicle.")

    waypoints = [start_waypoint]
    visited: set[tuple[float, float, float]] = {_waypoint_position_key(start_waypoint)}
    current = start_waypoint
    goal_location = goal_waypoint.transform.location

    for _ in range(max_steps):
        if distance_xyz(
            np.array(
                [current.transform.location.x, current.transform.location.y, current.transform.location.z],
                dtype=np.float32,
            ),
            np.array([goal_location.x, goal_location.y, goal_location.z], dtype=np.float32),
        ) <= goal_tolerance_m:
            break
        candidates = list(current.next(step_m))
        if not candidates:
            break
        same_lane = [
            waypoint
            for waypoint in candidates
            if waypoint.road_id == current.road_id
            and waypoint.section_id == current.section_id
            and waypoint.lane_id == current.lane_id
        ]
        preferred = same_lane or candidates
        next_waypoint = min(
            preferred,
            key=lambda waypoint: (
                distance_xy(
                    np.array(
                        [waypoint.transform.location.x, waypoint.transform.location.y],
                        dtype=np.float32,
                    ),
                    np.array([goal_location.x, goal_location.y], dtype=np.float32),
                ),
                abs(waypoint.lane_id - goal_waypoint.lane_id),
            ),
        )
        waypoint_key = _waypoint_position_key(next_waypoint)
        if waypoint_key in visited and distance_xy(
            np.array(
                [next_waypoint.transform.location.x, next_waypoint.transform.location.y],
                dtype=np.float32,
            ),
            np.array([goal_location.x, goal_location.y], dtype=np.float32),
        ) >= distance_xy(
            np.array([current.transform.location.x, current.transform.location.y], dtype=np.float32),
            np.array([goal_location.x, goal_location.y], dtype=np.float32),
        ):
            break
        visited.add(waypoint_key)
        waypoints.append(next_waypoint)
        current = next_waypoint

    if distance_xy(
        np.array([waypoints[-1].transform.location.x, waypoints[-1].transform.location.y], dtype=np.float32),
        np.array([goal_location.x, goal_location.y], dtype=np.float32),
    ) > 1.0:
        waypoints.append(goal_waypoint)

    route_waypoints: list[RouteWaypoint] = []
    cumulative_distance_m = 0.0
    previous_location = None
    for waypoint in waypoints:
        transform = waypoint.transform
        location = transform.location
        if previous_location is not None:
            cumulative_distance_m += distance_xyz(
                np.array([previous_location.x, previous_location.y, previous_location.z], dtype=np.float32),
                np.array([location.x, location.y, location.z], dtype=np.float32),
            )
        route_waypoints.append(
            RouteWaypoint(
                x=float(location.x),
                y=float(location.y),
                z=float(location.z),
                yaw=math.radians(float(transform.rotation.yaw)),
                cumulative_distance_m=float(cumulative_distance_m),
                target_speed_mps=float(target_speed_mps),
            )
        )
        previous_location = location

    return RoutePlan(
        waypoints=route_waypoints,
        goal_xyz=np.array([goal_location.x, goal_location.y, goal_location.z], dtype=np.float32),
        total_distance_m=float(cumulative_distance_m),
        goal_tolerance_m=float(goal_tolerance_m),
    )


class RouteFollowerMotionPlanner:
    """TODO(PRD 3.2.7): upgrade this route follower into the full behavior + Frenet planner stack."""

    def __init__(
        self,
        *,
        target_speed_mps: float = 12.0,
        horizon_waypoints: int = 12,
    ) -> None:
        self.target_speed_mps = target_speed_mps
        self.horizon_waypoints = horizon_waypoints
        self.route_plan: RoutePlan | None = None
        self.logger = get_logger(__name__, planner="route_follower")

    def prepare_route(self, simulation, scenario: ScenarioConfig) -> None:
        self.route_plan = build_route_plan_for_carla(
            simulation,
            scenario,
            target_speed_mps=self.target_speed_mps,
        )
        if self.route_plan.waypoints:
            start = self.route_plan.waypoints[0]
            goal = self.route_plan.goal_xyz
            self.logger.info(
                "Prepared CARLA route with %s waypoints, length %.1f m, start=(%.1f, %.1f), goal=(%.1f, %.1f)",
                len(self.route_plan.waypoints),
                self.route_plan.total_distance_m,
                start.x,
                start.y,
                float(goal[0]),
                float(goal[1]),
            )

    def run(
        self,
        local_map: LocalMap,
        ego_pose: EgoPose,
        predictions: list[AgentPrediction],
        behavior_state: BehaviorState,
    ) -> EgoTrajectory:
        if self.route_plan is None or not self.route_plan.waypoints:
            waypoints = [
                Waypoint(
                    x=float(ego_pose.world_xyz[0] + step * 2.0),
                    y=float(ego_pose.world_xyz[1]),
                    yaw=ego_pose.yaw_rad,
                    velocity=self.target_speed_mps,
                    timestamp=step * 0.1,
                )
                for step in range(5)
            ]
            return EgoTrajectory(waypoints=waypoints, cost=1.0, behavior_state=behavior_state)

        ego_xyz = np.asarray(ego_pose.world_xyz, dtype=np.float32)
        nearest_index = min(
            range(len(self.route_plan.waypoints)),
            key=lambda index: distance_xyz(
                np.array(
                    [
                        self.route_plan.waypoints[index].x,
                        self.route_plan.waypoints[index].y,
                        self.route_plan.waypoints[index].z,
                    ],
                    dtype=np.float32,
                ),
                ego_xyz,
            ),
        )
        goal_distance_m = distance_xyz(ego_xyz, self.route_plan.goal_xyz)
        final_behavior = (
            BehaviorState.GOAL_REACHED
            if goal_distance_m <= self.route_plan.goal_tolerance_m
            else behavior_state
        )
        sampled_route = self.route_plan.waypoints[
            nearest_index : min(len(self.route_plan.waypoints), nearest_index + self.horizon_waypoints)
        ]
        if not sampled_route:
            sampled_route = [self.route_plan.waypoints[-1]]
        base_time_s = 1.0 / 20.0
        trajectory = [
            Waypoint(
                x=waypoint.x,
                y=waypoint.y,
                yaw=waypoint.yaw,
                velocity=0.0 if final_behavior == BehaviorState.GOAL_REACHED else waypoint.target_speed_mps,
                timestamp=index * base_time_s,
            )
            for index, waypoint in enumerate(sampled_route)
        ]
        if trajectory:
            trajectory[0].yaw = normalize_angle(trajectory[0].yaw)
        return EgoTrajectory(
            waypoints=trajectory,
            cost=max(goal_distance_m / max(self.route_plan.total_distance_m, 1.0), 0.0),
            behavior_state=final_behavior,
        )
