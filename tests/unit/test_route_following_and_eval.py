from __future__ import annotations

import numpy as np

from autonomy_demo.control.controller import RouteFollowerController
from autonomy_demo.eval.harness import LiveEvaluationHarness
from autonomy_demo.interfaces.enums import BehaviorState, TopicName
from autonomy_demo.interfaces.types import (
    EgoPose,
    Point2D,
    Pose2D,
    RoutePlan,
    RouteWaypoint,
    ScenarioConfig,
    ScenarioEvalCriteria,
)
from autonomy_demo.planning.route_following import RouteFollowerMotionPlanner


def _route_plan() -> RoutePlan:
    return RoutePlan(
        waypoints=[
            RouteWaypoint(x=0.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=0.0, target_speed_mps=8.0),
            RouteWaypoint(x=5.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=5.0, target_speed_mps=8.0),
            RouteWaypoint(x=10.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=10.0, target_speed_mps=8.0),
            RouteWaypoint(x=15.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=15.0, target_speed_mps=8.0),
        ],
        goal_xyz=np.array([15.0, 0.0, 0.0], dtype=np.float32),
        total_distance_m=15.0,
        goal_tolerance_m=1.5,
    )


def test_route_follower_outputs_forward_horizon() -> None:
    planner = RouteFollowerMotionPlanner()
    planner.route_plan = _route_plan()
    ego_pose = EgoPose(
        world_xyz=np.array([4.5, 0.2, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=4.0,
        acceleration_mps2=0.0,
        current_lane_id="lane",
        frenet_s=4.5,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    trajectory = planner.run(local_map=None, ego_pose=ego_pose, predictions=[], behavior_state=BehaviorState.LANE_KEEP)
    assert trajectory.waypoints
    assert trajectory.waypoints[0].x >= 5.0
    assert trajectory.behavior_state == BehaviorState.LANE_KEEP


def test_route_controller_outputs_sane_command_ranges() -> None:
    planner = RouteFollowerMotionPlanner()
    planner.route_plan = _route_plan()
    ego_pose = EgoPose(
        world_xyz=np.array([0.0, 0.5, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=2.0,
        acceleration_mps2=0.0,
        current_lane_id="lane",
        frenet_s=0.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    trajectory = planner.run(local_map=None, ego_pose=ego_pose, predictions=[], behavior_state=BehaviorState.LANE_KEEP)
    command = RouteFollowerController().run(trajectory, ego_pose)
    assert 0.0 <= command.throttle <= 1.0
    assert -1.0 <= command.steer <= 1.0
    assert 0.0 <= command.brake <= 1.0


def test_live_evaluation_reports_distance_and_goal_reach() -> None:
    scenario = ScenarioConfig(
        scenario_id="SC-01",
        name="Highway Cruise",
        map_name="Town04",
        ego_spawn=Pose2D(x=0.0, y=0.0, z=0.0, yaw=0.0),
        ego_goal=Point2D(x=15.0, y=0.0, z=0.0),
        max_duration_s=10.0,
        npcs=[],
        props=[],
        triggers=[],
        eval=ScenarioEvalCriteria(min_completion_rate=0.8, max_collisions=0),
    )
    backend = type("Backend", (), {"state": type("State", (), {"collision_events": []})()})()
    harness = LiveEvaluationHarness(scenario, tick_hz=20, backend=backend)
    harness.set_route_plan(_route_plan())
    harness.update(
        0,
        {
            TopicName.LOCALIZATION_EGO_POSE.value: EgoPose(
                world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                yaw_rad=0.0,
                speed_mps=3.0,
                acceleration_mps2=0.0,
                current_lane_id="lane",
                frenet_s=0.0,
                frenet_d=0.0,
                heading_error_rad=0.0,
            )
        },
    )
    harness.update(
        1,
        {
            TopicName.LOCALIZATION_EGO_POSE.value: EgoPose(
                world_xyz=np.array([15.0, 0.0, 0.0], dtype=np.float32),
                yaw_rad=0.0,
                speed_mps=4.0,
                acceleration_mps2=0.0,
                current_lane_id="lane",
                frenet_s=15.0,
                frenet_d=0.0,
                heading_error_rad=0.0,
            )
        },
    )
    summary = harness.finalize()
    assert summary.distance_traveled_m > 0.0
    assert summary.goal_reached is True
    assert summary.completion_rate == 1.0
    assert any("Localization valid lane ratio" in note for note in summary.notes)
