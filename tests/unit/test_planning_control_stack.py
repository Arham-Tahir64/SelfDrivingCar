from __future__ import annotations

import numpy as np

from autonomy_demo.control.controller import RouteFollowerController
from autonomy_demo.interfaces.enums import BehaviorState, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    AgentPrediction,
    EgoPose,
    EgoTrajectory,
    LocalMap,
    ObjectDetection,
    RoutePlan,
    RouteWaypoint,
    StaticLaneSegment,
    TrafficLightDetection,
    Waypoint,
)
from autonomy_demo.planning.behavior_fsm import RuleBasedBehaviorPlanner
from autonomy_demo.planning.motion_planner import FrenetMotionPlanner, _solve_quartic, _solve_quintic


def _lane(lane_id: str, y_offset: float = 0.0) -> StaticLaneSegment:
    return StaticLaneSegment(
        lane_id=lane_id,
        centerline_world=np.array(
            [[0.0, y_offset, 0.0], [15.0, y_offset, 0.0], [30.0, y_offset, 0.0]],
            dtype=np.float32,
        ),
        left_boundary_world=np.array(
            [[0.0, y_offset + 1.75, 0.0], [15.0, y_offset + 1.75, 0.0], [30.0, y_offset + 1.75, 0.0]],
            dtype=np.float32,
        ),
        right_boundary_world=np.array(
            [[0.0, y_offset - 1.75, 0.0], [15.0, y_offset - 1.75, 0.0], [30.0, y_offset - 1.75, 0.0]],
            dtype=np.float32,
        ),
        speed_limit_mps=20.0,
    )


def _ego_pose() -> EgoPose:
    return EgoPose(
        world_xyz=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_1",
        frenet_s=2.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )


def _trajectory(
    *,
    speed: float = 8.0,
    behavior_state: BehaviorState = BehaviorState.LANE_KEEP,
    y_offset: float = 0.0,
    steps: int = 6,
    dt_s: float = 0.1,
) -> EgoTrajectory:
    return EgoTrajectory(
        waypoints=[
            Waypoint(
                x=2.0 + (step * speed * dt_s),
                y=y_offset,
                yaw=0.0,
                velocity=speed,
                timestamp=step * dt_s,
            )
            for step in range(steps)
        ],
        cost=0.0,
        behavior_state=behavior_state,
    )


def _lead_detection(
    x: float,
    y: float = 0.0,
    speed: float = 1.0,
    *,
    confidence: float = 0.95,
    source_modality: str = "bootstrap",
    track_state: TrackState = TrackState.CONFIRMED,
) -> ObjectDetection:
    return ObjectDetection(
        track_id=7,
        object_class=ObjectClass.VEHICLE,
        world_bbox_3d=np.array(
            [
                [x - 1.0, y - 0.5, 0.0],
                [x + 1.0, y - 0.5, 0.0],
                [x + 1.0, y + 0.5, 0.0],
                [x - 1.0, y + 0.5, 0.0],
                [x - 1.0, y - 0.5, 1.5],
                [x + 1.0, y - 0.5, 1.5],
                [x + 1.0, y + 0.5, 1.5],
                [x - 1.0, y + 0.5, 1.5],
            ],
            dtype=np.float32,
        ),
        velocity=np.array([speed, 0.0, 0.0], dtype=np.float32),
        confidence=confidence,
        track_state=track_state,
        image_bbox_xyxy=np.array([430.0, 210.0, 550.0, 360.0], dtype=np.float32),
        source_modality=source_modality,
        source_sensor_ids=["front_camera"] if source_modality in {"camera", "fused"} else [],
    )


def _oversized_detection(track_id: int, x: float, y: float, length: float, width: float, height: float) -> ObjectDetection:
    return ObjectDetection(
        track_id=track_id,
        object_class=ObjectClass.VEHICLE,
        world_bbox_3d=np.array(
            [
                [x - (length / 2.0), y - (width / 2.0), 0.0],
                [x + (length / 2.0), y - (width / 2.0), 0.0],
                [x + (length / 2.0), y + (width / 2.0), 0.0],
                [x - (length / 2.0), y + (width / 2.0), 0.0],
                [x - (length / 2.0), y - (width / 2.0), height],
                [x + (length / 2.0), y - (width / 2.0), height],
                [x + (length / 2.0), y + (width / 2.0), height],
                [x - (length / 2.0), y + (width / 2.0), height],
            ],
            dtype=np.float32,
        ),
        velocity=np.zeros(3, dtype=np.float32),
        confidence=0.95,
        track_state=TrackState.CONFIRMED,
    )


def test_behavior_planner_transitions_for_merge_and_red_and_goal() -> None:
    planner = RuleBasedBehaviorPlanner()
    planner.goal_xyz = np.array([40.0, 0.0, 0.0], dtype=np.float32)
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0), _lane("road_1:section_0:lane_2", 3.5)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=["road_1:section_0:lane_1"],
        traffic_signal_states=[],
        drivable_space=None,
    )
    assert planner.run(local_map, _ego_pose()) == BehaviorState.PREPARE_MERGE

    local_map.closed_lanes = []
    local_map.traffic_signal_states = [
        TrafficLightDetection(
            world_xyz=np.array([20.0, 0.0, 3.0], dtype=np.float32),
            state=TrafficLightState.RED,
            stop_line_distance_m=8.0,
            confidence=1.0,
        )
    ]
    assert planner.run(local_map, _ego_pose()) == BehaviorState.STOPPING_FOR_RED

    goal_pose = _ego_pose()
    goal_pose.world_xyz = np.array([40.0, 0.0, 0.0], dtype=np.float32)
    assert planner.run(local_map, goal_pose) == BehaviorState.GOAL_REACHED


def test_behavior_planner_finishes_merge_and_enters_cooldown() -> None:
    planner = RuleBasedBehaviorPlanner(merge_cooldown_ticks=5)
    planner.goal_xyz = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    closing_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0), _lane("road_1:section_0:lane_2", 3.5)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=["road_1:section_0:lane_1"],
        traffic_signal_states=[],
        drivable_space=None,
    )
    assert planner.run(closing_map, _ego_pose()) == BehaviorState.PREPARE_MERGE

    merged_pose = EgoPose(
        world_xyz=np.array([8.0, 3.5, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_2",
        frenet_s=8.0,
        frenet_d=0.1,
        heading_error_rad=0.0,
    )
    merged_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0), _lane("road_1:section_0:lane_2", 3.5)],
        dynamic_agents=[_lead_detection(13.0, 3.5, speed=1.0)],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=["road_1:section_0:lane_1"],
        traffic_signal_states=[],
        drivable_space=None,
    )
    assert planner.run(merged_map, merged_pose) == BehaviorState.LANE_KEEP


def test_quintic_lateral_polynomial_hits_terminal_conditions() -> None:
    polynomial = _solve_quintic(
        x0=0.0,
        x_dot0=0.0,
        x_ddot0=0.0,
        xT=3.5,
        x_dotT=0.0,
        x_ddotT=0.0,
        T=5.0,
    )
    assert polynomial.value(0.0) == 0.0
    assert abs(polynomial.derivative(0.0, 1)) < 1e-6
    assert abs(polynomial.derivative(0.0, 2)) < 1e-6
    assert abs(polynomial.value(5.0) - 3.5) < 1e-6
    assert abs(polynomial.derivative(5.0, 1)) < 1e-6
    assert abs(polynomial.derivative(5.0, 2)) < 1e-6


def test_quartic_longitudinal_polynomial_is_monotonic_and_hits_terminal_speed() -> None:
    polynomial = _solve_quartic(
        x0=2.0,
        x_dot0=8.0,
        x_ddot0=0.0,
        x_dotT=10.0,
        x_ddotT=0.0,
        T=5.0,
    )
    samples = [polynomial.value(step * 0.1) for step in range(51)]
    speeds = [polynomial.derivative(step * 0.1, 1) for step in range(51)]
    assert samples == sorted(samples)
    assert abs(speeds[-1] - 10.0) < 1e-6


def test_frenet_motion_planner_lane_keep_open_road_produces_dense_horizon() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    lane_keep_trajectory = planner.run(local_map, _ego_pose(), [], BehaviorState.LANE_KEEP)
    assert len(lane_keep_trajectory.waypoints) == 50
    assert abs(lane_keep_trajectory.waypoints[0].y) < 0.2
    assert lane_keep_trajectory.waypoints[-1].x > lane_keep_trajectory.waypoints[0].x
    assert planner.last_candidates
    assert any(candidate.feasible for candidate in planner.last_candidates)


def test_frenet_motion_planner_merge_when_current_lane_closed_prefers_adjacent_lane() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0), _lane("road_1:section_0:lane_2", 3.5)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=["road_1:section_0:lane_1"],
        traffic_signal_states=[],
        drivable_space=None,
    )
    blocked_prediction = AgentPrediction(
        track_id=11,
        object_class=ObjectClass.VEHICLE,
        predicted_trajectory=[
            Waypoint(x=8.0 + (step * 0.8), y=0.0, yaw=0.0, velocity=1.0, timestamp=step * 0.1)
            for step in range(50)
        ],
        confidence_by_step=[0.8] * 50,
    )
    merge_trajectory = planner.run(local_map, _ego_pose(), [blocked_prediction], BehaviorState.PREPARE_MERGE)
    best_candidate = min(
        (candidate for candidate in planner.last_candidates if candidate.feasible),
        key=lambda candidate: candidate.score,
    )
    assert len(merge_trajectory.waypoints) == 50
    assert merge_trajectory.waypoints[-1].y > 2.5
    assert best_candidate.target_lane_id == "road_1:section_0:lane_2"
    assert best_candidate.target_d_m >= 2.625


def test_frenet_motion_planner_prefers_route_guidance_for_non_merge_driving() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    route_plan = RoutePlan(
        waypoints=[
            RouteWaypoint(x=0.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=0.0, target_speed_mps=10.0),
            RouteWaypoint(x=5.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=5.0, target_speed_mps=10.0),
            RouteWaypoint(x=10.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=10.0, target_speed_mps=10.0),
            RouteWaypoint(x=15.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=15.0, target_speed_mps=10.0),
            RouteWaypoint(x=20.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=20.0, target_speed_mps=10.0),
        ],
        goal_xyz=np.array([20.0, 0.0, 0.0], dtype=np.float32),
        total_distance_m=20.0,
        goal_tolerance_m=2.0,
    )
    planner.route_plan = route_plan
    planner._fallback.route_plan = route_plan
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 3.5)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    lane_keep_trajectory = planner.run(local_map, _ego_pose(), [], BehaviorState.LANE_KEEP)
    assert len(lane_keep_trajectory.waypoints) == 50
    assert abs(lane_keep_trajectory.waypoints[0].y) < 0.25


def test_frenet_motion_planner_stops_before_red_light_stop_line() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[
            TrafficLightDetection(
                world_xyz=np.array([18.0, 0.0, 3.0], dtype=np.float32),
                state=TrafficLightState.RED,
                stop_line_distance_m=12.0,
                confidence=1.0,
            )
        ],
        drivable_space=None,
    )
    trajectory = planner.run(local_map, _ego_pose(), [], BehaviorState.STOPPING_FOR_RED)
    assert len(trajectory.waypoints) == 50
    assert trajectory.waypoints[-1].velocity < 0.25
    assert max(waypoint.x for waypoint in trajectory.waypoints) <= 11.25
    assert any(candidate.reject_reason == "stop_line_violation" for candidate in planner.last_candidates)


def test_frenet_motion_planner_pedestrian_yield_prefers_stop_candidate() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    pedestrian = ObjectDetection(
        track_id=44,
        object_class=ObjectClass.PEDESTRIAN,
        world_bbox_3d=np.array(
            [
                [16.0, -0.4, 0.0],
                [16.6, -0.4, 0.0],
                [16.6, 0.4, 0.0],
                [16.0, 0.4, 0.0],
                [16.0, -0.4, 1.8],
                [16.6, -0.4, 1.8],
                [16.6, 0.4, 1.8],
                [16.0, 0.4, 1.8],
            ],
            dtype=np.float32,
        ),
        velocity=np.zeros(3, dtype=np.float32),
        confidence=0.9,
        track_state=TrackState.CONFIRMED,
    )
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[pedestrian],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    trajectory = planner.run(local_map, _ego_pose(), [], BehaviorState.PEDESTRIAN_YIELD)
    best_candidate = min(
        (candidate for candidate in planner.last_candidates if candidate.feasible),
        key=lambda candidate: candidate.score,
    )
    assert len(trajectory.waypoints) == 50
    assert trajectory.waypoints[-1].velocity < 0.25
    assert trajectory.waypoints[-1].x < 13.0
    assert best_candidate.target_speed_mps == 0.0


def test_frenet_motion_planner_hard_rejects_collision_candidates() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    blocked_prediction = AgentPrediction(
        track_id=12,
        object_class=ObjectClass.VEHICLE,
        predicted_trajectory=[
            Waypoint(x=8.0 + (step * 0.4), y=0.0, yaw=0.0, velocity=0.0, timestamp=step * 0.1)
            for step in range(50)
        ],
        confidence_by_step=[0.9] * 50,
    )
    planner.run(local_map, _ego_pose(), [blocked_prediction], BehaviorState.LANE_KEEP)
    assert any(candidate.reject_reason == "dynamic_collision" for candidate in planner.last_candidates)


def test_frenet_motion_planner_falls_back_to_route_follower_when_all_candidates_are_rejected() -> None:
    planner = FrenetMotionPlanner(cruise_speed_mps=10.0)
    route_plan = RoutePlan(
        waypoints=[
            RouteWaypoint(x=0.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=0.0, target_speed_mps=10.0),
            RouteWaypoint(x=5.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=5.0, target_speed_mps=10.0),
            RouteWaypoint(x=10.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=10.0, target_speed_mps=10.0),
            RouteWaypoint(x=15.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=15.0, target_speed_mps=10.0),
            RouteWaypoint(x=20.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=20.0, target_speed_mps=10.0),
        ],
        goal_xyz=np.array([20.0, 0.0, 0.0], dtype=np.float32),
        total_distance_m=20.0,
        goal_tolerance_m=2.0,
    )
    planner.route_plan = route_plan
    planner._fallback.route_plan = route_plan
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    blocked_prediction = AgentPrediction(
        track_id=90,
        object_class=ObjectClass.VEHICLE,
        predicted_trajectory=[
            Waypoint(x=2.5 + (step * 0.2), y=0.0, yaw=0.0, velocity=0.0, timestamp=step * 0.1)
            for step in range(50)
        ],
        confidence_by_step=[1.0] * 50,
    )
    trajectory = planner.run(local_map, _ego_pose(), [blocked_prediction], BehaviorState.LANE_KEEP)
    assert len(trajectory.waypoints) == len(route_plan.waypoints)
    assert trajectory.waypoints[-1].x == route_plan.waypoints[-1].x
    assert all(not candidate.feasible for candidate in planner.last_candidates)


def test_controller_emergency_override_triggers_for_close_lead_prediction() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    prediction = AgentPrediction(
        track_id=21,
        object_class=ObjectClass.VEHICLE,
        predicted_trajectory=[
            Waypoint(x=5.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0),
            Waypoint(x=5.5, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.2),
        ],
        confidence_by_step=[0.9, 0.9],
    )
    controller.set_context(local_map, [prediction])
    trajectory = FrenetMotionPlanner(horizon_steps=4, dt_s=0.2, cruise_speed_mps=8.0).run(
        local_map,
        _ego_pose(),
        [],
        BehaviorState.LANE_KEEP,
    )
    command = controller.run(trajectory, _ego_pose())
    assert command.emergency_override is True
    assert command.brake >= 0.9


def test_controller_ignores_oversized_false_positive_detection_for_emergency_override() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[_oversized_detection(track_id=21, x=5.0, y=0.0, length=18.0, width=6.0, height=6.0)],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    prediction = AgentPrediction(
        track_id=21,
        object_class=ObjectClass.VEHICLE,
        predicted_trajectory=[
            Waypoint(x=5.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0),
            Waypoint(x=5.5, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.2),
        ],
        confidence_by_step=[0.9, 0.9],
    )
    controller.set_context(local_map, [prediction])
    trajectory = FrenetMotionPlanner(horizon_steps=4, dt_s=0.2, cruise_speed_mps=8.0).run(
        local_map,
        _ego_pose(),
        [],
        BehaviorState.LANE_KEEP,
    )
    command = controller.run(trajectory, _ego_pose())
    assert command.emergency_override is False
    assert command.brake < 0.9


def test_controller_emergency_override_accepts_tentative_camera_detection_with_lower_confidence() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[
            _lead_detection(
                5.0,
                0.0,
                speed=0.0,
                confidence=0.3,
                source_modality="camera",
                track_state=TrackState.TENTATIVE,
            )
        ],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    controller.set_context(local_map, [])
    trajectory = FrenetMotionPlanner(horizon_steps=4, dt_s=0.2, cruise_speed_mps=8.0).run(
        local_map,
        _ego_pose(),
        [],
        BehaviorState.LANE_KEEP,
    )
    command = controller.run(trajectory, _ego_pose())
    assert command.emergency_override is True
    assert command.brake >= 0.9


def test_controller_emergency_override_uses_front_camera_image_hazard_when_world_gap_is_weak() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[
            ObjectDetection(
                track_id=12,
                object_class=ObjectClass.VEHICLE,
                world_bbox_3d=np.array(
                    [
                        [18.0, 4.0, 0.0],
                        [20.0, 4.0, 0.0],
                        [20.0, 5.0, 0.0],
                        [18.0, 5.0, 0.0],
                        [18.0, 4.0, 1.5],
                        [20.0, 4.0, 1.5],
                        [20.0, 5.0, 1.5],
                        [18.0, 5.0, 1.5],
                    ],
                    dtype=np.float32,
                ),
                velocity=np.zeros(3, dtype=np.float32),
                confidence=0.35,
                track_state=TrackState.TENTATIVE,
                image_bbox_xyxy=np.array([420.0, 220.0, 560.0, 430.0], dtype=np.float32),
                source_modality="camera",
                source_sensor_ids=["front_camera"],
                position_estimate_kind="camera_projection",
            )
        ],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    controller.set_context(local_map, [])
    trajectory = _trajectory(speed=8.0)
    command = controller.run(trajectory, _ego_pose())
    assert command.emergency_override is True
    assert command.brake >= 0.9


def test_controller_emergency_override_brakes_for_small_front_camera_vehicle_box() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[
            ObjectDetection(
                track_id=13,
                object_class=ObjectClass.VEHICLE,
                world_bbox_3d=np.array(
                    [
                        [22.0, 2.5, 0.0],
                        [24.0, 2.5, 0.0],
                        [24.0, 3.5, 0.0],
                        [22.0, 3.5, 0.0],
                        [22.0, 2.5, 1.5],
                        [24.0, 2.5, 1.5],
                        [24.0, 3.5, 1.5],
                        [22.0, 3.5, 1.5],
                    ],
                    dtype=np.float32,
                ),
                velocity=np.zeros(3, dtype=np.float32),
                confidence=0.28,
                track_state=TrackState.TENTATIVE,
                image_bbox_xyxy=np.array([452.0, 240.0, 540.0, 268.0], dtype=np.float32),
                source_modality="camera",
                source_sensor_ids=["front_camera"],
                position_estimate_kind="camera_projection",
            )
        ],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    controller.set_context(local_map, [])
    trajectory = _trajectory(speed=8.0)
    command = controller.run(trajectory, _ego_pose())
    assert command.brake >= 0.4


def test_controller_launch_assist_clears_tiny_follow_brake_from_standstill() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[_lead_detection(10.0, 0.0, speed=1.0, source_modality="bootstrap")],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    controller.set_context(local_map, [])
    ego_pose = EgoPose(
        world_xyz=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_1",
        frenet_s=2.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    trajectory = _trajectory(speed=4.0)
    command = controller.run(trajectory, ego_pose)
    assert command.emergency_override is False
    assert command.brake == 0.0
    assert command.throttle >= 0.25


def test_controller_comfort_zone_follow_does_not_apply_brake_at_crawl_speed() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0)],
        dynamic_agents=[_lead_detection(10.0, 0.0, speed=1.0, source_modality="bootstrap")],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    controller.set_context(local_map, [])
    ego_pose = EgoPose(
        world_xyz=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=0.7,
        acceleration_mps2=0.0,
        current_lane_id="road_1:section_0:lane_1",
        frenet_s=2.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    trajectory = _trajectory(speed=1.2)
    command = controller.run(trajectory, ego_pose)
    assert command.emergency_override is False
    assert command.brake == 0.0
    assert command.throttle > 0.0


def test_controller_ignores_adjacent_lane_parked_vehicle_for_front_camera_risk() -> None:
    controller = RouteFollowerController()
    local_map = LocalMap(
        static_lanes=[_lane("road_1:section_0:lane_1", 0.0), _lane("road_1:section_0:lane_2", 3.5)],
        dynamic_agents=[_lead_detection(10.0, 3.5, speed=0.0, source_modality="camera")],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        drivable_space=None,
    )
    controller.set_context(local_map, [])
    ego_pose = _ego_pose()
    trajectory = FrenetMotionPlanner(horizon_steps=4, dt_s=0.2, cruise_speed_mps=8.0).run(
        local_map,
        ego_pose,
        [],
        BehaviorState.LANE_KEEP,
    )
    command = controller.run(trajectory, ego_pose)
    assert command.emergency_override is False
    assert command.brake < 0.05
