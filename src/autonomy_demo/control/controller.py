from __future__ import annotations

import math

from autonomy_demo.common.geometry import clamp, normalize_angle, signed_lateral_error
from autonomy_demo.interfaces.enums import BehaviorState
from autonomy_demo.interfaces.types import ControlCommand, EgoPose, EgoTrajectory


class StubController:
    """TODO(PRD 3.2.8): replace with Stanley + PID + emergency brake override."""

    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        target_speed = trajectory.waypoints[0].velocity if trajectory.waypoints else 0.0
        speed_error = max(target_speed - ego_pose.speed_mps, 0.0)
        throttle = min(speed_error / 10.0, 1.0)
        return ControlCommand(throttle=throttle, steer=0.0, brake=0.0, emergency_override=False)


class RouteFollowerController:
    """TODO(PRD 3.2.8): replace this simple lane follower with the full Stanley + PID stack."""

    def __init__(
        self,
        *,
        heading_gain: float = 1.4,
        lateral_gain: float = 2.0,
        speed_gain: float = 0.25,
        brake_gain: float = 0.4,
    ) -> None:
        self.heading_gain = heading_gain
        self.lateral_gain = lateral_gain
        self.speed_gain = speed_gain
        self.brake_gain = brake_gain

    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        if not trajectory.waypoints:
            return ControlCommand(throttle=0.0, steer=0.0, brake=1.0, emergency_override=True)

        target_index = min(2, len(trajectory.waypoints) - 1)
        target_waypoint = trajectory.waypoints[target_index]
        target_speed = max(waypoint.velocity for waypoint in trajectory.waypoints)
        heading_error = normalize_angle(target_waypoint.yaw - ego_pose.yaw_rad)
        lateral_error = signed_lateral_error(
            origin_x=float(ego_pose.world_xyz[0]),
            origin_y=float(ego_pose.world_xyz[1]),
            origin_yaw_rad=float(ego_pose.yaw_rad),
            target_x=float(target_waypoint.x),
            target_y=float(target_waypoint.y),
        )
        steering_term = self.heading_gain * heading_error
        cross_track_term = math.atan2(
            self.lateral_gain * lateral_error,
            max(abs(ego_pose.speed_mps), 1.0),
        )
        steer = clamp((steering_term + cross_track_term) / math.pi, -1.0, 1.0)

        if trajectory.behavior_state == BehaviorState.GOAL_REACHED:
            return ControlCommand(
                throttle=0.0,
                steer=steer,
                brake=0.8,
                emergency_override=False,
            )

        speed_error = target_speed - ego_pose.speed_mps
        throttle = clamp(self.speed_gain * speed_error, 0.0, 0.75)
        brake = clamp(-self.brake_gain * speed_error, 0.0, 1.0)
        return ControlCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            emergency_override=False,
        )
