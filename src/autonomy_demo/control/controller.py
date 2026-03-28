from __future__ import annotations

import math

import numpy as np

from autonomy_demo.common.geometry import clamp, normalize_angle, signed_lateral_error
from autonomy_demo.interfaces.enums import BehaviorState
from autonomy_demo.interfaces.types import ControlCommand, EgoPose, EgoTrajectory, LocalMap


class StubController:
    """Fallback controller for stub mode."""

    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        target_speed = trajectory.waypoints[0].velocity if trajectory.waypoints else 0.0
        speed_error = max(target_speed - ego_pose.speed_mps, 0.0)
        throttle = min(speed_error / 10.0, 1.0)
        return ControlCommand(throttle=throttle, steer=0.0, brake=0.0, emergency_override=False)


class RouteFollowerController:
    """Stanley-style lateral control with PID longitudinal control and emergency override."""

    def __init__(
        self,
        *,
        stanley_gain: float = 2.2,
        heading_gain: float = 1.1,
        kp_speed: float = 0.32,
        ki_speed: float = 0.04,
        kd_speed: float = 0.03,
        emergency_ttc_s: float = 1.6,
        emergency_gap_m: float = 6.0,
    ) -> None:
        self.stanley_gain = stanley_gain
        self.heading_gain = heading_gain
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.emergency_ttc_s = emergency_ttc_s
        self.emergency_gap_m = emergency_gap_m
        self._integral_speed_error = 0.0
        self._previous_speed_error = 0.0
        self._latest_local_map: LocalMap | None = None
        self._latest_predictions = []

    def set_context(self, local_map: LocalMap, predictions) -> None:
        self._latest_local_map = local_map
        self._latest_predictions = predictions

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
        stanley_term = math.atan2(
            self.stanley_gain * lateral_error,
            max(abs(ego_pose.speed_mps), 1.0),
        )
        steer = clamp(
            (self.heading_gain * heading_error + stanley_term) / math.pi,
            -1.0,
            1.0,
        )

        if trajectory.behavior_state == BehaviorState.GOAL_REACHED:
            command = ControlCommand(
                throttle=0.0,
                steer=steer,
                brake=0.65,
                emergency_override=False,
            )
            return self._apply_emergency_override(command, ego_pose)

        speed_error = float(target_speed - ego_pose.speed_mps)
        self._integral_speed_error = clamp(
            self._integral_speed_error + (speed_error * 0.1),
            -10.0,
            10.0,
        )
        derivative = (speed_error - self._previous_speed_error) / 0.1
        self._previous_speed_error = speed_error
        pid_output = (
            (self.kp_speed * speed_error)
            + (self.ki_speed * self._integral_speed_error)
            + (self.kd_speed * derivative)
        )
        throttle = clamp(pid_output, 0.0, 0.8)
        brake = clamp(-pid_output, 0.0, 1.0)
        if trajectory.behavior_state == BehaviorState.STOPPING_FOR_RED:
            brake = max(brake, 0.45)
            throttle = min(throttle, 0.15)
        elif trajectory.behavior_state == BehaviorState.INTERSECTION_APPROACH:
            throttle = min(throttle, 0.45)
        command = ControlCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            emergency_override=False,
        )
        return self._apply_emergency_override(command, ego_pose)

    def _apply_emergency_override(self, command: ControlCommand, ego_pose: EgoPose) -> ControlCommand:
        risk = self._lead_vehicle_risk(ego_pose)
        if risk is None:
            return command
        gap_m, ttc_s = risk
        if gap_m <= self.emergency_gap_m or ttc_s <= self.emergency_ttc_s:
            return ControlCommand(
                throttle=0.0,
                steer=command.steer,
                brake=max(command.brake, 0.9),
                hand_brake=command.hand_brake,
                reverse=command.reverse,
                emergency_override=True,
            )
        return command

    def _lead_vehicle_risk(self, ego_pose: EgoPose) -> tuple[float, float] | None:
        if not self._latest_predictions:
            return None
        ego_xy = np.asarray(ego_pose.world_xyz, dtype=np.float32)[:2]
        heading_vec = np.array([math.cos(ego_pose.yaw_rad), math.sin(ego_pose.yaw_rad)], dtype=np.float32)
        best: tuple[float, float] | None = None
        for prediction in self._latest_predictions:
            if not prediction.predicted_trajectory:
                continue
            predicted = prediction.predicted_trajectory[0]
            target_xy = np.array([predicted.x, predicted.y], dtype=np.float32)
            delta = target_xy - ego_xy
            longitudinal_gap = float(np.dot(delta, heading_vec))
            lateral_gap = float(abs((-heading_vec[1] * delta[0]) + (heading_vec[0] * delta[1])))
            if longitudinal_gap <= 0.0 or lateral_gap > 3.0:
                continue
            lead_speed = float(predicted.velocity)
            closing_speed = max(ego_pose.speed_mps - lead_speed, 0.1)
            ttc_s = longitudinal_gap / closing_speed
            if best is None or longitudinal_gap < best[0]:
                best = (float(longitudinal_gap), float(ttc_s))
        return best


StanleyPidController = RouteFollowerController
