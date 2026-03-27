from __future__ import annotations

from autonomy_demo.interfaces.types import ControlCommand, EgoPose, EgoTrajectory


class StubController:
    """TODO(PRD 3.2.8): replace with Stanley + PID + emergency brake override."""

    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        target_speed = trajectory.waypoints[0].velocity if trajectory.waypoints else 0.0
        speed_error = max(target_speed - ego_pose.speed_mps, 0.0)
        throttle = min(speed_error / 10.0, 1.0)
        return ControlCommand(throttle=throttle, steer=0.0, brake=0.0, emergency_override=False)

