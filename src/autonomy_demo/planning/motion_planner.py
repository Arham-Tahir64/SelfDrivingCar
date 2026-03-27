from __future__ import annotations

from autonomy_demo.interfaces.enums import BehaviorState
from autonomy_demo.interfaces.types import AgentPrediction, EgoPose, EgoTrajectory, LocalMap, Waypoint


class StubMotionPlanner:
    """TODO(PRD 3.2.7): replace with Frenet candidate generation and cost evaluation."""

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
