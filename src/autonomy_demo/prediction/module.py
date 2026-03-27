from __future__ import annotations

from autonomy_demo.interfaces.types import AgentPrediction, LocalMap, Waypoint


class StubPredictionModule:
    """TODO(PRD 3.2.6): replace with kinematic and learned forecasting adapters."""

    def run(self, local_map: LocalMap) -> list[AgentPrediction]:
        predictions: list[AgentPrediction] = []
        for agent in local_map.dynamic_agents:
            waypoints = [
                Waypoint(
                    x=float(agent.world_bbox_3d[0][0] + step * 0.5),
                    y=float(agent.world_bbox_3d[0][1]),
                    yaw=0.0,
                    velocity=float(agent.velocity[0]),
                    timestamp=step * 0.1,
                )
                for step in range(5)
            ]
            predictions.append(
                AgentPrediction(
                    track_id=agent.track_id,
                    object_class=agent.object_class,
                    predicted_trajectory=waypoints,
                    confidence_by_step=[0.75] * len(waypoints),
                )
            )
        return predictions

