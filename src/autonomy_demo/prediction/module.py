from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import AgentPrediction, LocalMap, Waypoint
from autonomy_demo.mapping.lane_graph import project_point_to_centerline, sample_centerline_at_s


class LaneAwarePredictionModule:
    """Lightweight lane-aware kinematic prediction with constant-velocity fallback."""

    def prepare(self, simulation, scenario) -> None:
        return None

    def run(self, local_map: LocalMap) -> list[AgentPrediction]:
        predictions: list[AgentPrediction] = []
        for agent in local_map.dynamic_agents:
            predictions.append(
                AgentPrediction(
                    track_id=agent.track_id,
                    object_class=agent.object_class,
                    predicted_trajectory=self._predict_agent(agent, local_map),
                    confidence_by_step=[0.75] * 5,
                )
            )
        return predictions

    def _predict_agent(self, agent, local_map: LocalMap) -> list[Waypoint]:
        center_xyz = np.mean(np.asarray(agent.world_bbox_3d, dtype=np.float32), axis=0)
        velocity = np.asarray(agent.velocity, dtype=np.float32)
        speed_mps = float(np.linalg.norm(velocity[:2]))
        if speed_mps <= 0.1:
            speed_mps = 2.0

        lane_match = self._nearest_lane(local_map.static_lanes, center_xyz)
        if lane_match is None:
            return [
                Waypoint(
                    x=float(center_xyz[0] + velocity[0] * step * 0.5),
                    y=float(center_xyz[1] + velocity[1] * step * 0.5),
                    yaw=float(np.arctan2(float(velocity[1]), float(velocity[0] + 1e-6))),
                    velocity=float(speed_mps),
                    timestamp=step * 0.1,
                )
                for step in range(5)
            ]

        projection = project_point_to_centerline(lane_match.centerline_world, center_xyz)
        return [
            self._lane_waypoint(
                lane_match.centerline_world,
                s_m=projection.s + (speed_mps * step * 0.5),
                speed_mps=speed_mps,
                timestamp=step * 0.1,
            )
            for step in range(5)
        ]

    def _nearest_lane(self, static_lanes, world_xyz: np.ndarray):
        if not static_lanes:
            return None
        return min(
            static_lanes,
            key=lambda lane: project_point_to_centerline(
                np.asarray(lane.centerline_world, dtype=np.float32),
                world_xyz,
            ).distance_m,
        )

    def _lane_waypoint(
        self,
        centerline_world: np.ndarray,
        *,
        s_m: float,
        speed_mps: float,
        timestamp: float,
    ) -> Waypoint:
        point_xyz, heading_rad = sample_centerline_at_s(centerline_world, s_m)
        return Waypoint(
            x=float(point_xyz[0]),
            y=float(point_xyz[1]),
            yaw=float(heading_rad),
            velocity=float(speed_mps),
            timestamp=float(timestamp),
        )


def build_prediction_module(runtime_config):
    return LaneAwarePredictionModule()


StubPredictionModule = LaneAwarePredictionModule
