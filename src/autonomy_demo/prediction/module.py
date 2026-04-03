from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import AgentPrediction, FloatArray, LocalMap, Waypoint
from autonomy_demo.mapping.lane_graph import project_point_to_centerline, sample_centerline_at_s


# Default prediction parameters (overridable via tuning config)
_HORIZON_S = 5.0
_DT_S = 0.1
_NUM_STEPS = int(_HORIZON_S / _DT_S)  # 50 steps

_DEFAULT_LANE_SPEED_MPS = 8.0
_VELOCITY_DECAY_RATE = 0.04

_SIGMA_LAT_BASE = 0.3
_SIGMA_LAT_GROWTH = 0.06
_SIGMA_LON_BASE = 0.5
_SIGMA_LON_GROWTH = 0.12


class LaneAwarePredictionModule:
    """Lane-aware kinematic prediction with 5-second horizon and Gaussian uncertainty."""

    def __init__(
        self,
        *,
        horizon_s: float = _HORIZON_S,
        dt_s: float = _DT_S,
        default_lane_speed_mps: float = _DEFAULT_LANE_SPEED_MPS,
        velocity_decay_rate: float = _VELOCITY_DECAY_RATE,
        sigma_lat_base: float = _SIGMA_LAT_BASE,
        sigma_lat_growth: float = _SIGMA_LAT_GROWTH,
        sigma_lon_base: float = _SIGMA_LON_BASE,
        sigma_lon_growth: float = _SIGMA_LON_GROWTH,
    ) -> None:
        self.horizon_s = horizon_s
        self.dt_s = dt_s
        self.num_steps = int(horizon_s / dt_s)
        self.default_lane_speed_mps = default_lane_speed_mps
        self.velocity_decay_rate = velocity_decay_rate
        self.sigma_lat_base = sigma_lat_base
        self.sigma_lat_growth = sigma_lat_growth
        self.sigma_lon_base = sigma_lon_base
        self.sigma_lon_growth = sigma_lon_growth

    def prepare(self, simulation, scenario) -> None:
        return None

    def run(self, local_map: LocalMap) -> list[AgentPrediction]:
        predictions: list[AgentPrediction] = []
        for agent in local_map.dynamic_agents:
            trajectory, confidence, covariance = self._predict_agent(agent, local_map)
            predictions.append(
                AgentPrediction(
                    track_id=agent.track_id,
                    object_class=agent.object_class,
                    predicted_trajectory=trajectory,
                    confidence_by_step=confidence,
                    covariance_by_step=covariance,
                )
            )
        return predictions

    def _predict_agent(
        self, agent, local_map: LocalMap
    ) -> tuple[list[Waypoint], list[float], list[FloatArray]]:
        center_xyz = np.mean(np.asarray(agent.world_bbox_3d, dtype=np.float32), axis=0)
        velocity = np.asarray(agent.velocity, dtype=np.float32)
        speed_mps = float(np.linalg.norm(velocity[:2]))
        if speed_mps <= 0.1:
            speed_mps = 2.0

        lane_match = self._nearest_lane(local_map.static_lanes, center_xyz)
        lane_speed = (
            float(lane_match.speed_limit_mps)
            if lane_match is not None and lane_match.speed_limit_mps > 0.0
            else self.default_lane_speed_mps
        )

        if lane_match is None:
            return self._predict_constant_velocity(center_xyz, velocity, speed_mps, lane_speed)

        return self._predict_lane_following(
            lane_match, center_xyz, speed_mps, lane_speed
        )

    def _predict_constant_velocity(
        self,
        center_xyz: np.ndarray,
        velocity: np.ndarray,
        speed_mps: float,
        lane_speed: float,
    ) -> tuple[list[Waypoint], list[float], list[FloatArray]]:
        heading = float(np.arctan2(float(velocity[1]), float(velocity[0]) + 1e-6))
        trajectory: list[Waypoint] = []
        confidence: list[float] = []
        covariance: list[FloatArray] = []

        current_speed = speed_mps
        x = float(center_xyz[0])
        y = float(center_xyz[1])

        for step in range(self.num_steps):
            t = step * self.dt_s
            # Decay speed toward lane speed
            current_speed = current_speed + self.velocity_decay_rate * (lane_speed - current_speed)
            x += float(velocity[0] / max(speed_mps, 0.1)) * current_speed * self.dt_s
            y += float(velocity[1] / max(speed_mps, 0.1)) * current_speed * self.dt_s

            trajectory.append(
                Waypoint(x=x, y=y, yaw=heading, velocity=current_speed, timestamp=t)
            )

            # Confidence decays over time
            conf = max(0.1, 0.9 - 0.016 * step)
            confidence.append(conf)

            # Uncertainty grows over time
            sigma_lon = self.sigma_lon_base + self.sigma_lon_growth * step
            sigma_lat = self.sigma_lat_base + self.sigma_lat_growth * step
            covariance.append(
                np.array([[sigma_lon**2, 0.0], [0.0, sigma_lat**2]], dtype=np.float32)
            )

        return trajectory, confidence, covariance

    def _predict_lane_following(
        self,
        lane,
        center_xyz: np.ndarray,
        speed_mps: float,
        lane_speed: float,
    ) -> tuple[list[Waypoint], list[float], list[FloatArray]]:
        projection = project_point_to_centerline(lane.centerline_world, center_xyz)
        trajectory: list[Waypoint] = []
        confidence: list[float] = []
        covariance: list[FloatArray] = []

        current_speed = speed_mps
        accumulated_s = projection.s

        for step in range(self.num_steps):
            t = step * self.dt_s
            # Decay speed toward lane speed limit
            current_speed = current_speed + self.velocity_decay_rate * (lane_speed - current_speed)
            accumulated_s += current_speed * self.dt_s

            point_xyz, heading_rad = sample_centerline_at_s(lane.centerline_world, accumulated_s)
            trajectory.append(
                Waypoint(
                    x=float(point_xyz[0]),
                    y=float(point_xyz[1]),
                    yaw=float(heading_rad),
                    velocity=float(current_speed),
                    timestamp=float(t),
                )
            )

            # Confidence decays, but lane-following is more confident than free-space
            conf = max(0.15, 0.95 - 0.014 * step)
            confidence.append(conf)

            # Uncertainty: longitudinal grows faster than lateral (lane constrains lateral)
            sigma_lon = self.sigma_lon_base + self.sigma_lon_growth * step
            sigma_lat = self.sigma_lat_base + (self.sigma_lat_growth * step * 0.6)  # lane constrains lateral
            covariance.append(
                np.array([[sigma_lon**2, 0.0], [0.0, sigma_lat**2]], dtype=np.float32)
            )

        return trajectory, confidence, covariance

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


def build_prediction_module(runtime_config):
    tuning = getattr(runtime_config, "tuning", {}) or {}
    prediction_tuning = tuning.get("prediction", {})
    if prediction_tuning:
        valid_keys = LaneAwarePredictionModule.__init__.__code__.co_varnames
        kwargs = {k: v for k, v in prediction_tuning.items() if k in valid_keys}
        return LaneAwarePredictionModule(**kwargs)
    return LaneAwarePredictionModule()


StubPredictionModule = LaneAwarePredictionModule
