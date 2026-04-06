from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import EgoPose, SensorFrameBundle
from autonomy_demo.mapping.lane_graph import LaneGraphProvider, heading_error_to_lane


class MapAwareLocalizationModule:
    """Map-aware ego localization with typed lane/Frenet outputs.

    Fuses GNSS position with IMU dead-reckoning via a complementary filter
    to smooth pose and add resilience to GNSS drift.
    """

    def __init__(
        self,
        lane_graph_provider: LaneGraphProvider | None = None,
        *,
        gnss_alpha: float = 0.85,
    ) -> None:
        self.lane_graph_provider = lane_graph_provider
        self.gnss_alpha = gnss_alpha
        self._prev_xyz: np.ndarray | None = None
        self._prev_yaw: float | None = None
        self._prev_time: float | None = None
        self._prev_speed: float = 0.0

    def prepare(self, simulation, scenario) -> None:
        if self.lane_graph_provider is not None:
            self.lane_graph_provider.prepare_from_simulation(simulation)
        self._prev_xyz = None
        self._prev_yaw = None
        self._prev_time = None
        self._prev_speed = 0.0

    def run(self, bundle: SensorFrameBundle) -> EgoPose:
        metadata = bundle.metadata or {}
        synthetic = bool(metadata.get("synthetic", False))
        gnss_xyz = np.asarray(bundle.gnss.world_xyz, dtype=np.float32)
        metadata_yaw = float(metadata.get("ego_yaw_rad", 0.0))
        speed_mps = float(metadata.get("ego_speed_mps", 12.5 if synthetic else 0.0))
        acceleration_mps2 = float(
            metadata.get("ego_acceleration_mps2", 0.1 if synthetic else 0.0)
        )

        # --- IMU-augmented fusion ---
        imu = bundle.imu
        imu_time = float(imu.timestamp_s)
        dt = imu_time - self._prev_time if self._prev_time is not None else 0.0

        if self._prev_xyz is not None and 0.0 < dt < 1.0:
            # Dead-reckon position from previous state using IMU acceleration.
            accel_world = np.asarray(imu.acceleration_xyz, dtype=np.float32)
            dr_xyz = self._prev_xyz.copy()
            # Integrate velocity: v = v_prev + a * dt
            dr_speed = self._prev_speed + float(np.linalg.norm(accel_world[:2])) * dt
            # Integrate position along heading: p = p_prev + v * heading * dt
            yaw = self._prev_yaw if self._prev_yaw is not None else metadata_yaw
            dr_xyz[0] += float(np.cos(yaw)) * self._prev_speed * dt
            dr_xyz[1] += float(np.sin(yaw)) * self._prev_speed * dt
            dr_xyz[2] = gnss_xyz[2]  # Trust GNSS for altitude

            # Complementary filter: blend GNSS (high-freq noise) with dead-reckoning.
            alpha = self.gnss_alpha
            world_xyz = alpha * gnss_xyz + (1.0 - alpha) * dr_xyz

            # Smooth yaw using gyro z-axis (yaw rate).
            gyro_yaw_rate = float(imu.gyro_xyz[2])
            dr_yaw = yaw + gyro_yaw_rate * dt
            yaw_rad = alpha * metadata_yaw + (1.0 - alpha) * dr_yaw
        else:
            # First tick or unreasonable dt — use raw GNSS.
            world_xyz = gnss_xyz
            yaw_rad = metadata_yaw

        self._prev_xyz = world_xyz.copy()
        self._prev_yaw = yaw_rad
        self._prev_time = imu_time
        self._prev_speed = speed_mps

        # --- Lane projection (unchanged) ---
        lane_graph = self.lane_graph_provider.lane_graph if self.lane_graph_provider else None
        projection = lane_graph.nearest_projection(world_xyz) if lane_graph is not None else None
        if projection is not None:
            current_lane_id = projection.lane_id
            frenet_s = float(projection.s)
            frenet_d = float(projection.d)
            heading_error_rad = heading_error_to_lane(yaw_rad, projection)
        else:
            current_lane_id = str(metadata.get("ego_lane_id", "lane_001"))
            frenet_s = float(metadata.get("ego_route_progress_m", world_xyz[0]))
            frenet_d = float(metadata.get("ego_lateral_error_m", 0.0))
            heading_error_rad = float(metadata.get("ego_heading_error_rad", 0.0))

        return EgoPose(
            world_xyz=world_xyz,
            yaw_rad=yaw_rad,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            current_lane_id=current_lane_id,
            frenet_s=frenet_s,
            frenet_d=frenet_d,
            heading_error_rad=heading_error_rad,
        )


def build_localization_module(runtime_config, lane_graph_provider: LaneGraphProvider | None = None):
    return MapAwareLocalizationModule(lane_graph_provider=lane_graph_provider)


StubLocalizationModule = MapAwareLocalizationModule
