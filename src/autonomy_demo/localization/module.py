from __future__ import annotations

from autonomy_demo.interfaces.types import EgoPose, SensorFrameBundle


class StubLocalizationModule:
    """TODO(PRD 3.2.4): replace with GNSS/map alignment and Frenet computations."""

    def run(self, bundle: SensorFrameBundle) -> EgoPose:
        metadata = bundle.metadata or {}
        synthetic = bool(metadata.get("synthetic", False))
        return EgoPose(
            world_xyz=bundle.gnss.world_xyz,
            yaw_rad=float(metadata.get("ego_yaw_rad", 0.0)),
            speed_mps=float(metadata.get("ego_speed_mps", 12.5 if synthetic else 0.0)),
            acceleration_mps2=float(
                metadata.get("ego_acceleration_mps2", 0.1 if synthetic else 0.0)
            ),
            current_lane_id=str(metadata.get("ego_lane_id", "lane_001")),
            frenet_s=float(metadata.get("ego_route_progress_m", bundle.gnss.world_xyz[0])),
            frenet_d=float(metadata.get("ego_lateral_error_m", 0.0)),
            heading_error_rad=float(metadata.get("ego_heading_error_rad", 0.0)),
        )
