from __future__ import annotations

from autonomy_demo.interfaces.types import EgoPose, SensorFrameBundle


class StubLocalizationModule:
    """TODO(PRD 3.2.4): replace with GNSS/map alignment and Frenet computations."""

    def run(self, bundle: SensorFrameBundle) -> EgoPose:
        return EgoPose(
            world_xyz=bundle.gnss.world_xyz,
            yaw_rad=0.0,
            speed_mps=12.5,
            acceleration_mps2=0.1,
            current_lane_id="lane_001",
            frenet_s=float(bundle.gnss.world_xyz[0]),
            frenet_d=0.0,
            heading_error_rad=0.0,
        )

