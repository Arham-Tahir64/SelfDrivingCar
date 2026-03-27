from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import EgoPose, SensorFrameBundle
from autonomy_demo.mapping.lane_graph import LaneGraphProvider, heading_error_to_lane


class MapAwareLocalizationModule:
    """Map-aware ego localization with typed lane/Frenet outputs."""

    def __init__(self, lane_graph_provider: LaneGraphProvider | None = None) -> None:
        self.lane_graph_provider = lane_graph_provider

    def prepare(self, simulation, scenario) -> None:
        if self.lane_graph_provider is not None:
            self.lane_graph_provider.prepare_from_simulation(simulation)

    def run(self, bundle: SensorFrameBundle) -> EgoPose:
        metadata = bundle.metadata or {}
        synthetic = bool(metadata.get("synthetic", False))
        world_xyz = np.asarray(bundle.gnss.world_xyz, dtype=np.float32)
        yaw_rad = float(metadata.get("ego_yaw_rad", 0.0))
        speed_mps = float(metadata.get("ego_speed_mps", 12.5 if synthetic else 0.0))
        acceleration_mps2 = float(
            metadata.get("ego_acceleration_mps2", 0.1 if synthetic else 0.0)
        )

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
