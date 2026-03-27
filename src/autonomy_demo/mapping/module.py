from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import LaneLineType
from autonomy_demo.interfaces.types import (
    ConeDetection,
    DrivableSpaceMask,
    EgoPose,
    LaneLine,
    LocalMap,
    ObjectDetection,
    StaticLaneSegment,
    TrafficLightDetection,
)
from autonomy_demo.mapping.lane_graph import LaneGraphProvider


class MapAwareMappingModule:
    """Build an ego-centered typed local map from the cached lane graph and perception outputs."""

    def __init__(
        self,
        lane_graph_provider: LaneGraphProvider | None = None,
        *,
        lane_horizon_radius_m: float = 75.0,
        lane_limit: int = 14,
    ) -> None:
        self.lane_graph_provider = lane_graph_provider
        self.lane_horizon_radius_m = lane_horizon_radius_m
        self.lane_limit = lane_limit

    def prepare(self, simulation, scenario) -> None:
        if self.lane_graph_provider is not None:
            self.lane_graph_provider.prepare_from_simulation(simulation)

    def run(
        self,
        detections: list[ObjectDetection],
        lanes: list[LaneLine],
        drivable_space: DrivableSpaceMask,
        cones: list[ConeDetection],
        traffic_lights: list[TrafficLightDetection],
        ego_pose: EgoPose,
    ) -> LocalMap:
        static_lanes = self._static_lanes_from_graph(ego_pose)
        if not static_lanes:
            static_lanes = self._fallback_static_lanes(lanes, ego_pose)
        temporary_boundaries = [
            lane for lane in lanes if lane.line_type == LaneLineType.TEMPORARY
        ]
        closed_lanes = [ego_pose.current_lane_id] if cones and ego_pose.current_lane_id else []
        return LocalMap(
            static_lanes=static_lanes,
            dynamic_agents=detections,
            cone_instances=cones,
            temporary_boundaries=temporary_boundaries,
            closed_lanes=closed_lanes,
            traffic_signal_states=traffic_lights,
            drivable_space=drivable_space,
        )

    def _static_lanes_from_graph(self, ego_pose: EgoPose) -> list[StaticLaneSegment]:
        lane_graph = self.lane_graph_provider.lane_graph if self.lane_graph_provider else None
        if lane_graph is None:
            return []
        nearby = lane_graph.nearby_lanes(
            np.asarray(ego_pose.world_xyz, dtype=np.float32),
            radius_m=self.lane_horizon_radius_m,
            limit=self.lane_limit,
        )
        if ego_pose.current_lane_id and ego_pose.current_lane_id in lane_graph.segments:
            current_lane = lane_graph.segments[ego_pose.current_lane_id]
            if all(segment.lane_id != current_lane.lane_id for segment in nearby):
                nearby.insert(0, current_lane)
        return nearby

    def _fallback_static_lanes(
        self,
        lanes: list[LaneLine],
        ego_pose: EgoPose,
    ) -> list[StaticLaneSegment]:
        static_lanes = [
            StaticLaneSegment(
                lane_id=lane.lane_id,
                centerline_world=np.asarray(lane.polyline_world, dtype=np.float32),
                speed_limit_mps=22.35,
            )
            for lane in lanes
        ]
        if static_lanes:
            return static_lanes
        return [
            StaticLaneSegment(
                lane_id=ego_pose.current_lane_id,
                centerline_world=np.array(
                    [
                        [float(ego_pose.world_xyz[0]), float(ego_pose.world_xyz[1]), float(ego_pose.world_xyz[2])],
                        [
                            float(ego_pose.world_xyz[0] + 30.0),
                            float(ego_pose.world_xyz[1]),
                            float(ego_pose.world_xyz[2]),
                        ],
                    ],
                    dtype=np.float32,
                ),
                speed_limit_mps=22.35,
            )
        ]


def build_mapping_module(runtime_config, lane_graph_provider: LaneGraphProvider | None = None):
    return MapAwareMappingModule(lane_graph_provider=lane_graph_provider)


StubMappingModule = MapAwareMappingModule
