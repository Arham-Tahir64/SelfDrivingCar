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
    ScenarioTrigger,
    StaticLaneSegment,
    TrafficLightDetection,
)
from autonomy_demo.mapping.lane_graph import LaneGraphProvider, parse_lane_id


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
        self._scenario_triggers: list[ScenarioTrigger] = []
        self._activated_trigger_lanes: dict[str, str] = {}

    def prepare(self, simulation, scenario) -> None:
        if self.lane_graph_provider is not None:
            self.lane_graph_provider.prepare_from_simulation(simulation)
        self._scenario_triggers = list(getattr(scenario, "triggers", []))
        self._activated_trigger_lanes.clear()

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
        temporary_boundaries = [lane for lane in lanes if lane.line_type == LaneLineType.TEMPORARY]
        closed_lanes: list[str] = []
        trigger_closed_lanes, trigger_boundaries = self._scenario_trigger_cues(
            ego_pose=ego_pose,
            static_lanes=static_lanes,
        )
        for lane_id in trigger_closed_lanes:
            if lane_id not in closed_lanes:
                closed_lanes.append(lane_id)
        temporary_boundaries.extend(trigger_boundaries)
        return LocalMap(
            static_lanes=static_lanes,
            dynamic_agents=detections,
            cone_instances=[],
            temporary_boundaries=temporary_boundaries,
            closed_lanes=closed_lanes,
            traffic_signal_states=traffic_lights,
            perceived_lanes=list(lanes),
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

    def _scenario_trigger_cues(
        self,
        *,
        ego_pose: EgoPose,
        static_lanes: list[StaticLaneSegment],
    ) -> tuple[list[str], list[LaneLine]]:
        if not self._scenario_triggers:
            return [], []
        closed_lanes: list[str] = []
        temporary_boundaries: list[LaneLine] = []
        for trigger in self._scenario_triggers:
            if trigger.at_s is not None and ego_pose.frenet_s < float(trigger.at_s):
                continue
            if trigger.type not in {"merge_required", "lane_closure"}:
                continue
            target_lane_id = self._resolved_or_activate_trigger_lane_id(
                trigger=trigger,
                ego_lane_id=ego_pose.current_lane_id,
                static_lanes=static_lanes,
            )
            if target_lane_id is None:
                continue
            if target_lane_id not in closed_lanes:
                closed_lanes.append(target_lane_id)
            if ego_pose.current_lane_id != target_lane_id:
                continue
            segment = next((lane for lane in static_lanes if lane.lane_id == target_lane_id), None)
            if segment is None:
                continue
            boundary_world = (
                segment.right_boundary_world
                if len(segment.right_boundary_world)
                else segment.centerline_world
            )
            temporary_boundaries.append(
                LaneLine(
                    lane_id=f"trigger:{trigger.type}:{target_lane_id}",
                    polyline_image=np.zeros((len(boundary_world), 2), dtype=np.float32),
                    polyline_world=np.asarray(boundary_world, dtype=np.float32),
                    line_type=LaneLineType.TEMPORARY,
                    confidence=1.0,
                )
            )
        return closed_lanes, temporary_boundaries

    def _resolved_or_activate_trigger_lane_id(
        self,
        *,
        trigger: ScenarioTrigger,
        ego_lane_id: str,
        static_lanes: list[StaticLaneSegment],
    ) -> str | None:
        trigger_key = self._trigger_key(trigger)
        if trigger_key in self._activated_trigger_lanes:
            return self._activated_trigger_lanes[trigger_key]
        resolved_lane_id = self._resolve_trigger_lane_id(
            trigger=trigger,
            ego_lane_id=ego_lane_id,
            static_lanes=static_lanes,
        )
        if resolved_lane_id is not None:
            self._activated_trigger_lanes[trigger_key] = resolved_lane_id
        return resolved_lane_id

    def _resolve_trigger_lane_id(
        self,
        *,
        trigger: ScenarioTrigger,
        ego_lane_id: str,
        static_lanes: list[StaticLaneSegment],
    ) -> str | None:
        if isinstance(trigger.lane_id, str):
            if any(segment.lane_id == trigger.lane_id for segment in static_lanes):
                return trigger.lane_id
        if isinstance(trigger.lane_id, int):
            for segment in static_lanes:
                parsed = parse_lane_id(segment.lane_id)
                if parsed is not None and parsed[2] == int(trigger.lane_id):
                    return segment.lane_id
        if any(segment.lane_id == ego_lane_id for segment in static_lanes):
            return ego_lane_id
        if static_lanes:
            return static_lanes[0].lane_id
        return None

    def _trigger_key(self, trigger: ScenarioTrigger) -> str:
        return f"{trigger.type}:{trigger.at_s}:{trigger.lane_id}:{sorted(trigger.metadata.items())}"


def build_mapping_module(runtime_config, lane_graph_provider: LaneGraphProvider | None = None):
    return MapAwareMappingModule(lane_graph_provider=lane_graph_provider)


StubMappingModule = MapAwareMappingModule
