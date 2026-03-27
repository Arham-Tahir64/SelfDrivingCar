from __future__ import annotations

import numpy as np

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


class StubMappingModule:
    """TODO(PRD 3.2.5): replace with OpenDRIVE lane graph and cone-driven closures."""

    def run(
        self,
        detections: list[ObjectDetection],
        lanes: list[LaneLine],
        drivable_space: DrivableSpaceMask,
        cones: list[ConeDetection],
        traffic_lights: list[TrafficLightDetection],
        ego_pose: EgoPose,
    ) -> LocalMap:
        static_lane = StaticLaneSegment(
            lane_id=ego_pose.current_lane_id,
            centerline_world=np.array([[0.0, 50.0, 0.0], [100.0, 50.0, 0.0]], dtype=np.float32),
            speed_limit_mps=22.35,
        )
        return LocalMap(
            static_lanes=[static_lane],
            dynamic_agents=detections,
            cone_instances=cones,
            temporary_boundaries=[lane for lane in lanes if lane.line_type.value == "TEMPORARY"],
            closed_lanes=[],
            traffic_signal_states=traffic_lights,
            drivable_space=drivable_space,
        )

