from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import LaneLineType, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    ConeDetection,
    DrivableSpaceMask,
    LaneLine,
    ObjectDetection,
    SensorFrameBundle,
    TrafficLightDetection,
)


class StubPerceptionModule:
    """TODO(PRD 3.2.3): replace with model adapters and tracking."""

    def run(self, bundle: SensorFrameBundle):
        detection = ObjectDetection(
            track_id=1,
            object_class=ObjectClass.VEHICLE,
            world_bbox_3d=np.array(
                [
                    [10.0, 1.0, 0.0],
                    [11.0, 1.0, 0.0],
                    [11.0, 2.0, 0.0],
                    [10.0, 2.0, 0.0],
                    [10.0, 1.0, 1.5],
                    [11.0, 1.0, 1.5],
                    [11.0, 2.0, 1.5],
                    [10.0, 2.0, 1.5],
                ],
                dtype=np.float32,
            ),
            velocity=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            confidence=0.95,
            track_state=TrackState.CONFIRMED,
        )
        lane = LaneLine(
            lane_id="lane_001",
            polyline_image=np.array([[0, 5], [10, 5], [20, 5]], dtype=np.float32),
            polyline_world=np.array([[0, 50, 0], [10, 50, 0], [20, 50, 0]], dtype=np.float32),
            line_type=LaneLineType.SOLID,
            confidence=0.9,
        )
        drivable = DrivableSpaceMask(
            mask=np.ones((32, 32), dtype=np.bool_),
            class_probabilities=np.ones((32, 32, 2), dtype=np.float32) * 0.5,
            source_sensor_id=bundle.front_camera.sensor_id,
        )
        traffic_light = TrafficLightDetection(
            world_xyz=np.array([30.0, 50.0, 5.0], dtype=np.float32),
            state=TrafficLightState.GREEN,
            stop_line_distance_m=20.0,
            confidence=0.8,
        )
        cone = ConeDetection(world_xyz=np.array([25.0, 48.0, 0.0], dtype=np.float32), confidence=0.88)
        return [detection], [lane], drivable, [traffic_light], [cone]

