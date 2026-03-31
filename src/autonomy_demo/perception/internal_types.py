from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from autonomy_demo.interfaces.enums import ObjectClass, TrackState, TrafficLightState


FloatArray = NDArray[np.float32]


@dataclass(slots=True)
class FrameDetection2D:
    bbox_xyxy: FloatArray
    object_class: ObjectClass
    confidence: float
    source_sensor_id: str = "front_camera"
    source_modality: str = "camera"
    source_sensor_ids: list[str] = field(default_factory=list)
    position_estimate_kind: str = "camera_projection"
    world_bbox_3d: FloatArray | None = None
    velocity_xyz: FloatArray | None = None
    world_xyz: FloatArray | None = None
    preferred_track_id: int | None = None
    traffic_light_state: TrafficLightState | None = None


@dataclass(slots=True)
class TrackedDetection2D(FrameDetection2D):
    track_id: int = -1
    track_state: TrackState = TrackState.TENTATIVE


@dataclass(slots=True)
class LidarClusterDetection:
    centroid_xyz: FloatArray
    world_bbox_3d: FloatArray
    object_class: ObjectClass
    confidence: float
    point_count: int
    source_modality: str = "lidar"
    source_sensor_ids: list[str] = field(default_factory=lambda: ["lidar"])
    position_estimate_kind: str = "lidar_cluster"


@dataclass(slots=True)
class TrackedLidarClusterDetection(LidarClusterDetection):
    track_id: int = -1
    track_state: TrackState = TrackState.TENTATIVE
    velocity_xyz: FloatArray | None = None
