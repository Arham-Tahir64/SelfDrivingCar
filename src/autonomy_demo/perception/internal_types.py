from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from autonomy_demo.interfaces.enums import ObjectClass, TrackState, TrafficLightState


FloatArray = NDArray[np.float32]


@dataclass(slots=True)
class FrameDetection2D:
    bbox_xyxy: FloatArray
    object_class: ObjectClass
    confidence: float
    world_bbox_3d: FloatArray | None = None
    velocity_xyz: FloatArray | None = None
    world_xyz: FloatArray | None = None
    preferred_track_id: int | None = None
    traffic_light_state: TrafficLightState | None = None


@dataclass(slots=True)
class TrackedDetection2D(FrameDetection2D):
    track_id: int = -1
    track_state: TrackState = TrackState.TENTATIVE
