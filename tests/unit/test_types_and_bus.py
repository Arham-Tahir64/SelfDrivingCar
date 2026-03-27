import numpy as np

from autonomy_demo.interfaces.enums import BehaviorState, TrackState
from autonomy_demo.interfaces.types import CameraFrame, ObjectDetection
from autonomy_demo.orchestration.event_bus import InProcessEventBus


def test_object_detection_validation() -> None:
    detection = ObjectDetection(
        track_id=1,
        object_class="vehicle",
        world_bbox_3d=np.zeros((8, 3), dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        confidence=0.5,
        track_state=TrackState.CONFIRMED,
    )
    assert detection.track_state == TrackState.CONFIRMED


def test_camera_frame_validation() -> None:
    frame = CameraFrame("front", np.zeros((8, 8, 3), dtype=np.float32), 0.0)
    assert frame.frame.shape == (8, 8, 3)


def test_event_bus_publish_subscribe() -> None:
    bus = InProcessEventBus()
    observed: list[str] = []
    bus.subscribe("topic", lambda topic, payload: observed.append(topic))
    bus.publish("topic", {"ok": True})
    assert observed == ["topic"]
    assert bus.snapshot()["topic"]["ok"] is True


def test_behavior_state_enum_contains_prd_states() -> None:
    assert BehaviorState.CONSTRUCTION_NAVIGATE.value == "CONSTRUCTION_NAVIGATE"

