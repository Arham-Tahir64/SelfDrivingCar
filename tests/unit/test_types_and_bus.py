import numpy as np

from autonomy_demo.interfaces.enums import BehaviorState, LaneLineType, TrackState
from autonomy_demo.interfaces.types import CameraFrame, LaneLine, ObjectDetection, PerceptionStatus
from autonomy_demo.common.serialization import serialize
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


def test_serialization_preserves_perception_provenance_fields() -> None:
    detection = ObjectDetection(
        track_id=2,
        object_class="vehicle",
        world_bbox_3d=np.zeros((8, 3), dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        confidence=0.7,
        track_state=TrackState.TENTATIVE,
        source_modality="fused",
        source_sensor_ids=["front_camera", "lidar"],
        position_estimate_kind="fusion",
    )
    status = PerceptionStatus(
        active_mode="fused_v1",
        fallback_state="fused",
        counts_by_modality={"fused": 1},
        active_camera_sensors=["front_camera"],
        detection_count=1,
    )
    payload = serialize({"perception/detections": [detection], "perception/status": status})
    assert payload["perception/detections"][0]["source_modality"] == "fused"
    assert payload["perception/detections"][0]["source_sensor_ids"] == ["front_camera", "lidar"]
    assert payload["perception/detections"][0]["position_estimate_kind"] == "fusion"
    assert payload["perception/status"]["fallback_state"] == "fused"


def test_serialization_preserves_lane_provenance_fields() -> None:
    lane = LaneLine(
        lane_id="lane_left",
        polyline_image=np.array([[10.0, 10.0], [12.0, 20.0]], dtype=np.float32),
        polyline_world=np.array([[5.0, -1.5, 0.0], [15.0, -1.2, 0.0]], dtype=np.float32),
        line_type=LaneLineType.SOLID,
        confidence=0.8,
        source_modality="camera",
        source_sensor_ids=["front_camera"],
        position_estimate_kind="camera_projection",
    )
    payload = serialize({"perception/lanes": [lane]})
    assert payload["perception/lanes"][0]["source_modality"] == "camera"
    assert payload["perception/lanes"][0]["source_sensor_ids"] == ["front_camera"]
    assert payload["perception/lanes"][0]["position_estimate_kind"] == "camera_projection"
