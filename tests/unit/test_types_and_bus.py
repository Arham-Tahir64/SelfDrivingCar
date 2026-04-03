import numpy as np

from autonomy_demo.interfaces.enums import BehaviorState, LaneLineType, TrackState
from autonomy_demo.interfaces.types import CameraFrame, EgoTrajectory, LaneLine, ObjectDetection, PerceptionStatus, Waypoint
from autonomy_demo.common.serialization import serialize
from autonomy_demo.orchestration.event_bus import InProcessEventBus
from autonomy_demo.planning.motion_planner import PlannerCandidate, PlannerCostBreakdown
from autonomy_demo.visualization.websocket_bridge import _serialize_candidates_for_dashboard


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


def test_planner_candidate_dashboard_serialization_preserves_debug_fields() -> None:
    candidate = PlannerCandidate(
        trajectory=EgoTrajectory(
            waypoints=[
                Waypoint(x=0.0, y=0.0, yaw=0.0, velocity=4.0, timestamp=0.0),
                Waypoint(x=1.0, y=0.2, yaw=0.1, velocity=3.5, timestamp=0.1),
            ],
            cost=12.5,
            behavior_state=BehaviorState.LANE_KEEP,
        ),
        lane_id="road_1:section_0:lane_2",
        target_speed_mps=5.0,
        score=12.5,
        feasible=False,
        reject_reason="dynamic_collision",
        reference_lane_id="road_1:section_0:lane_1",
        target_lane_id="road_1:section_0:lane_2",
        target_d_m=3.5,
        terminal_time_s=5.0,
        cost_breakdown=PlannerCostBreakdown(
            collision=0.8,
            cone_proximity=0.1,
            lane_deviation=0.2,
            jerk=0.3,
            speed_error=0.4,
            traffic_violation=1.0,
            route_progress=-0.6,
            total=12.5,
        ),
    )

    payload = _serialize_candidates_for_dashboard([candidate])

    assert payload[0]["feasible"] is False
    assert payload[0]["reject_reason"] == "dynamic_collision"
    assert payload[0]["reference_lane_id"] == "road_1:section_0:lane_1"
    assert payload[0]["target_lane_id"] == "road_1:section_0:lane_2"
    assert payload[0]["target_d_m"] == 3.5
    assert payload[0]["terminal_time_s"] == 5.0
    assert payload[0]["cost_breakdown"]["traffic_violation"] == 1.0
    assert payload[0]["cost_breakdown"]["total"] == 12.5
