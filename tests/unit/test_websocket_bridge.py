from __future__ import annotations

import json

import numpy as np

from autonomy_demo.interfaces.enums import BehaviorState, LaneLineType, ObjectClass, TopicName, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    AgentPrediction,
    ControlCommand,
    EgoPose,
    EgoTrajectory,
    LaneLine,
    LidarFrame,
    LocalMap,
    ObjectDetection,
    StaticLaneSegment,
    TrafficLightDetection,
    Waypoint,
)
from autonomy_demo.orchestration.event_bus import InProcessEventBus
from autonomy_demo.visualization.websocket_bridge import WebSocketBridge


def _ego_pose(*, x: float = 0.0) -> EgoPose:
    return EgoPose(
        world_xyz=np.array([x, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=8.0,
        acceleration_mps2=0.0,
        current_lane_id="lane_route",
        frenet_s=x,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )


def _lane_line() -> LaneLine:
    return LaneLine(
        lane_id="temp-1",
        polyline_image=np.array([[0.0, 0.0], [4.0, 4.0]], dtype=np.float32),
        polyline_world=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
        line_type=LaneLineType.TEMPORARY,
        confidence=0.9,
    )


def _local_map() -> LocalMap:
    lane = StaticLaneSegment(
        lane_id="lane_route",
        centerline_world=np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
        speed_limit_mps=12.0,
        left_boundary_world=np.array([[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]], dtype=np.float32),
        right_boundary_world=np.array([[0.0, -1.75, 0.0], [20.0, -1.75, 0.0]], dtype=np.float32),
    )
    return LocalMap(
        static_lanes=[lane],
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[_lane_line()],
        closed_lanes=["lane_closed"],
        traffic_signal_states=[],
        perceived_lanes=[_lane_line()],
        drivable_space=None,
    )


def _trajectory() -> EgoTrajectory:
    return EgoTrajectory(
        waypoints=[
            Waypoint(x=0.0, y=0.0, yaw=0.0, velocity=8.0, timestamp=0.0),
            Waypoint(x=5.0, y=0.0, yaw=0.0, velocity=8.0, timestamp=0.3),
            Waypoint(x=10.0, y=0.0, yaw=0.0, velocity=8.0, timestamp=0.6),
        ],
        cost=1.0,
        behavior_state=BehaviorState.LANE_KEEP,
    )


def _world_layer(*, signature: str, state: str = "RED") -> dict[str, object]:
    return {
        "signature": signature,
        "roads": [
            {
                "lane_id": "lane_route",
                "polygon_world": [[0.0, -1.75, 0.0], [20.0, -1.75, 0.0], [20.0, 1.75, 0.0], [0.0, 1.75, 0.0]],
                "centerline_world": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "is_route": True,
                "visibility_class": "route",
            }
        ],
        "lane_markers": [
            {
                "marker_id": "route-left",
                "polyline_world": [[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]],
                "is_route": True,
                "visibility_class": "route",
            }
        ],
        "sidewalks": [
            {
                "sidewalk_id": "left",
                "polygon_world": [[0.0, 1.75, 0.0], [20.0, 1.75, 0.0], [20.0, 3.5, 0.0], [0.0, 3.5, 0.0]],
                "edge_world": [[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]],
                "visibility_class": "route",
            }
        ],
        "traffic_lights": [
            {
                "actor_id": 101,
                "world_xyz": [8.0, 1.5, 3.2],
                "yaw_deg": 90.0,
                "state": state,
                "confidence": 0.95,
                "visibility_class": "route",
            }
        ],
    }


def _prior_map(*, signature: str = "prior-a") -> dict[str, object]:
    return {
        "map_name": "Town01",
        "signature": signature,
        "bounds_world": {"min_x": -20.0, "max_x": 40.0, "min_y": -12.0, "max_y": 12.0},
        "roads": [
            {
                "lane_id": "lane_route",
                "polygon_world": [[0.0, -1.75, 0.0], [20.0, -1.75, 0.0], [20.0, 1.75, 0.0], [0.0, 1.75, 0.0]],
                "centerline_world": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "is_route": False,
                "is_junction": False,
            }
        ],
        "lane_markers": [
            {
                "marker_id": "prior-left",
                "polyline_world": [[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]],
                "is_route": False,
            }
        ],
        "sidewalks": [
            {
                "sidewalk_id": "prior-sidewalk-left",
                "polygon_world": [[0.0, 1.75, 0.0], [20.0, 1.75, 0.0], [20.0, 3.5, 0.0], [0.0, 3.5, 0.0]],
                "edge_world": [[0.0, 1.75, 0.0], [20.0, 1.75, 0.0]],
            }
        ],
        "traffic_lights": [
            {
                "actor_id": 101,
                "world_xyz": [8.0, 1.5, 3.2],
                "yaw_deg": 90.0,
                "state": "RED",
                "confidence": 0.35,
            }
        ],
        "route_polyline_world": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
    }


def _bev_grid(fill: int) -> dict[str, object]:
    grid = np.full((8, 8), fill, dtype=np.uint8)
    return {
        "grid": grid,
        "rows": 8,
        "cols": 8,
        "cell_size_m": 0.5,
        "x_min_m": -2.0,
        "x_max_m": 2.0,
        "y_min_m": -2.0,
        "y_max_m": 2.0,
    }


def _object_detection(
    *,
    track_id: int,
    source_modality: str,
    track_state: TrackState,
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> ObjectDetection:
    return ObjectDetection(
        track_id=track_id,
        object_class=ObjectClass.VEHICLE,
        world_bbox_3d=np.array(
            [
                [x - 1.0, y - 0.9, 0.0],
                [x + 1.0, y - 0.9, 0.0],
                [x + 1.0, y + 0.9, 0.0],
                [x - 1.0, y + 0.9, 0.0],
                [x - 1.0, y - 0.9, 1.6],
                [x + 1.0, y - 0.9, 1.6],
                [x + 1.0, y + 0.9, 1.6],
                [x - 1.0, y + 0.9, 1.6],
            ],
            dtype=np.float32,
        ),
        velocity=np.array([vx, vy, 0.0], dtype=np.float32),
        confidence=0.88,
        track_state=track_state,
        source_modality=source_modality,
        source_sensor_ids=["lidar"],
        position_estimate_kind="lidar_cluster" if source_modality == "lidar" else "fusion",
    )


def _agent_prediction(track_id: int, *, x: float, y: float, vx: float, vy: float) -> AgentPrediction:
    return AgentPrediction(
        track_id=track_id,
        object_class=ObjectClass.VEHICLE,
        predicted_trajectory=[
            Waypoint(x=x + (vx * 0.3), y=y + (vy * 0.3), yaw=0.0, velocity=max(vx, 0.1), timestamp=0.3),
            Waypoint(x=x + (vx * 0.6), y=y + (vy * 0.6), yaw=0.0, velocity=max(vx, 0.1), timestamp=0.6),
            Waypoint(x=x + (vx * 0.9), y=y + (vy * 0.9), yaw=0.0, velocity=max(vx, 0.1), timestamp=0.9),
        ],
        confidence_by_step=[0.9, 0.82, 0.74],
        covariance_by_step=None,
    )


def _publish_common(bus: InProcessEventBus) -> None:
    bus.publish(
        TopicName.SCENARIO_INFO.value,
        {
            "scenario_id": "SC-01",
            "name": "Smoke",
            "map_name": "Town01",
            "max_duration_s": 60.0,
        },
    )
    bus.publish(TopicName.MAP_LOCAL_MAP.value, _local_map())
    bus.publish(TopicName.VISUALIZATION_WORLD_LAYER.value, _world_layer(signature="sig-a"))
    bus.publish(TopicName.VISUALIZATION_PRIOR_MAP.value, _prior_map())
    bus.publish(TopicName.LOCALIZATION_EGO_POSE.value, _ego_pose())
    bus.publish(
        TopicName.PERCEPTION_STATUS.value,
        {
            "active_mode": "fused_v1",
            "fallback_state": "fused",
        },
    )
    bus.publish(
        TopicName.PERCEPTION_DETECTIONS.value,
        [
            _object_detection(
                track_id=7,
                source_modality="lidar",
                track_state=TrackState.CONFIRMED,
                x=12.0,
                y=0.4,
                vx=4.0,
                vy=0.0,
            ),
            _object_detection(
                track_id=8,
                source_modality="lidar",
                track_state=TrackState.TENTATIVE,
                x=18.0,
                y=5.8,
                vx=0.5,
                vy=0.0,
            ),
            _object_detection(
                track_id=17,
                source_modality="camera",
                track_state=TrackState.CONFIRMED,
                x=10.0,
                y=-2.0,
                vx=2.0,
                vy=0.0,
            ),
        ],
    )
    bus.publish(
        TopicName.PREDICTION_AGENTS.value,
        [
            _agent_prediction(track_id=7, x=12.0, y=0.4, vx=4.0, vy=0.0),
            _agent_prediction(track_id=8, x=18.0, y=5.8, vx=0.5, vy=0.0),
        ],
    )
    bus.publish(
        TopicName.CONTROL_VEHICLE_COMMAND.value,
        ControlCommand(throttle=0.2, steer=0.0, brake=0.0),
    )
    bus.publish(TopicName.PLANNING_EGO_TRAJECTORY.value, _trajectory())
    bus.publish(TopicName.PIPELINE_LATENCY.value, {"perception": 8.0, "total": 18.0})
    bus.publish(
        TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
        [
            TrafficLightDetection(
                world_xyz=np.array([8.0, 1.5, 3.0], dtype=np.float32),
                state=TrafficLightState.RED,
                stop_line_distance_m=10.0,
                confidence=0.9,
            )
        ],
    )
    bus.publish(
        TopicName.SENSOR_LIDAR.value,
        LidarFrame(
            points_xyz=np.stack(
                [
                    np.linspace(0.0, 20.0, 1000, dtype=np.float32),
                    np.linspace(-5.0, 5.0, 1000, dtype=np.float32),
                    np.zeros(1000, dtype=np.float32),
                ],
                axis=1,
            ),
            timestamp_s=0.0,
        ),
    )
    bus.publish(TopicName.VISUALIZATION_BEV_DRIVABLE.value, _bev_grid(40))


def _capture_envelopes(bridge: WebSocketBridge) -> list[dict[str, object]]:
    queued: list[dict[str, object]] = []

    def _capture(*, message_kind: str, text: str) -> None:
        envelope = json.loads(text)
        assert envelope["message_kind"] == message_kind
        queued.append(envelope)

    bridge._queue_text = _capture  # type: ignore[method-assign]
    return queued


def test_bootstrap_envelope_merges_retained_dynamic_and_heavy_topics(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    bridge.register(object())
    _publish_common(bus)

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 10.0)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 1, "sim_time_s": 0.1})

    assert len(queued) == 2
    assert queued[0]["message_kind"] == "bootstrap"
    assert queued[0]["topics"] == {}
    envelope = queued[-1]
    assert envelope["message_kind"] == "bootstrap"
    topics = envelope["topics"]
    assert TopicName.SCENARIO_INFO.value in topics
    assert TopicName.MAP_LOCAL_MAP.value in topics
    assert "perceived_lanes" not in topics[TopicName.MAP_LOCAL_MAP.value]
    assert topics[TopicName.MAP_LOCAL_MAP.value]["closed_lanes"] == ["lane_closed"]
    assert len(topics[TopicName.MAP_LOCAL_MAP.value]["temporary_boundaries"]) == 1
    assert TopicName.VISUALIZATION_PRIOR_MAP.value in topics
    assert topics[TopicName.VISUALIZATION_PRIOR_MAP.value]["map_name"] == "Town01"
    assert TopicName.VISUALIZATION_WORLD_LAYER.value in topics
    assert "traffic_lights" in topics[TopicName.VISUALIZATION_WORLD_LAYER.value]
    assert TopicName.VISUALIZATION_LIDAR_PREVIEW.value in topics
    assert len(topics[TopicName.VISUALIZATION_LIDAR_PREVIEW.value]["points"]) == 1000
    lidar_preview = topics[TopicName.VISUALIZATION_LIDAR_PREVIEW.value]
    assert len(lidar_preview["objects"]) == 2
    assert {obj["track_id"] for obj in lidar_preview["objects"]} == {7, 8}
    assert lidar_preview["threat_ids"] == [7]
    threat_object = next(obj for obj in lidar_preview["objects"] if obj["track_id"] == 7)
    assert threat_object["track_state"] == "CONFIRMED"
    assert threat_object["is_path_relevant"] is True
    assert threat_object["ghost_xy"] is not None
    tentative_object = next(obj for obj in lidar_preview["objects"] if obj["track_id"] == 8)
    assert tentative_object["track_state"] == "TENTATIVE"
    assert tentative_object["threat_rank"] == 0
    assert TopicName.VISUALIZATION_BEV_DRIVABLE.value in topics
    assert envelope["ws_stats"]["topic_count"] >= 6
    assert envelope["ws_stats"]["topic_bytes"][TopicName.MAP_LOCAL_MAP.value] > 0


def test_register_queues_immediate_bootstrap_before_first_tick(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    _publish_common(bus)

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 15.0)
    bridge.register(object())

    assert len(queued) == 1
    envelope = queued[0]
    assert envelope["message_kind"] == "bootstrap"
    assert envelope["tick_id"] == -1
    assert envelope["sim_time_s"] == 0.0
    assert TopicName.SCENARIO_INFO.value in envelope["topics"]
    assert TopicName.VISUALIZATION_PRIOR_MAP.value in envelope["topics"]
    assert TopicName.VISUALIZATION_WORLD_LAYER.value in envelope["topics"]
    assert TopicName.VISUALIZATION_LIDAR_PREVIEW.value in envelope["topics"]


def test_lidar_preview_filters_to_panel_roi_before_capping(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    _publish_common(bus)
    bus.publish(
        TopicName.SENSOR_LIDAR.value,
        LidarFrame(
            points_xyz=np.vstack(
                [
                    np.stack(
                        [
                            np.linspace(0.0, 40.0, 1800, dtype=np.float32),
                            np.linspace(-10.0, 10.0, 1800, dtype=np.float32),
                            np.zeros(1800, dtype=np.float32),
                        ],
                        axis=1,
                    ),
                    np.stack(
                        [
                            np.linspace(55.0, 80.0, 800, dtype=np.float32),
                            np.linspace(-5.0, 5.0, 800, dtype=np.float32),
                            np.zeros(800, dtype=np.float32),
                        ],
                        axis=1,
                    ),
                    np.stack(
                        [
                            np.linspace(5.0, 25.0, 400, dtype=np.float32),
                            np.linspace(55.0, 70.0, 400, dtype=np.float32),
                            np.zeros(400, dtype=np.float32),
                        ],
                        axis=1,
                    ),
                ]
            ),
            timestamp_s=0.0,
        ),
    )

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 16.0)
    bridge.register(object())

    lidar_preview = queued[-1]["topics"][TopicName.VISUALIZATION_LIDAR_PREVIEW.value]
    assert len(lidar_preview["points"]) == 1500
    assert lidar_preview["status"]["point_count"] == 1500
    assert all(-8.0 <= point[0] <= 50.0 for point in lidar_preview["points"])
    assert all(abs(point[1]) <= 50.0 for point in lidar_preview["points"])


def test_world_layer_static_updates_only_when_signature_changes(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    bridge.register(object())
    _publish_common(bus)

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 20.0)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 1, "sim_time_s": 0.1})
    queued.clear()

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 20.05)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 2, "sim_time_s": 0.15})
    assert not any(message["message_kind"] == "static_update" for message in queued)

    bus.publish(TopicName.VISUALIZATION_WORLD_LAYER.value, _world_layer(signature="sig-b", state="GREEN"))
    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 20.25)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 3, "sim_time_s": 0.3})

    static_messages = [message for message in queued if message["message_kind"] == "static_update"]
    assert static_messages
    world_layer_update = static_messages[-1]["topics"][TopicName.VISUALIZATION_WORLD_LAYER.value]
    assert world_layer_update["signature"] == "sig-b"
    assert "traffic_lights" not in world_layer_update


def test_prior_map_is_retained_and_only_updates_when_signature_changes(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    bridge.register(object())
    _publish_common(bus)

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 21.0)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 1, "sim_time_s": 0.1})
    queued.clear()

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 21.1)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 2, "sim_time_s": 0.2})
    assert not any(message["message_kind"] == "static_update" for message in queued)

    bus.publish(TopicName.VISUALIZATION_PRIOR_MAP.value, _prior_map(signature="prior-b"))
    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 21.3)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 3, "sim_time_s": 0.3})

    static_messages = [message for message in queued if message["message_kind"] == "static_update"]
    assert static_messages
    prior_map_update = static_messages[-1]["topics"][TopicName.VISUALIZATION_PRIOR_MAP.value]
    assert prior_map_update["signature"] == "prior-b"


def test_dynamic_frames_do_not_wait_for_heavy_topics(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    bridge.register(object())
    _publish_common(bus)

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 30.0)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 1, "sim_time_s": 0.1})
    queued.clear()

    bus.publish(TopicName.LOCALIZATION_EGO_POSE.value, _ego_pose(x=2.0))
    bus.publish(TopicName.VISUALIZATION_BEV_DRIVABLE.value, _bev_grid(80))

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 30.2)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 2, "sim_time_s": 0.22})

    assert queued
    first_dynamic = queued[-1]
    assert first_dynamic["message_kind"] == "dynamic_frame"
    assert TopicName.LOCALIZATION_EGO_POSE.value in first_dynamic["topics"]
    assert TopicName.VISUALIZATION_BEV_DRIVABLE.value not in first_dynamic["topics"]

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 30.65)
    bus.publish(TopicName.TICK_COMPLETE.value, {"tick_id": 3, "sim_time_s": 0.65})

    latest_dynamic = queued[-1]
    assert latest_dynamic["message_kind"] == "dynamic_frame"
    assert TopicName.VISUALIZATION_BEV_DRIVABLE.value in latest_dynamic["topics"]


def test_lidar_preview_marks_camera_only_mode_as_degraded(monkeypatch) -> None:
    bus = InProcessEventBus()
    bridge = WebSocketBridge()
    bridge.attach(bus)
    queued = _capture_envelopes(bridge)
    _publish_common(bus)
    bus.publish(
        TopicName.PERCEPTION_STATUS.value,
        {
            "active_mode": "camera_v1",
            "fallback_state": "camera_only",
        },
    )
    bus.publish(
        TopicName.PERCEPTION_DETECTIONS.value,
        [
            _object_detection(
                track_id=20,
                source_modality="camera",
                track_state=TrackState.CONFIRMED,
                x=10.0,
                y=1.0,
                vx=2.0,
                vy=0.0,
            )
        ],
    )

    monkeypatch.setattr("autonomy_demo.visualization.websocket_bridge.time.monotonic", lambda: 44.0)
    bridge.register(object())

    lidar_preview = queued[-1]["topics"][TopicName.VISUALIZATION_LIDAR_PREVIEW.value]
    assert lidar_preview["objects"] == []
    assert lidar_preview["status"]["degraded"] is True
