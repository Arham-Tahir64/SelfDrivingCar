from __future__ import annotations

import json

import numpy as np

from autonomy_demo.interfaces.enums import BehaviorState, LaneLineType, TopicName, TrafficLightState
from autonomy_demo.interfaces.types import (
    ControlCommand,
    EgoPose,
    EgoTrajectory,
    LaneLine,
    LidarFrame,
    LocalMap,
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
    bus.publish(TopicName.LOCALIZATION_EGO_POSE.value, _ego_pose())
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
    assert TopicName.VISUALIZATION_WORLD_LAYER.value in topics
    assert "traffic_lights" in topics[TopicName.VISUALIZATION_WORLD_LAYER.value]
    assert TopicName.VISUALIZATION_LIDAR_PREVIEW.value in topics
    assert len(topics[TopicName.VISUALIZATION_LIDAR_PREVIEW.value]["points"]) == 300
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
    assert TopicName.VISUALIZATION_WORLD_LAYER.value in envelope["topics"]


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
