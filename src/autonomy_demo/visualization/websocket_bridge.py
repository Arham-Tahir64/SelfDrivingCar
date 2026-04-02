from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.serialization import serialize
from autonomy_demo.interfaces.enums import TopicName

logger = get_logger(__name__)

_WS_TOPICS = frozenset(
    {
        TopicName.LOCALIZATION_EGO_POSE.value,
        TopicName.PERCEPTION_DETECTIONS.value,
        TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
        TopicName.PERCEPTION_STATUS.value,
        TopicName.MAP_LOCAL_MAP.value,
        TopicName.PREDICTION_AGENTS.value,
        TopicName.PLANNING_EGO_TRAJECTORY.value,
        TopicName.PLANNING_CANDIDATES.value,
        TopicName.CONTROL_VEHICLE_COMMAND.value,
        TopicName.SCENARIO_INFO.value,
        TopicName.VISUALIZATION_CAMERA_OVERLAY.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PIPELINE_LATENCY.value,
    }
)

_SKIP_TOPICS = frozenset(
    {
        TopicName.SENSOR_CAMERA_FRONT.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PERCEPTION_DRIVABLE_SPACE.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
    }
)

_RETAINED_TOPICS = frozenset(
    {
        TopicName.SCENARIO_INFO.value,
        TopicName.MAP_LOCAL_MAP.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
    }
)

_FAST_DYNAMIC_TOPICS = frozenset(
    {
        TopicName.LOCALIZATION_EGO_POSE.value,
        TopicName.PERCEPTION_DETECTIONS.value,
        TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
        TopicName.PERCEPTION_STATUS.value,
        TopicName.CONTROL_VEHICLE_COMMAND.value,
        TopicName.PLANNING_EGO_TRAJECTORY.value,
        TopicName.PIPELINE_LATENCY.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
    }
)

_HEAVY_DYNAMIC_TOPICS = frozenset(
    {
        TopicName.VISUALIZATION_CAMERA_OVERLAY.value,
        TopicName.VISUALIZATION_LIDAR_PREVIEW.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
        TopicName.PREDICTION_AGENTS.value,
        TopicName.PLANNING_CANDIDATES.value,
    }
)

_WORLD_LAYER_STATIC_KEYS = ("signature", "roads", "lane_markers", "sidewalks")
_WORLD_LAYER_DYNAMIC_KEYS = ("traffic_lights",)

_LIDAR_MAX_POINTS = 300
_JPEG_QUALITY = 35
_OVERLAY_MAX_WIDTH = 360
_FAST_DYNAMIC_FPS = 8.0
_HEAVY_DYNAMIC_FPS = 2.0
_WAYPOINT_STRIDE = 2
_WS_STATS_LOG_INTERVAL_S = 5.0


def _stable_json(value: Any) -> str:
    return json.dumps(serialize(value), default=str, sort_keys=True, separators=(",", ":"))


def _json_size_bytes(value: Any) -> int:
    return len(_stable_json(value).encode("utf-8"))


def _hash_payload(value: Any) -> str:
    return _stable_json(value)


def _merge_topic(topics: dict[str, Any], topic: str, payload: Any) -> None:
    if payload is None:
        return
    existing = topics.get(topic)
    if isinstance(existing, dict) and isinstance(payload, dict):
        topics[topic] = {**existing, **payload}
        return
    topics[topic] = payload


def _serialize_payload(payload: Any) -> Any:
    return serialize({"payload": payload}).get("payload")


def _encode_overlay_jpeg(frame_rgb: np.ndarray) -> str | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    height, width = frame_rgb.shape[:2]
    if width > _OVERLAY_MAX_WIDTH:
        scale = _OVERLAY_MAX_WIDTH / width
        frame_rgb = cv2.resize(
            frame_rgb,
            (_OVERLAY_MAX_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    success, buf = cv2.imencode(".jpg", frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not success:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _downsample_lidar(points_xyz: np.ndarray) -> list[list[float]]:
    count = int(points_xyz.shape[0])
    if count == 0:
        return []
    if count <= _LIDAR_MAX_POINTS:
        return points_xyz.tolist()
    indices = np.linspace(0, count - 1, _LIDAR_MAX_POINTS, dtype=np.int32)
    return points_xyz[indices].tolist()


def _serialize_local_map_for_dashboard(local_map: Any) -> dict[str, Any] | None:
    if local_map is None:
        return None
    return serialize(
        {
        "closed_lanes": list(getattr(local_map, "closed_lanes", []) or []),
        "temporary_boundaries": list(getattr(local_map, "temporary_boundaries", []) or []),
        }
    )


def _sample_waypoints(waypoints: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    if not waypoints:
        return []
    sampled = list(waypoints[::_WAYPOINT_STRIDE])
    if sampled and sampled[-1] is not waypoints[-1]:
        sampled.append(waypoints[-1])
    return sampled


def _serialize_ego_trajectory_for_dashboard(trajectory: Any) -> dict[str, Any] | None:
    if trajectory is None:
        return None
    return serialize(
        {
            "waypoints": _sample_waypoints(list(getattr(trajectory, "waypoints", []) or [])),
            "cost": getattr(trajectory, "cost", 0.0),
            "behavior_state": getattr(
                getattr(trajectory, "behavior_state", None),
                "value",
                getattr(trajectory, "behavior_state", ""),
            ),
        }
    )


def _serialize_candidates_for_dashboard(candidates: Any) -> list[dict[str, Any]]:
    if not candidates:
        return []
    serialized: list[dict[str, Any]] = []
    for candidate in list(candidates):
        serialized.append(
            {
                "trajectory": _serialize_ego_trajectory_for_dashboard(getattr(candidate, "trajectory", None)),
                "lane_id": getattr(candidate, "lane_id", ""),
                "target_speed_mps": getattr(candidate, "target_speed_mps", 0.0),
                "score": getattr(candidate, "score", 0.0),
            }
        )
    return serialize(serialized)


def _serialize_predictions_for_dashboard(predictions: Any) -> list[dict[str, Any]]:
    if not predictions:
        return []
    serialized: list[dict[str, Any]] = []
    for prediction in list(predictions):
        serialized.append(
            {
                "track_id": getattr(prediction, "track_id", -1),
                "object_class": getattr(getattr(prediction, "object_class", None), "value", getattr(prediction, "object_class", "")),
                "predicted_trajectory": _sample_waypoints(list(getattr(prediction, "predicted_trajectory", []) or [])),
                "confidence_by_step": list(getattr(prediction, "confidence_by_step", []) or [])[::_WAYPOINT_STRIDE],
                "covariance_by_step": list(getattr(prediction, "covariance_by_step", []) or [])[::_WAYPOINT_STRIDE],
            }
        )
    return serialize(serialized)


def _serialize_bev_grid(bev_grid: Any) -> dict[str, Any] | None:
    if not isinstance(bev_grid, dict) or not isinstance(bev_grid.get("grid"), np.ndarray):
        return None
    grid = np.asarray(bev_grid["grid"], dtype=np.uint8)
    return {
        "grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"),
        "rows": int(bev_grid["rows"]),
        "cols": int(bev_grid["cols"]),
        "cell_size_m": float(bev_grid["cell_size_m"]),
        "x_min_m": float(bev_grid["x_min_m"]),
        "x_max_m": float(bev_grid["x_max_m"]),
        "y_min_m": float(bev_grid["y_min_m"]),
        "y_max_m": float(bev_grid["y_max_m"]),
    }


def _split_world_layer(world_layer: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(world_layer, dict):
        return None, None
    static_payload = {
        key: world_layer[key]
        for key in _WORLD_LAYER_STATIC_KEYS
        if key in world_layer
    }
    dynamic_payload = {
        key: world_layer[key]
        for key in _WORLD_LAYER_DYNAMIC_KEYS
        if key in world_layer
    }
    return static_payload or None, dynamic_payload or None


class WebSocketBridge:
    """Streams pipeline state over WebSocket to browser clients."""

    def __init__(self) -> None:
        self._event_bus: Any | None = None
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_bootstrap_text: str | None = None
        self._pending_static_text: str | None = None
        self._pending_dynamic_text: str | None = None
        self._send_active = False
        self._send_lock = threading.Lock()
        self._bootstrap_needed = False
        self._last_dynamic_monotonic = 0.0
        self._retained_hashes: dict[str, str] = {}
        self._retained_payloads: dict[str, Any] = {}
        self._heavy_hashes: dict[str, str] = {}
        self._heavy_last_sent_monotonic: dict[str, float] = {}
        self._stats_last_log_monotonic = 0.0

    def attach(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        event_bus.subscribe(TopicName.TICK_COMPLETE.value, self._on_tick_complete)
        logger.info("WebSocketBridge attached to event bus")

    def flush(self) -> None:
        logger.info("WebSocketBridge flush â€” %d clients connected", len(self._clients))

    def register(self, ws: Any) -> None:
        self._clients.add(ws)
        with self._send_lock:
            self._bootstrap_needed = True
        logger.info("WebSocket client connected (%d total)", len(self._clients))
        self._queue_bootstrap_from_latest_state()

    def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def _snapshot(self) -> dict[str, Any]:
        if self._event_bus is None:
            return {}
        if hasattr(self._event_bus, "latest_topics"):
            return self._event_bus.latest_topics(_WS_TOPICS)
        return self._event_bus.snapshot()

    def _latest_tick_payload(self) -> dict[str, Any]:
        if self._event_bus is None:
            return {}
        if hasattr(self._event_bus, "latest_topics"):
            latest = self._event_bus.latest_topics({TopicName.TICK_COMPLETE.value})
            payload = latest.get(TopicName.TICK_COMPLETE.value)
            return payload if isinstance(payload, dict) else {}
        snapshot = self._event_bus.snapshot()
        payload = snapshot.get(TopicName.TICK_COMPLETE.value)
        return payload if isinstance(payload, dict) else {}

    def _prepare_snapshot_topics(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        retained: dict[str, Any] = {}
        dynamic: dict[str, Any] = {}
        heavy: dict[str, Any] = {}

        scenario_info = snapshot.get(TopicName.SCENARIO_INFO.value)
        if scenario_info is not None:
            retained[TopicName.SCENARIO_INFO.value] = _serialize_payload(scenario_info)

        local_map = snapshot.get(TopicName.MAP_LOCAL_MAP.value)
        serialized_map = _serialize_local_map_for_dashboard(local_map)
        if serialized_map is not None:
            retained[TopicName.MAP_LOCAL_MAP.value] = serialized_map

        world_layer = snapshot.get(TopicName.VISUALIZATION_WORLD_LAYER.value)
        static_world_layer, dynamic_world_layer = _split_world_layer(world_layer)
        if static_world_layer is not None:
            retained[TopicName.VISUALIZATION_WORLD_LAYER.value] = static_world_layer
        if dynamic_world_layer is not None:
            dynamic[TopicName.VISUALIZATION_WORLD_LAYER.value] = dynamic_world_layer

        for topic in (
            TopicName.LOCALIZATION_EGO_POSE.value,
            TopicName.PERCEPTION_DETECTIONS.value,
            TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
            TopicName.PERCEPTION_STATUS.value,
            TopicName.CONTROL_VEHICLE_COMMAND.value,
            TopicName.PIPELINE_LATENCY.value,
        ):
            payload = snapshot.get(topic)
            if payload is not None:
                dynamic[topic] = _serialize_payload(payload)

        trajectory = snapshot.get(TopicName.PLANNING_EGO_TRAJECTORY.value)
        serialized_trajectory = _serialize_ego_trajectory_for_dashboard(trajectory)
        if serialized_trajectory is not None:
            dynamic[TopicName.PLANNING_EGO_TRAJECTORY.value] = serialized_trajectory

        overlay_frame = snapshot.get(TopicName.VISUALIZATION_CAMERA_OVERLAY.value)
        if isinstance(overlay_frame, np.ndarray):
            jpeg_b64 = _encode_overlay_jpeg(overlay_frame)
            if jpeg_b64:
                heavy[TopicName.VISUALIZATION_CAMERA_OVERLAY.value] = jpeg_b64

        lidar_frame = snapshot.get(TopicName.SENSOR_LIDAR.value)
        if lidar_frame is not None and hasattr(lidar_frame, "points_xyz"):
            heavy[TopicName.VISUALIZATION_LIDAR_PREVIEW.value] = {
                "points": _downsample_lidar(lidar_frame.points_xyz),
            }

        bev_grid = _serialize_bev_grid(snapshot.get(TopicName.VISUALIZATION_BEV_DRIVABLE.value))
        if bev_grid is not None:
            heavy[TopicName.VISUALIZATION_BEV_DRIVABLE.value] = bev_grid

        predictions = snapshot.get(TopicName.PREDICTION_AGENTS.value)
        if predictions is not None:
            heavy[TopicName.PREDICTION_AGENTS.value] = _serialize_predictions_for_dashboard(predictions)

        candidates = snapshot.get(TopicName.PLANNING_CANDIDATES.value)
        if candidates is not None:
            heavy[TopicName.PLANNING_CANDIDATES.value] = _serialize_candidates_for_dashboard(candidates)

        return retained, dynamic, heavy

    def _update_retained_cache(self, retained_topics: dict[str, Any]) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        for topic, payload in retained_topics.items():
            payload_hash = _hash_payload(payload)
            if self._retained_hashes.get(topic) != payload_hash:
                changed[topic] = payload
                self._retained_hashes[topic] = payload_hash
            self._retained_payloads[topic] = payload
        return changed

    def _collect_heavy_updates(self, heavy_topics: dict[str, Any], now_monotonic: float) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        min_interval_s = 1.0 / _HEAVY_DYNAMIC_FPS
        for topic, payload in heavy_topics.items():
            payload_hash = _hash_payload(payload)
            last_sent = self._heavy_last_sent_monotonic.get(topic, 0.0)
            if topic in {
                TopicName.PREDICTION_AGENTS.value,
                TopicName.PLANNING_CANDIDATES.value,
            }:
                should_send = payload_hash != self._heavy_hashes.get(topic) or (now_monotonic - last_sent) >= min_interval_s
            else:
                should_send = payload_hash != self._heavy_hashes.get(topic) and (
                    last_sent <= 0.0 or (now_monotonic - last_sent) >= min_interval_s
                )
            if not should_send:
                continue
            changed[topic] = payload
            self._heavy_hashes[topic] = payload_hash
            self._heavy_last_sent_monotonic[topic] = now_monotonic
        return changed

    def _mark_bootstrap_sent(self, retained_topics: dict[str, Any], heavy_topics: dict[str, Any], now_monotonic: float) -> None:
        for topic, payload in retained_topics.items():
            self._retained_hashes[topic] = _hash_payload(payload)
            self._retained_payloads[topic] = payload
        for topic, payload in heavy_topics.items():
            self._heavy_hashes[topic] = _hash_payload(payload)
            self._heavy_last_sent_monotonic[topic] = now_monotonic

    def _queue_bootstrap_from_latest_state(self) -> None:
        if self._event_bus is None:
            return
        now_monotonic = time.monotonic()
        snapshot = self._snapshot()
        retained_topics, dynamic_topics, heavy_topics = self._prepare_snapshot_topics(snapshot)
        bootstrap_topics: dict[str, Any] = {}
        for source_topics in (retained_topics, dynamic_topics, heavy_topics):
            for topic, topic_payload in source_topics.items():
                _merge_topic(bootstrap_topics, topic, topic_payload)
        tick_payload = self._latest_tick_payload()
        tick_id = tick_payload.get("tick_id", -1)
        sim_time_s = tick_payload.get("sim_time_s", 0.0)
        envelope, text = self._build_envelope("bootstrap", tick_id, sim_time_s, bootstrap_topics)
        with self._send_lock:
            self._bootstrap_needed = not bool(bootstrap_topics)
        self._mark_bootstrap_sent(retained_topics, heavy_topics, now_monotonic)
        self._last_dynamic_monotonic = now_monotonic
        self._log_ws_stats(envelope, now_monotonic)
        self._queue_text(message_kind="bootstrap", text=text)

    def _build_envelope(self, message_kind: str, tick_id: int, sim_time_s: float, topics: dict[str, Any]) -> tuple[dict[str, Any], str]:
        serialized_topics = serialize(topics)
        topic_bytes = {topic: _json_size_bytes(payload) for topic, payload in serialized_topics.items()}
        envelope = {
            "message_kind": message_kind,
            "tick_id": int(tick_id),
            "sim_time_s": float(sim_time_s),
            "topics": serialized_topics,
            "ws_stats": {
                "message_kind": message_kind,
                "total_bytes": 0,
                "topic_count": len(serialized_topics),
                "topic_bytes": topic_bytes,
            },
        }
        text = json.dumps(envelope, default=str)
        envelope["ws_stats"]["total_bytes"] = len(text.encode("utf-8"))
        text = json.dumps(envelope, default=str)
        return envelope, text

    def _log_ws_stats(self, envelope: dict[str, Any], now_monotonic: float) -> None:
        if now_monotonic - self._stats_last_log_monotonic < _WS_STATS_LOG_INTERVAL_S:
            return
        self._stats_last_log_monotonic = now_monotonic
        stats = envelope.get("ws_stats", {})
        topic_bytes = dict(stats.get("topic_bytes", {}))
        top_topics = sorted(topic_bytes.items(), key=lambda item: item[1], reverse=True)[:4]
        logger.info(
            "WS %s total=%s bytes topics=%s",
            stats.get("message_kind", "unknown"),
            stats.get("total_bytes", 0),
            top_topics,
        )

    def _queue_text(self, *, message_kind: str, text: str) -> None:
        should_schedule = False
        with self._send_lock:
            if message_kind == "bootstrap":
                self._pending_bootstrap_text = text
            elif message_kind == "static_update":
                self._pending_static_text = text
            else:
                self._pending_dynamic_text = text
            if not self._send_active:
                self._send_active = True
                should_schedule = True
        if should_schedule and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._drain_pending(), self._loop)

    def _next_pending_text(self) -> str | None:
        if self._pending_bootstrap_text is not None:
            text = self._pending_bootstrap_text
            self._pending_bootstrap_text = None
            return text
        if self._pending_dynamic_text is not None:
            text = self._pending_dynamic_text
            self._pending_dynamic_text = None
            return text
        if self._pending_static_text is not None:
            text = self._pending_static_text
            self._pending_static_text = None
            return text
        return None

    def _on_tick_complete(self, _topic: str, payload: Any) -> None:
        if not self._clients or self._event_bus is None:
            return

        now_monotonic = time.monotonic()
        tick_id = payload.get("tick_id", -1) if isinstance(payload, dict) else -1
        sim_time_s = payload.get("sim_time_s", 0.0) if isinstance(payload, dict) else 0.0
        snapshot = self._snapshot()
        retained_topics, dynamic_topics, heavy_topics = self._prepare_snapshot_topics(snapshot)

        with self._send_lock:
            bootstrap_needed = self._bootstrap_needed
            if bootstrap_needed:
                self._bootstrap_needed = False

        if bootstrap_needed:
            bootstrap_topics: dict[str, Any] = {}
            for source_topics in (retained_topics, dynamic_topics, heavy_topics):
                for topic, topic_payload in source_topics.items():
                    _merge_topic(bootstrap_topics, topic, topic_payload)
            envelope, text = self._build_envelope("bootstrap", tick_id, sim_time_s, bootstrap_topics)
            self._mark_bootstrap_sent(retained_topics, heavy_topics, now_monotonic)
            self._last_dynamic_monotonic = now_monotonic
            self._log_ws_stats(envelope, now_monotonic)
            self._queue_text(message_kind="bootstrap", text=text)
            return

        retained_updates = self._update_retained_cache(retained_topics)
        if retained_updates:
            envelope, text = self._build_envelope("static_update", tick_id, sim_time_s, retained_updates)
            self._log_ws_stats(envelope, now_monotonic)
            self._queue_text(message_kind="static_update", text=text)

        min_dynamic_interval_s = 1.0 / _FAST_DYNAMIC_FPS
        if (now_monotonic - self._last_dynamic_monotonic) < min_dynamic_interval_s:
            return

        dynamic_frame_topics: dict[str, Any] = {}
        for topic, topic_payload in dynamic_topics.items():
            _merge_topic(dynamic_frame_topics, topic, topic_payload)
        for topic, topic_payload in self._collect_heavy_updates(heavy_topics, now_monotonic).items():
            _merge_topic(dynamic_frame_topics, topic, topic_payload)

        if not dynamic_frame_topics:
            return
        envelope, text = self._build_envelope("dynamic_frame", tick_id, sim_time_s, dynamic_frame_topics)
        self._last_dynamic_monotonic = now_monotonic
        self._log_ws_stats(envelope, now_monotonic)
        self._queue_text(message_kind="dynamic_frame", text=text)

    async def _drain_pending(self) -> None:
        while True:
            with self._send_lock:
                text = self._next_pending_text()
                if text is None:
                    self._send_active = False
                    return
            await self._broadcast_once(text)

    async def _broadcast_once(self, text: str) -> None:
        stale: list[Any] = []
        for ws in self._clients:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket send timed out; dropping client")
                stale.append(ws)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)
