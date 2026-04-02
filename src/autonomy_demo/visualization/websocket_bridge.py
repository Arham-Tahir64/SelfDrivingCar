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
        TopicName.PERCEPTION_LANES.value,
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
        TopicName.VISUALIZATION_ROAD_CORRIDOR.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PIPELINE_LATENCY.value,
    }
)

# Topics with large binary payloads that need special handling.
_SKIP_TOPICS = frozenset(
    {
        TopicName.SENSOR_CAMERA_FRONT.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PERCEPTION_DRIVABLE_SPACE.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
    }
)

_LIDAR_MAX_POINTS = 750
_JPEG_QUALITY = 45
_OVERLAY_MAX_WIDTH = 480
_MAX_WS_FPS = 5.0
_MAX_PREDICTION_AGENTS = 12
_MAX_CANDIDATES = 10
_WAYPOINT_STRIDE = 2


def _encode_overlay_jpeg(frame_rgb: np.ndarray) -> str | None:
    """Compress an RGB uint8 frame to base64-encoded JPEG."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    # Resize aggressively so the dashboard stays responsive on modest hardware.
    h, w = frame_rgb.shape[:2]
    if w > _OVERLAY_MAX_WIDTH:
        scale = _OVERLAY_MAX_WIDTH / w
        frame_rgb = cv2.resize(
            frame_rgb,
            (_OVERLAY_MAX_WIDTH, int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    success, buf = cv2.imencode(".jpg", frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not success:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _downsample_lidar(points_xyz: np.ndarray) -> list[list[float]]:
    """Downsample LiDAR points to a manageable count for the browser."""
    n = points_xyz.shape[0]
    if n == 0:
        return []
    if n > _LIDAR_MAX_POINTS:
        indices = np.random.choice(n, _LIDAR_MAX_POINTS, replace=False)
        points_xyz = points_xyz[indices]
    return points_xyz.tolist()


def _serialize_local_map_for_dashboard(local_map: Any) -> dict[str, Any] | None:
    if local_map is None:
        return None
    return {
        "closed_lanes": list(getattr(local_map, "closed_lanes", []) or []),
        "temporary_boundaries": list(getattr(local_map, "temporary_boundaries", []) or []),
        "perceived_lanes": list(getattr(local_map, "perceived_lanes", []) or []),
    }


def _sample_waypoints(waypoints: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    if not waypoints:
        return []
    sampled = list(waypoints[::_WAYPOINT_STRIDE])
    if sampled[-1] is not waypoints[-1]:
        sampled.append(waypoints[-1])
    return sampled


def _serialize_ego_trajectory_for_dashboard(trajectory: Any) -> dict[str, Any] | None:
    if trajectory is None:
        return None
    return {
        "waypoints": _sample_waypoints(list(getattr(trajectory, "waypoints", []) or [])),
        "cost": getattr(trajectory, "cost", 0.0),
        "behavior_state": getattr(getattr(trajectory, "behavior_state", None), "value", getattr(trajectory, "behavior_state", "")),
    }


def _serialize_candidates_for_dashboard(candidates: Any) -> list[dict[str, Any]]:
    if not candidates:
        return []
    trimmed = list(candidates)[:_MAX_CANDIDATES]
    serialized: list[dict[str, Any]] = []
    for candidate in trimmed:
        serialized.append(
            {
                "trajectory": _serialize_ego_trajectory_for_dashboard(getattr(candidate, "trajectory", None)),
                "lane_id": getattr(candidate, "lane_id", ""),
                "target_speed_mps": getattr(candidate, "target_speed_mps", 0.0),
                "score": getattr(candidate, "score", 0.0),
            }
        )
    return serialized


def _serialize_predictions_for_dashboard(predictions: Any) -> list[dict[str, Any]]:
    if not predictions:
        return []
    serialized: list[dict[str, Any]] = []
    for prediction in list(predictions)[:_MAX_PREDICTION_AGENTS]:
        serialized.append(
            {
                "track_id": getattr(prediction, "track_id", -1),
                "object_class": getattr(getattr(prediction, "object_class", None), "value", getattr(prediction, "object_class", "")),
                "predicted_trajectory": _sample_waypoints(list(getattr(prediction, "predicted_trajectory", []) or [])),
                "confidence_by_step": list(getattr(prediction, "confidence_by_step", []) or [])[::_WAYPOINT_STRIDE],
                "covariance_by_step": list(getattr(prediction, "covariance_by_step", []) or [])[::_WAYPOINT_STRIDE],
            }
        )
    return serialized


class WebSocketBridge:
    """Streams pipeline state over WebSocket to browser clients.

    Implements the ``VisualizationSink`` protocol (``attach`` / ``flush``).
    """

    def __init__(self) -> None:
        self._event_bus: Any | None = None
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_text: str | None = None
        self._broadcast_active = False
        self._broadcast_lock = threading.Lock()
        self._last_broadcast_monotonic = 0.0

    # -- VisualizationSink protocol ------------------------------------------

    def attach(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        event_bus.subscribe(TopicName.TICK_COMPLETE.value, self._on_tick_complete)
        logger.info("WebSocketBridge attached to event bus")

    def flush(self) -> None:
        logger.info("WebSocketBridge flush — %d clients connected", len(self._clients))

    # -- Client management (called from asyncio server thread) ---------------

    def register(self, ws: Any) -> None:
        self._clients.add(ws)
        logger.info("WebSocket client connected (%d total)", len(self._clients))

    def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- Internal ------------------------------------------------------------

    def _on_tick_complete(self, _topic: str, payload: Any) -> None:
        if not self._clients or self._event_bus is None:
            return
        now = time.monotonic()
        min_interval_s = 1.0 / _MAX_WS_FPS
        if (now - self._last_broadcast_monotonic) < min_interval_s:
            return
        self._last_broadcast_monotonic = now
        if hasattr(self._event_bus, "latest_topics"):
            snapshot = self._event_bus.latest_topics(_WS_TOPICS)
        else:
            snapshot = self._event_bus.snapshot()
        filtered = {
            k: v
            for k, v in snapshot.items()
            if k not in _SKIP_TOPICS and k in _WS_TOPICS
        }
        local_map = snapshot.get(TopicName.MAP_LOCAL_MAP.value)
        if local_map is not None:
            filtered[TopicName.MAP_LOCAL_MAP.value] = _serialize_local_map_for_dashboard(local_map)
        predictions = snapshot.get(TopicName.PREDICTION_AGENTS.value)
        if predictions is not None:
            filtered[TopicName.PREDICTION_AGENTS.value] = _serialize_predictions_for_dashboard(predictions)
        trajectory = snapshot.get(TopicName.PLANNING_EGO_TRAJECTORY.value)
        if trajectory is not None:
            filtered[TopicName.PLANNING_EGO_TRAJECTORY.value] = _serialize_ego_trajectory_for_dashboard(trajectory)
        candidates = snapshot.get(TopicName.PLANNING_CANDIDATES.value)
        if candidates is not None:
            filtered[TopicName.PLANNING_CANDIDATES.value] = _serialize_candidates_for_dashboard(candidates)
        tick_id = payload.get("tick_id", -1) if isinstance(payload, dict) else -1
        sim_time_s = payload.get("sim_time_s", 0.0) if isinstance(payload, dict) else 0.0
        message = {
            "tick_id": tick_id,
            "sim_time_s": sim_time_s,
            **serialize(filtered),
        }

        # Inject compressed camera overlay (from NullVisualizationService)
        overlay_frame = snapshot.get(TopicName.VISUALIZATION_CAMERA_OVERLAY.value)
        if isinstance(overlay_frame, np.ndarray):
            jpeg_b64 = _encode_overlay_jpeg(overlay_frame)
            if jpeg_b64:
                message[TopicName.VISUALIZATION_CAMERA_OVERLAY.value] = jpeg_b64

        # Inject downsampled LiDAR preview
        lidar_frame = snapshot.get(TopicName.SENSOR_LIDAR.value)
        if lidar_frame is not None and hasattr(lidar_frame, "points_xyz"):
            message[TopicName.VISUALIZATION_LIDAR_PREVIEW.value] = {
                "points": _downsample_lidar(lidar_frame.points_xyz),
            }

        # Inject BEV drivable grid as base64
        bev_grid = snapshot.get(TopicName.VISUALIZATION_BEV_DRIVABLE.value)
        if isinstance(bev_grid, dict) and isinstance(bev_grid.get("grid"), np.ndarray):
            message[TopicName.VISUALIZATION_BEV_DRIVABLE.value] = {
                "grid_b64": base64.b64encode(bev_grid["grid"].tobytes()).decode("ascii"),
                "rows": int(bev_grid["rows"]),
                "cols": int(bev_grid["cols"]),
                "cell_size_m": float(bev_grid["cell_size_m"]),
                "x_min_m": float(bev_grid["x_min_m"]),
                "x_max_m": float(bev_grid["x_max_m"]),
                "y_min_m": float(bev_grid["y_min_m"]),
                "y_max_m": float(bev_grid["y_max_m"]),
            }

        # Include pipeline latency if present
        latency = snapshot.get(TopicName.PIPELINE_LATENCY.value)
        if latency is not None:
            message[TopicName.PIPELINE_LATENCY.value] = serialize(latency)

        try:
            text = json.dumps(message, default=str)
        except Exception:
            logger.exception("Failed to serialize tick %s", tick_id)
            return
        if self._loop is not None:
            should_schedule = False
            with self._broadcast_lock:
                self._pending_text = text
                if not self._broadcast_active:
                    self._broadcast_active = True
                    should_schedule = True
            if should_schedule:
                asyncio.run_coroutine_threadsafe(self._drain_pending(), self._loop)

    async def _drain_pending(self) -> None:
        while True:
            with self._broadcast_lock:
                text = self._pending_text
                self._pending_text = None
                if text is None:
                    self._broadcast_active = False
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
