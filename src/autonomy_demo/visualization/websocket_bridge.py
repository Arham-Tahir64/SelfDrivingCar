from __future__ import annotations

import asyncio
import base64
import json
import threading
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.serialization import serialize
from autonomy_demo.interfaces.enums import TopicName

logger = get_logger(__name__)

# Topics with large binary payloads that need special handling.
_SKIP_TOPICS = frozenset(
    {
        TopicName.SENSOR_CAMERA_FRONT.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PERCEPTION_DRIVABLE_SPACE.value,
    }
)

_LIDAR_MAX_POINTS = 2000
_JPEG_QUALITY = 65


def _encode_overlay_jpeg(frame_rgb: np.ndarray) -> str | None:
    """Compress an RGB uint8 frame to base64-encoded JPEG."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    # Resize to keep payload small (max 640px wide)
    h, w = frame_rgb.shape[:2]
    if w > 640:
        scale = 640 / w
        frame_rgb = cv2.resize(frame_rgb, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
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
        snapshot = self._event_bus.snapshot()
        filtered = {k: v for k, v in snapshot.items() if k not in _SKIP_TOPICS}
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
                await asyncio.wait_for(ws.send_text(text), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)
