from __future__ import annotations

import asyncio
import json
from typing import Any

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.serialization import serialize
from autonomy_demo.interfaces.enums import TopicName

logger = get_logger(__name__)

# Topics with large binary payloads that the BEV frontend does not need.
_SKIP_TOPICS = frozenset(
    {
        TopicName.SENSOR_CAMERA_FRONT.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PERCEPTION_DRIVABLE_SPACE.value,
    }
)


class WebSocketBridge:
    """Streams pipeline state over WebSocket to browser clients.

    Implements the ``VisualizationSink`` protocol (``attach`` / ``flush``).
    """

    def __init__(self) -> None:
        self._event_bus: Any | None = None
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

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
        try:
            text = json.dumps(message, default=str)
        except Exception:
            logger.exception("Failed to serialize tick %s", tick_id)
            return
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(text), self._loop)

    async def _broadcast(self, text: str) -> None:
        stale: list[Any] = []
        for ws in self._clients:
            try:
                await ws.send_text(text)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)
