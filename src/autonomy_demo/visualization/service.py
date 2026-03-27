from __future__ import annotations

from collections import deque
from typing import Any

from autonomy_demo.common.logging import get_logger


class NullVisualizationService:
    """Read-only subscriber that captures the latest events without affecting the pipeline."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.logger = get_logger(__name__, enabled=enabled)
        self.events: deque[tuple[str, Any]] = deque(maxlen=32)

    def attach(self, event_bus) -> None:
        if not self.enabled:
            return
        event_bus.subscribe("*", self._handle)
        self.logger.info("Visualization subscriber attached")

    def _handle(self, topic: str, payload: Any) -> None:
        self.events.append((topic, payload))

    def flush(self) -> None:
        if self.enabled:
            self.logger.info("Visualization captured %s events", len(self.events))

