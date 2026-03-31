from __future__ import annotations

from collections import defaultdict
from typing import Any

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.interfaces.types import SensorFrameBundle


class SensorSynchronizer:
    """Placeholder for PRD Phase 2 synchronization logic."""

    def align(self, bundle: SensorFrameBundle) -> SensorFrameBundle:
        return bundle


class FrameBucketSynchronizer:
    """Collect per-frame payloads and release a bundle only when all required entries exist."""

    def __init__(self, required_names: tuple[str, ...]) -> None:
        self.required_names = required_names
        self._frames: dict[int, dict[str, Any]] = defaultdict(dict)

    def push(self, sensor_name: str, payload: Any) -> None:
        self._frames[int(payload.frame)][sensor_name] = payload

    def pop_complete(self, frame_id: int) -> dict[str, Any]:
        payloads = self._frames.get(frame_id, {})
        missing = [name for name in self.required_names if name not in payloads]
        if missing:
            raise CarlaRuntimeError(
                f"Frame {frame_id} is incomplete, missing sensors: {', '.join(missing)}"
            )
        return self._frames.pop(frame_id)
