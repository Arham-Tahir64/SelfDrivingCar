from __future__ import annotations

import json
from pathlib import Path

from autonomy_demo.interfaces.types import ReplayFrame


class JsonReplayReader:
    """Simple scaffold reader for replay artifacts emitted by the placeholder writer."""

    def __init__(self, replay_path: Path) -> None:
        self.replay_path = replay_path

    def read(self) -> list[ReplayFrame]:
        data = json.loads(self.replay_path.read_text(encoding="utf-8"))
        return [ReplayFrame(**item) for item in data]

