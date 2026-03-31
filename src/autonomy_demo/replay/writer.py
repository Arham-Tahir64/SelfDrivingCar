from __future__ import annotations

import json
from pathlib import Path

from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.common.serialization import serialize as _serialize
from autonomy_demo.interfaces.types import ReplayFrame


class Hdf5OrJsonReplayWriter:
    """HDF5-oriented scaffold with JSON fallback until a full binary schema lands."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.frames: list[ReplayFrame] = []
        self.path = self.output_dir / "replay.json"

    def record(self, frame: ReplayFrame) -> None:
        self.frames.append(frame)

    def finalize(self) -> Path:
        payload = [_serialize(frame) for frame in self.frames]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

