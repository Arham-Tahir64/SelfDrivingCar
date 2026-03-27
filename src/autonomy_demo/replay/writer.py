from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.interfaces.types import ReplayFrame


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


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

