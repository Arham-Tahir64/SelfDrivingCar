from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def serialize(value: Any) -> Any:
    """Recursively serialize dataclasses, numpy arrays, and enums to JSON-safe types."""
    if is_dataclass(value):
        return {key: serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value
