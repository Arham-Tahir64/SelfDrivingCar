from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from autonomy_demo.common.exceptions import ConfigurationError
from autonomy_demo.interfaces.types import RuntimeConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_runtime_config(app_config_path: Path) -> RuntimeConfig:
    data = _load_yaml(app_config_path)
    runtime = data.get("runtime", {})
    carla = data.get("carla", {})
    evaluation = data.get("evaluation", {})
    if not runtime:
        raise ConfigurationError("runtime config is required")
    return RuntimeConfig(
        backend=runtime.get("backend", "stub"),
        tick_hz=int(runtime.get("tick_hz", 20)),
        max_ticks=int(runtime.get("max_ticks", 5)),
        output_dir=Path(runtime.get("output_dir", "outputs")),
        log_level=str(runtime.get("log_level", "INFO")),
        record_replay=bool(runtime.get("record_replay", True)),
        enable_visualization=bool(runtime.get("enable_visualization", True)),
        weather_preset=str(runtime.get("weather_preset", "ClearNoon")),
        carla_host=str(carla.get("host", "127.0.0.1")),
        carla_port=int(carla.get("port", 2000)),
        town=str(carla.get("town", "Town04")),
        latency_budget_ms={
            key: float(value) for key, value in (evaluation.get("latency_budget_ms", {}) or {}).items()
        },
    )


def load_sensor_config(sensor_config_path: Path) -> dict[str, Any]:
    data = _load_yaml(sensor_config_path)
    sensors = data.get("sensors")
    if not isinstance(sensors, dict) or not sensors:
        raise ConfigurationError("sensor definitions are required")
    return data

