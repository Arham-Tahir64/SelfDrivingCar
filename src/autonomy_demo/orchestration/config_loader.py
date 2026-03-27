from __future__ import annotations

import os
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
        output_dir=Path(os.getenv("AUTONOMY_DEMO_OUTPUT_DIR", runtime.get("output_dir", "outputs"))),
        log_level=str(os.getenv("AUTONOMY_DEMO_LOG_LEVEL", runtime.get("log_level", "INFO"))),
        record_replay=bool(runtime.get("record_replay", True)),
        enable_visualization=bool(runtime.get("enable_visualization", True)),
        weather_preset=str(runtime.get("weather_preset", "ClearNoon")),
        carla_host=str(os.getenv("CARLA_HOST", carla.get("host", "127.0.0.1"))),
        carla_port=int(os.getenv("CARLA_PORT", carla.get("port", 2000))),
        carla_timeout_s=float(os.getenv("CARLA_TIMEOUT_S", carla.get("timeout_s", 10.0))),
        carla_sync_fps=int(carla.get("sync_fps", runtime.get("tick_hz", 20))),
        carla_root=Path(os.getenv("CARLA_ROOT", carla.get("carla_root", ""))),
        carla_python_api_wheel=Path(
            os.getenv("CARLA_PYTHON_API_WHEEL", carla.get("python_api_wheel", ""))
        ),
        carla_launch_executable=Path(
            os.getenv("CARLA_LAUNCH_EXECUTABLE", carla.get("launch_executable", ""))
        ),
        town=str(carla.get("town", "Town04")),
        ego_vehicle_blueprint=str(carla.get("ego_vehicle_blueprint", "vehicle.tesla.model3")),
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
