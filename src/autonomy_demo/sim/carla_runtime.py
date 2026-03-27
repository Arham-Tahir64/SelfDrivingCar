from __future__ import annotations

import importlib
import platform
import sys
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.common.paths import ensure_directory, repo_root


def _extracted_wheel_root(python_api_wheel: Path) -> Path:
    cache_root = ensure_directory(repo_root() / ".carla_api_cache")
    stat = python_api_wheel.stat()
    fingerprint = sha256(
        f"{python_api_wheel.resolve()}::{stat.st_mtime_ns}::{stat.st_size}".encode("utf-8")
    ).hexdigest()[:12]
    return cache_root / f"{python_api_wheel.stem}-{fingerprint}"


def _extract_wheel_if_needed(python_api_wheel: Path) -> Path:
    target_dir = _extracted_wheel_root(python_api_wheel)
    package_init = target_dir / "carla" / "__init__.py"
    package_lib = target_dir / "carla" / "libcarla.cp312-win_amd64.pyd"
    if package_init.exists() and package_lib.exists():
        return target_dir
    ensure_directory(target_dir)
    with ZipFile(python_api_wheel) as archive:
        archive.extractall(target_dir)
    if not package_init.exists() or not package_lib.exists():
        raise CarlaRuntimeError(
            f"Extracted CARLA wheel is incomplete at {target_dir}"
        )
    return target_dir


def ensure_carla_importable(python_api_wheel: Path):
    """Lazily prepare the CARLA Python API for live runtime use."""

    if sys.version_info[:2] != (3, 12):
        current = platform.python_version()
        raise CarlaRuntimeError(
            f"CARLA 0.9.16 live mode requires Python 3.12; current interpreter is {current}."
        )
    if not python_api_wheel.exists():
        raise CarlaRuntimeError(
            f"CARLA Python API wheel not found: {python_api_wheel}"
        )
    extracted_root = _extract_wheel_if_needed(python_api_wheel)
    extracted_path = str(extracted_root)
    if extracted_path not in sys.path:
        sys.path.insert(0, extracted_path)
    sys.modules.pop("carla", None)
    sys.modules.pop("carla.libcarla", None)
    try:
        return importlib.import_module("carla")
    except ImportError as exc:
        raise CarlaRuntimeError(
            f"Unable to import carla from extracted wheel at {extracted_root}: {exc}"
        ) from exc


def weather_from_name(carla_module, weather_name: str):
    """Map configured preset names onto CARLA weather presets."""

    presets = {
        "clearnoon": carla_module.WeatherParameters.ClearNoon,
        "cloudysunset": carla_module.WeatherParameters.CloudySunset,
        "wetcloudy": carla_module.WeatherParameters.WetCloudyNoon,
        "wetcloudynoon": carla_module.WeatherParameters.WetCloudyNoon,
    }
    try:
        return getattr(carla_module.WeatherParameters, weather_name)
    except AttributeError:
        key = weather_name.replace(" ", "").lower()
        return presets.get(key, carla_module.WeatherParameters.ClearNoon)


@dataclass(slots=True)
class CarlaSessionState:
    carla: Any | None = None
    client: Any | None = None
    world: Any | None = None
    blueprint_library: Any | None = None
    ego_actor: Any | None = None
    npc_actors: list[Any] = field(default_factory=list)
    prop_actors: list[Any] = field(default_factory=list)
    sensor_actors: dict[str, Any] = field(default_factory=dict)
    original_settings: Any | None = None
    current_snapshot: Any | None = None
    current_frame: int | None = None
