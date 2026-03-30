from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.orchestration.config_loader import load_runtime_config, load_sensor_config
from autonomy_demo.orchestration.scenario_runner import ScenarioRunner
from autonomy_demo.sensors.carla_sensor_suite import CarlaSensorSuite
from autonomy_demo.sim.backends import CarlaSimulationBackend, StubSimulationBackend
from autonomy_demo.sim.carla_runtime import ensure_carla_importable


def test_missing_carla_wheel_raises() -> None:
    with pytest.raises(CarlaRuntimeError):
        ensure_carla_importable(Path("C:/missing/carla.whl"))


def test_runtime_component_selection_stub() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    sensor_config = load_sensor_config(Path("configs/sensors.default.yaml"))
    runner = ScenarioRunner(runtime, sensor_config, Path("outputs/tests"))
    backend, sensors = runner._build_runtime_components()
    assert isinstance(backend, StubSimulationBackend)
    assert not isinstance(sensors, CarlaSensorSuite)


def test_runtime_component_selection_carla() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    runtime.backend = "carla"
    sensor_config = load_sensor_config(Path("configs/sensors.default.yaml"))
    runner = ScenarioRunner(runtime, sensor_config, Path("outputs/tests"))
    backend, sensors = runner._build_runtime_components()
    assert isinstance(backend, CarlaSimulationBackend)
    assert isinstance(sensors, CarlaSensorSuite)


class _FakeImage:
    def __init__(self) -> None:
        self.width = 2
        self.height = 1
        self.timestamp = 1.5
        self.frame = 7
        self.raw_data = bytes([10, 20, 30, 255, 40, 50, 60, 255])


class _FakeLidar:
    def __init__(self) -> None:
        self.timestamp = 2.0
        self.frame = 8
        self.raw_data = np.array([1.0, 2.0, 3.0, 0.4, 4.0, 5.0, 6.0, 0.9], dtype=np.float32).tobytes()


class _FakeRadarDetection:
    def __init__(self, depth: float, velocity: float, azimuth: float, altitude: float) -> None:
        self.depth = depth
        self.velocity = velocity
        self.azimuth = azimuth
        self.altitude = altitude


class _FakeRadar:
    def __init__(self) -> None:
        self.timestamp = 2.5
        self.frame = 9
        self._items = [_FakeRadarDetection(10.0, 2.0, 0.1, 0.05)]

    def __iter__(self):
        return iter(self._items)


class _Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeImu:
    def __init__(self) -> None:
        self.timestamp = 3.0
        self.frame = 10
        self.accelerometer = _Vec3(0.1, 0.2, 0.3)
        self.gyroscope = _Vec3(0.4, 0.5, 0.6)


class _Backend:
    def __init__(self) -> None:
        self.runtime_config = type("Cfg", (), {"carla_timeout_s": 1.0, "carla_sync_fps": 20})()
        self.state = type("State", (), {"carla": None, "world": None, "ego_actor": None})()


class _FakeLocation:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeRotation:
    def __init__(self, yaw: float) -> None:
        self.yaw = yaw


class _FakeTransform:
    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.location = _FakeLocation(x, y, 0.5)
        self.rotation = _FakeRotation(yaw)


def test_carla_sensor_converters() -> None:
    suite = CarlaSensorSuite({"sensors": {}}, _Backend())
    camera = suite._camera_frame("front_camera", _FakeImage(), current_frame=7)
    lidar = suite._lidar_frame(_FakeLidar())
    radar = suite._radar_frame(_FakeRadar())
    imu = suite._imu_reading(_FakeImu())
    gnss = suite._gnss_reading(type("Gnss", (), {"timestamp": 4.0, "frame": 11})(), _Vec3(1.0, 2.0, 3.0))
    assert camera.frame.shape == (1, 2, 3)
    assert lidar.points_xyz.shape == (2, 3)
    assert radar.detections.shape == (1, 4)
    assert imu.acceleration_xyz.shape == (3,)
    assert gnss.world_xyz.tolist() == [1.0, 2.0, 3.0]


def test_spawn_candidate_score_prefers_goal_aligned_transform() -> None:
    aligned = _FakeTransform(12.0, -18.0, 90.0)
    wrong_way = _FakeTransform(12.0, -18.0, -90.0)
    aligned_score = CarlaSimulationBackend._spawn_candidate_score(
        transform=aligned,
        requested_xy=(15.0, -20.0),
        goal_xy=(15.0, 80.0),
    )
    wrong_way_score = CarlaSimulationBackend._spawn_candidate_score(
        transform=wrong_way,
        requested_xy=(15.0, -20.0),
        goal_xy=(15.0, 80.0),
    )
    assert aligned_score < wrong_way_score
