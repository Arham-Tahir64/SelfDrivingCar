from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.interfaces.types import Point2D, Pose2D, ScenarioNpcConfig
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
    def __init__(self, x: float | _FakeLocation, y: float | _FakeRotation, yaw: float | None = None) -> None:
        if isinstance(x, _FakeLocation):
            self.location = x
            self.rotation = y if isinstance(y, _FakeRotation) else _FakeRotation(0.0)
            return
        if yaw is None:
            raise ValueError("yaw is required when constructing from coordinates")
        self.location = _FakeLocation(float(x), float(y), 0.5)
        self.rotation = _FakeRotation(float(yaw))


class _FakeVelocity:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _FakeVehicleControl:
    def __init__(self, **kwargs) -> None:
        self.throttle = kwargs.get("throttle", 0.0)
        self.steer = kwargs.get("steer", 0.0)
        self.brake = kwargs.get("brake", 0.0)
        self.hand_brake = kwargs.get("hand_brake", False)
        self.reverse = kwargs.get("reverse", False)


class _FakeLaneType:
    Driving = "Driving"


class _FakeCarla:
    Transform = _FakeTransform
    Location = _FakeLocation
    Rotation = _FakeRotation
    VehicleControl = _FakeVehicleControl
    LaneType = _FakeLaneType


class _FakeWaypoint:
    def __init__(self, transform: _FakeTransform, next_waypoints: list["_FakeWaypoint"] | None = None) -> None:
        self.transform = transform
        self._next_waypoints = list(next_waypoints or [])

    def next(self, distance_m: float):
        del distance_m
        return list(self._next_waypoints)


class _FakeMap:
    def __init__(
        self,
        *,
        road_waypoint: _FakeWaypoint | None,
        spawn_points: list[_FakeTransform],
        waypoints_by_xy: dict[tuple[float, float], _FakeWaypoint] | None = None,
    ) -> None:
        self._road_waypoint = road_waypoint
        self._spawn_points = spawn_points
        self._waypoints_by_xy = {
            (round(float(x), 2), round(float(y), 2)): waypoint
            for (x, y), waypoint in (waypoints_by_xy or {}).items()
        }

    def get_spawn_points(self) -> list[_FakeTransform]:
        return list(self._spawn_points)

    def get_waypoint(self, requested_location, project_to_road: bool, lane_type):
        del project_to_road, lane_type
        key = (round(float(requested_location.x), 2), round(float(requested_location.y), 2))
        if key in self._waypoints_by_xy:
            return self._waypoints_by_xy[key]
        return self._road_waypoint


class _FakeBlueprintLibrary:
    def find(self, blueprint_id: str):
        return blueprint_id


class _FakeWorld:
    def __init__(self, map_obj: _FakeMap) -> None:
        self._map = map_obj
        self.spawned_transforms: list[_FakeTransform] = []

    def get_map(self) -> _FakeMap:
        return self._map

    def try_spawn_actor(self, blueprint, transform):
        del blueprint
        self.spawned_transforms.append(transform)
        return _FakeNpcActor(
            x=float(transform.location.x),
            y=float(transform.location.y),
            yaw=float(transform.rotation.yaw),
        )


class _FakeNpcActor:
    def __init__(self, *, x: float, y: float, yaw: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self._transform = _FakeTransform(x, y, yaw)
        self._velocity = _FakeVelocity(vx, vy)
        self.last_control = None
        self.is_alive = True

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def apply_control(self, control) -> None:
        self.last_control = control


class _FakeExistingActor:
    def __init__(self, type_id: str) -> None:
        self.type_id = type_id
        self.destroyed = False
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def destroy(self) -> None:
        self.destroyed = True


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


def test_npc_motion_plan_accelerates_toward_route_target() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    backend = CarlaSimulationBackend(runtime)
    backend.state.carla = type("Carla", (), {"VehicleControl": _FakeVehicleControl})()
    actor = _FakeNpcActor(x=0.0, y=0.0, yaw=0.0)
    plan = type(
        "Plan",
        (),
        {
            "actor": actor,
            "behavior": "cross_traffic",
            "route_xy": [(20.0, 0.0)],
            "waypoint_index": 0,
            "target_speed_mps": 6.0,
        },
    )()
    backend._apply_npc_motion_plan(plan)
    assert actor.last_control is not None
    assert actor.last_control.throttle > 0.0
    assert abs(actor.last_control.steer) < 0.05
    assert actor.last_control.brake == 0.0


def test_npc_target_speed_defaults_by_behavior() -> None:
    assert CarlaSimulationBackend._npc_target_speed_mps("cross_traffic") == pytest.approx(6.0)
    assert CarlaSimulationBackend._npc_target_speed_mps("parked") == pytest.approx(0.0)
    assert CarlaSimulationBackend._npc_target_speed_mps("highway_flow_fast") == pytest.approx(11.0)
    assert CarlaSimulationBackend._npc_target_speed_mps("highway_flow_aggressive") == pytest.approx(13.0)
    assert CarlaSimulationBackend._npc_target_speed_mps("highway_flow_speed_12.5") == pytest.approx(12.5)
    assert CarlaSimulationBackend._npc_target_speed_mps("default") == pytest.approx(8.0)


def test_clear_dynamic_actors_removes_stale_runtime_actors_only() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    backend = CarlaSimulationBackend(runtime)
    vehicle = _FakeExistingActor("vehicle.tesla.model3")
    sensor = _FakeExistingActor("sensor.camera.rgb")
    walker_controller = _FakeExistingActor("controller.ai.walker")
    traffic_light = _FakeExistingActor("traffic.traffic_light")
    backend.state.world = type(
        "World",
        (),
        {"get_actors": lambda self: [vehicle, sensor, walker_controller, traffic_light]},
    )()

    backend._clear_dynamic_actors()

    assert vehicle.destroyed is True
    assert sensor.destroyed is True
    assert walker_controller.destroyed is True
    assert traffic_light.destroyed is False


def test_spawn_scenario_actors_prefers_goal_aligned_lane_for_unset_npc_yaw() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    backend = CarlaSimulationBackend(runtime)
    backend.state.carla = _FakeCarla()
    backend.state.blueprint_library = _FakeBlueprintLibrary()
    backend.state.world = _FakeWorld(
        _FakeMap(
            road_waypoint=_FakeWaypoint(_FakeTransform(140.0, 52.0, 180.0)),
            spawn_points=[_FakeTransform(141.0, 52.0, 0.0)],
        )
    )

    scenario = type(
        "Scenario",
        (),
        {
            "npcs": [
                ScenarioNpcConfig(
                    model="vehicle.tesla.model3",
                    behavior="cruise",
                    spawn=Pose2D(x=140.0, y=52.0, z=0.0, yaw=0.0),
                    route=[Point2D(x=320.0, y=52.0, z=0.0)],
                )
            ],
            "props": [],
        },
    )()

    backend._spawn_scenario_actors(scenario)

    assert backend.state.world.spawned_transforms
    assert backend.state.world.spawned_transforms[0].rotation.yaw == pytest.approx(0.0)


def test_resolve_npc_route_xy_follows_lane_instead_of_cutting_to_adjacent_lane() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    backend = CarlaSimulationBackend(runtime)
    backend.state.carla = _FakeCarla()

    lane_wp_3 = _FakeWaypoint(_FakeTransform(18.0, 0.0, 0.0))
    lane_wp_2 = _FakeWaypoint(_FakeTransform(12.0, 0.0, 0.0), next_waypoints=[lane_wp_3])
    lane_wp_1 = _FakeWaypoint(_FakeTransform(6.0, 0.0, 0.0), next_waypoints=[lane_wp_2])
    start_wp = _FakeWaypoint(_FakeTransform(0.0, 0.0, 0.0), next_waypoints=[lane_wp_1])
    adjacent_goal_wp = _FakeWaypoint(_FakeTransform(18.0, 4.0, 0.0))

    backend.state.world = _FakeWorld(
        _FakeMap(
            road_waypoint=None,
            spawn_points=[],
            waypoints_by_xy={
                (0.0, 0.0): start_wp,
                (18.0, 0.0): adjacent_goal_wp,
                (18.0, 4.0): adjacent_goal_wp,
            },
        )
    )

    route_xy = backend._resolve_npc_route_xy(
        spawn_xy=(0.0, 0.0),
        route_points=[Point2D(x=18.0, y=0.0, z=0.0)],
    )

    assert route_xy == pytest.approx([(6.0, 0.0), (12.0, 0.0), (18.0, 0.0)])


def test_resolve_npc_route_xy_prefers_raw_goal_over_projected_turn_lane() -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    backend = CarlaSimulationBackend(runtime)
    backend.state.carla = _FakeCarla()

    straight_wp_2 = _FakeWaypoint(_FakeTransform(12.0, 0.0, 0.0))
    straight_wp_1 = _FakeWaypoint(_FakeTransform(6.0, 0.0, 0.0), next_waypoints=[straight_wp_2])
    right_turn_wp = _FakeWaypoint(_FakeTransform(6.0, -6.0, -90.0))
    start_wp = _FakeWaypoint(
        _FakeTransform(0.0, 0.0, 0.0),
        next_waypoints=[straight_wp_1, right_turn_wp],
    )
    projected_turn_lane_wp = _FakeWaypoint(_FakeTransform(18.0, -6.0, 0.0))

    backend.state.world = _FakeWorld(
        _FakeMap(
            road_waypoint=None,
            spawn_points=[],
            waypoints_by_xy={
                (0.0, 0.0): start_wp,
                (18.0, 0.0): projected_turn_lane_wp,
            },
        )
    )

    route_xy = backend._resolve_npc_route_xy(
        spawn_xy=(0.0, 0.0),
        route_points=[Point2D(x=18.0, y=0.0, z=0.0)],
    )

    assert route_xy[0] == pytest.approx((6.0, 0.0))
