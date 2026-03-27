from __future__ import annotations

from dataclasses import dataclass

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.common.logging import get_logger
from autonomy_demo.sim.carla_runtime import CarlaSessionState, ensure_carla_importable, weather_from_name


@dataclass(slots=True)
class SpawnedActorSummary:
    ego_actor_id: int | None = None
    npc_actor_ids: list[int] | None = None
    prop_actor_ids: list[int] | None = None


class StubSimulationBackend:
    """Deterministic no-CARLA backend for tests and local development."""

    def __init__(self, runtime_config) -> None:
        self.runtime_config = runtime_config
        self.logger = get_logger(__name__, backend="stub")
        self.current_snapshot = None
        self.current_frame: int | None = None

    def bootstrap(self, scenario) -> None:
        self.logger.info("Bootstrapping stub scenario %s", scenario.scenario_id)

    def attach_sensors(self) -> None:
        self.logger.info("Attaching synthetic sensors")

    def tick(self, tick_id: int) -> None:
        self.logger.debug("Tick %s", tick_id)
        self.current_frame = tick_id

    def apply_control(self, command) -> None:
        self.logger.debug(
            "Applying command throttle=%s steer=%s brake=%s",
            command.throttle,
            command.steer,
            command.brake,
        )

    def shutdown(self) -> None:
        self.logger.info("Shutting down stub backend")


class CarlaSimulationBackend(StubSimulationBackend):
    """Live CARLA 0.9.16 session bootstrap for PRD Section 3.2.1."""

    def __init__(self, runtime_config) -> None:
        super().__init__(runtime_config)
        self.logger = get_logger(__name__, backend="carla")
        self.state = CarlaSessionState()

    def bootstrap(self, scenario) -> None:
        self.state.carla = ensure_carla_importable(self.runtime_config.carla_python_api_wheel)
        self.state.client = self.state.carla.Client(
            self.runtime_config.carla_host, self.runtime_config.carla_port
        )
        self.state.client.set_timeout(self.runtime_config.carla_timeout_s)
        self.logger.info(
            "Connecting to CARLA %s:%s using wheel %s",
            self.runtime_config.carla_host,
            self.runtime_config.carla_port,
            self.runtime_config.carla_python_api_wheel,
        )
        try:
            self.state.world = self.state.client.load_world(scenario.map_name)
        except RuntimeError as exc:
            raise CarlaRuntimeError(
                f"Failed to connect/load CARLA world '{scenario.map_name}'. "
                f"Start the simulator manually at {self.runtime_config.carla_launch_executable} "
                f"and ensure the server is listening on {self.runtime_config.carla_host}:{self.runtime_config.carla_port}."
            ) from exc
        self.state.blueprint_library = self.state.world.get_blueprint_library()
        self.state.original_settings = self.state.world.get_settings()
        settings = self.state.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.runtime_config.carla_sync_fps)
        settings.no_rendering_mode = False
        self.state.world.apply_settings(settings)
        self.state.world.set_weather(
            weather_from_name(self.state.carla, self.runtime_config.weather_preset)
        )
        self._spawn_ego_actor(scenario)
        self._spawn_scenario_actors(scenario)
        self.logger.info(
            "Spawned ego=%s npcs=%s props=%s on map %s",
            getattr(self.state.ego_actor, "id", None),
            len(self.state.npc_actors),
            len(self.state.prop_actors),
            scenario.map_name,
        )

    def _spawn_ego_actor(self, scenario) -> None:
        blueprint = self.state.blueprint_library.find(self.runtime_config.ego_vehicle_blueprint)
        transform = self.state.carla.Transform(
            self.state.carla.Location(
                x=float(scenario.ego_spawn.x),
                y=float(scenario.ego_spawn.y),
                z=float(scenario.ego_spawn.z),
            ),
            self.state.carla.Rotation(yaw=float(scenario.ego_spawn.yaw)),
        )
        actor = self.state.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise CarlaRuntimeError(
                f"Unable to spawn ego vehicle with blueprint {self.runtime_config.ego_vehicle_blueprint}"
            )
        self.state.ego_actor = actor

    def _spawn_scenario_actors(self, scenario) -> None:
        for npc in scenario.npcs:
            transform = self.state.carla.Transform(
                self.state.carla.Location(
                    x=float(npc.spawn.x),
                    y=float(npc.spawn.y),
                    z=float(npc.spawn.z),
                ),
                self.state.carla.Rotation(yaw=float(npc.spawn.yaw)),
            )
            try:
                blueprint = self.state.blueprint_library.find(npc.model)
            except IndexError:
                self.logger.warning("Skipping NPC with unknown blueprint %s", npc.model)
                continue
            actor = self.state.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                self.state.npc_actors.append(actor)
        for prop in scenario.props:
            blueprint_id = prop.type
            if "." not in blueprint_id:
                blueprint_id = f"static.prop.{prop.type}"
            try:
                blueprint = self.state.blueprint_library.find(blueprint_id)
            except IndexError:
                self.logger.warning("Skipping prop with unknown blueprint %s", blueprint_id)
                continue
            transform = self.state.carla.Transform(
                self.state.carla.Location(x=float(prop.x), y=float(prop.y), z=float(prop.z))
            )
            actor = self.state.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                self.state.prop_actors.append(actor)

    def attach_sensors(self) -> None:
        if self.state.ego_actor is None:
            raise CarlaRuntimeError("Cannot attach sensors before the ego actor exists.")
        self.logger.info("CARLA backend ready for sensor attachment")

    def tick(self, tick_id: int) -> None:
        if self.state.world is None:
            raise CarlaRuntimeError("CARLA world is not initialized.")
        self.current_frame = self.state.world.tick()
        self.state.current_frame = self.current_frame
        self.current_snapshot = self.state.world.get_snapshot()
        self.state.current_snapshot = self.current_snapshot
        self.logger.debug("CARLA world tick %s -> frame %s", tick_id, self.current_frame)

    def apply_control(self, command) -> None:
        if self.state.ego_actor is None:
            return
        vehicle_control = self.state.carla.VehicleControl(
            throttle=float(command.throttle),
            steer=float(command.steer),
            brake=float(command.brake),
            hand_brake=bool(command.hand_brake),
            reverse=bool(command.reverse),
        )
        self.state.ego_actor.apply_control(vehicle_control)

    def shutdown(self) -> None:
        for actor in list(self.state.sensor_actors.values()):
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.state.sensor_actors.clear()
        for actor in reversed(self.state.npc_actors + self.state.prop_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.state.npc_actors.clear()
        self.state.prop_actors.clear()
        if self.state.ego_actor is not None:
            try:
                self.state.ego_actor.destroy()
            except Exception:
                pass
            self.state.ego_actor = None
        if self.state.world is not None and self.state.original_settings is not None:
            self.state.world.apply_settings(self.state.original_settings)
        self.logger.info("CARLA backend shutdown complete")
