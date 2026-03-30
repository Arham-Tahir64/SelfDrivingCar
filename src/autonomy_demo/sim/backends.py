from __future__ import annotations

import time
from dataclasses import dataclass

import math

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
            current_world = self._wait_for_server_ready()
            current_map_name = current_world.get_map().name.split("/")[-1]
            if current_map_name == scenario.map_name:
                self.logger.info(
                    "Simulator already on requested map %s; reusing current world",
                    scenario.map_name,
                )
                self.state.world = current_world
            else:
                load_timeout_s = max(float(self.runtime_config.carla_timeout_s), 60.0)
                self.logger.info(
                    "Loading CARLA world %s with timeout %.1f s",
                    scenario.map_name,
                    load_timeout_s,
                )
                self.state.client.set_timeout(load_timeout_s)
                self.state.world = self.state.client.load_world(scenario.map_name)
                self.state.client.set_timeout(self.runtime_config.carla_timeout_s)
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
        self._attach_collision_sensor()
        self._spawn_scenario_actors(scenario)
        self.logger.info(
            "Spawned ego=%s npcs=%s props=%s on map %s",
            getattr(self.state.ego_actor, "id", None),
            len(self.state.npc_actors),
            len(self.state.prop_actors),
            scenario.map_name,
        )

    def _wait_for_server_ready(self):
        startup_wait_s = max(float(self.runtime_config.carla_timeout_s), 90.0)
        rpc_timeout_s = min(float(self.runtime_config.carla_timeout_s), 5.0)
        deadline = time.monotonic() + startup_wait_s
        attempt = 0
        last_error: RuntimeError | None = None
        while time.monotonic() < deadline:
            attempt += 1
            try:
                self.state.client.set_timeout(rpc_timeout_s)
                world = self.state.client.get_world()
                self.state.client.set_timeout(self.runtime_config.carla_timeout_s)
                self.logger.info(
                    "CARLA server became ready after %s attempt(s)",
                    attempt,
                )
                return world
            except RuntimeError as exc:
                last_error = exc
                remaining_s = max(0.0, deadline - time.monotonic())
                self.logger.info(
                    "Waiting for CARLA server readiness (attempt %s, %.0f s remaining): %s",
                    attempt,
                    remaining_s,
                    exc,
                )
                time.sleep(2.0)
        self.state.client.set_timeout(self.runtime_config.carla_timeout_s)
        if last_error is not None:
            raise last_error
        raise RuntimeError("CARLA server did not become ready before the startup timeout expired.")

    def _spawn_ego_actor(self, scenario) -> None:
        blueprint = self.state.blueprint_library.find(self.runtime_config.ego_vehicle_blueprint)
        transform = self._resolve_vehicle_spawn_transform(
            x=float(scenario.ego_spawn.x),
            y=float(scenario.ego_spawn.y),
            z=float(scenario.ego_spawn.z),
            yaw=float(scenario.ego_spawn.yaw),
            actor_label="ego",
            goal_x=float(scenario.ego_goal.x),
            goal_y=float(scenario.ego_goal.y),
        )
        actor = self.state.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise CarlaRuntimeError(
                f"Unable to spawn ego vehicle with blueprint {self.runtime_config.ego_vehicle_blueprint}"
            )
        self.state.ego_actor = actor
        self._update_spectator_view()

    def _attach_collision_sensor(self) -> None:
        if self.state.ego_actor is None:
            raise CarlaRuntimeError("Cannot attach collision sensor before the ego actor exists.")
        blueprint = self.state.blueprint_library.find("sensor.other.collision")
        sensor = self.state.world.spawn_actor(
            blueprint,
            self.state.carla.Transform(),
            attach_to=self.state.ego_actor,
        )

        def _on_collision(event) -> None:
            impulse = getattr(event, "normal_impulse", None)
            impulse_magnitude = 0.0
            if impulse is not None:
                impulse_magnitude = float(
                    (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5
                )
            self.state.collision_events.append(
                {
                    "frame": int(getattr(event, "frame", -1)),
                    "other_actor_id": int(getattr(getattr(event, "other_actor", None), "id", -1)),
                    "impulse": impulse_magnitude,
                }
            )

        sensor.listen(_on_collision)
        self.state.sensor_actors["collision_sensor"] = sensor

    def _spawn_scenario_actors(self, scenario) -> None:
        for npc in scenario.npcs:
            transform = self._resolve_vehicle_spawn_transform(
                x=float(npc.spawn.x),
                y=float(npc.spawn.y),
                z=float(npc.spawn.z),
                yaw=float(npc.spawn.yaw),
                actor_label=f"npc:{npc.model}",
            )
            try:
                blueprint = self.state.blueprint_library.find(npc.model)
            except IndexError:
                self.logger.warning("Skipping NPC with unknown blueprint %s", npc.model)
                continue
            actor = self.state.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                self.state.npc_actors.append(actor)
            else:
                self.logger.warning(
                    "Failed to spawn NPC %s at resolved transform %s",
                    npc.model,
                    transform,
                )
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

    def _resolve_vehicle_spawn_transform(
        self,
        *,
        x: float,
        y: float,
        z: float,
        yaw: float,
        actor_label: str,
        goal_x: float | None = None,
        goal_y: float | None = None,
    ):
        carla = self.state.carla
        requested_location = carla.Location(x=x, y=y, z=z)
        spawn_points = self.state.world.get_map().get_spawn_points()
        road_waypoint = self.state.world.get_map().get_waypoint(
            requested_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        candidates = []
        if road_waypoint is not None:
            resolved = road_waypoint.transform
            resolved.location.z += 0.5
            if abs(yaw) > 1e-3:
                resolved.rotation.yaw = yaw
            candidates.append(resolved)
            if goal_x is None or goal_y is None or abs(yaw) > 1e-3:
                self.logger.info(
                    "Resolved %s spawn from (%s, %s, %s, yaw=%s) to road waypoint (%s, %s, %s, yaw=%s)",
                    actor_label,
                    x,
                    y,
                    z,
                    yaw,
                    round(resolved.location.x, 2),
                    round(resolved.location.y, 2),
                    round(resolved.location.z, 2),
                    round(resolved.rotation.yaw, 2),
                )
                return resolved

        if goal_x is not None and goal_y is not None and abs(yaw) <= 1e-3 and spawn_points:
            ranked_spawn_points = sorted(
                spawn_points,
                key=lambda transform: self._distance_sq_xy(transform.location.x, transform.location.y, x, y),
            )[:12]
            for spawn_point in ranked_spawn_points:
                candidate = self._copy_transform(
                    location_xyz=(spawn_point.location.x, spawn_point.location.y, spawn_point.location.z + 0.5),
                    yaw_deg=spawn_point.rotation.yaw,
                )
                candidates.append(candidate)
            if candidates:
                best = min(
                    candidates,
                    key=lambda transform: self._spawn_candidate_score(
                        transform=transform,
                        requested_xy=(x, y),
                        goal_xy=(goal_x, goal_y),
                    ),
                )
                self.logger.info(
                    "Resolved %s spawn using goal-aware candidate (%s, %s, %s, yaw=%s) for goal (%s, %s)",
                    actor_label,
                    round(best.location.x, 2),
                    round(best.location.y, 2),
                    round(best.location.z, 2),
                    round(best.rotation.yaw, 2),
                    round(goal_x, 2),
                    round(goal_y, 2),
                )
                return best

        if road_waypoint is not None:
            self.logger.info(
                "Resolved %s spawn from (%s, %s, %s, yaw=%s) to road waypoint (%s, %s, %s, yaw=%s)",
                actor_label,
                x,
                y,
                z,
                yaw,
                round(resolved.location.x, 2),
                round(resolved.location.y, 2),
                round(resolved.location.z, 2),
                round(resolved.rotation.yaw, 2),
            )
            return resolved

        if spawn_points:
            fallback = min(
                spawn_points,
                key=lambda transform: (
                    (transform.location.x - x) ** 2 + (transform.location.y - y) ** 2
                ),
            )
            fallback.location.z += 0.5
            if abs(yaw) > 1e-3:
                fallback.rotation.yaw = yaw
            self.logger.warning(
                "Could not project %s spawn to a driving lane; using nearest spawn point (%s, %s, %s, yaw=%s)",
                actor_label,
                round(fallback.location.x, 2),
                round(fallback.location.y, 2),
                round(fallback.location.z, 2),
                round(fallback.rotation.yaw, 2),
            )
            return fallback

        self.logger.warning(
            "No valid road waypoint or map spawn point found for %s; falling back to raw transform",
            actor_label,
        )
        return carla.Transform(
            carla.Location(x=x, y=y, z=z + 0.5),
            carla.Rotation(yaw=yaw),
        )

    def _copy_transform(self, *, location_xyz: tuple[float, float, float], yaw_deg: float):
        return self.state.carla.Transform(
            self.state.carla.Location(
                x=float(location_xyz[0]),
                y=float(location_xyz[1]),
                z=float(location_xyz[2]),
            ),
            self.state.carla.Rotation(yaw=float(yaw_deg)),
        )

    @staticmethod
    def _distance_sq_xy(x1: float, y1: float, x2: float, y2: float) -> float:
        return ((float(x1) - float(x2)) ** 2) + ((float(y1) - float(y2)) ** 2)

    @classmethod
    def _spawn_candidate_score(
        cls,
        *,
        transform,
        requested_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> float:
        distance_sq = cls._distance_sq_xy(
            transform.location.x,
            transform.location.y,
            requested_xy[0],
            requested_xy[1],
        )
        goal_dx = float(goal_xy[0]) - float(transform.location.x)
        goal_dy = float(goal_xy[1]) - float(transform.location.y)
        goal_norm = math.hypot(goal_dx, goal_dy)
        if goal_norm <= 1e-6:
            return distance_sq
        heading_rad = math.radians(float(transform.rotation.yaw))
        forward_x = math.cos(heading_rad)
        forward_y = math.sin(heading_rad)
        alignment = ((forward_x * goal_dx) + (forward_y * goal_dy)) / goal_norm
        alignment_penalty = (1.0 - alignment) * 40.0
        if alignment < 0.0:
            alignment_penalty += 60.0
        return distance_sq + alignment_penalty

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
        self._update_spectator_view()
        self.logger.debug("CARLA world tick %s -> frame %s", tick_id, self.current_frame)

    def _update_spectator_view(self) -> None:
        if self.state.world is None or self.state.ego_actor is None:
            return
        ego_transform = self.state.ego_actor.get_transform()
        rotation = ego_transform.rotation
        forward = ego_transform.get_forward_vector()
        spectator_location = self.state.carla.Location(
            x=ego_transform.location.x - forward.x * 12.0,
            y=ego_transform.location.y - forward.y * 12.0,
            z=ego_transform.location.z + 6.0,
        )
        spectator_rotation = self.state.carla.Rotation(
            pitch=-20.0,
            yaw=rotation.yaw,
            roll=0.0,
        )
        spectator = self.state.world.get_spectator()
        spectator.set_transform(
            self.state.carla.Transform(spectator_location, spectator_rotation)
        )

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
