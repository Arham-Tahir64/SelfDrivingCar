from __future__ import annotations

from autonomy_demo.common.logging import get_logger


class StubSimulationBackend:
    """Deterministic no-CARLA backend for tests and local development."""

    def __init__(self, runtime_config) -> None:
        self.runtime_config = runtime_config
        self.logger = get_logger(__name__, backend="stub")

    def bootstrap(self, scenario) -> None:
        self.logger.info("Bootstrapping stub scenario %s", scenario.scenario_id)

    def attach_sensors(self) -> None:
        self.logger.info("Attaching synthetic sensors")

    def tick(self, tick_id: int) -> None:
        self.logger.debug("Tick %s", tick_id)

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
    """Placeholder for PRD Section 3.2.1 integration with CARLA 0.9.15."""

    def bootstrap(self, scenario) -> None:
        # TODO(PRD 3.2.1): Replace with real carla.Client bootstrap and actor spawning.
        self.logger.info("CARLA backend placeholder bootstrap for %s", scenario.scenario_id)

