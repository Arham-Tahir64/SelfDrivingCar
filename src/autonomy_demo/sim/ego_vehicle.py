from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EgoVehicle:
    """PRD Phase 1 placeholder ego vehicle façade."""

    actor_id: str = "ego.stub"
    attached_sensors: list[str] = field(default_factory=list)

    def apply_control(self, throttle: float, steer: float, brake: float) -> dict[str, float]:
        return {"throttle": throttle, "steer": steer, "brake": brake}

