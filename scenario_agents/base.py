from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ScenarioAgentContext:
    scenario_id: str
    tick_id: int


class ScenarioAgentBehavior(Protocol):
    """PRD Section 4 placeholder contract for custom scenario-side behaviors."""

    def step(self, context: ScenarioAgentContext) -> None:
        ...

