from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autonomy_demo.interfaces.types import EvaluationSummary


class SimpleEvaluationHarness:
    """Minimal online evaluator aligned with the scenario eval block."""

    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self.tick_count = 0
        self.latest_snapshot: dict[str, Any] = {}

    def update(self, tick_id: int, snapshot: dict[str, Any]) -> None:
        self.tick_count = tick_id + 1
        self.latest_snapshot = snapshot

    def finalize(self) -> EvaluationSummary:
        completion_rate = 1.0 if self.tick_count else 0.0
        success = completion_rate >= self.scenario.eval.min_completion_rate
        return EvaluationSummary(
            scenario_id=self.scenario.scenario_id,
            success=success,
            completion_rate=completion_rate,
            collision_count=0,
            red_light_violations=0,
            pedestrian_clearance_min_m=max(self.scenario.eval.min_pedestrian_clearance_m, 2.5),
            latency_ms={
                "perception": 5.0,
                "localization": 1.0,
                "mapping": 2.0,
                "prediction": 2.0,
                "planning": 3.0,
                "control": 1.0,
            },
            notes=["Stub evaluation only; replace with scenario-specific metrics."],
        )

    def write_summary(self, output_dir: Path) -> Path:
        summary = self.finalize()
        path = output_dir / "evaluation_summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return path
