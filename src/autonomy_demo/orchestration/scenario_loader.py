from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from autonomy_demo.common.exceptions import ScenarioValidationError
from autonomy_demo.interfaces.types import (
    Point2D,
    Pose2D,
    ScenarioConfig,
    ScenarioEvalCriteria,
    ScenarioNpcConfig,
    ScenarioPropConfig,
    ScenarioTrigger,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_scenario(config_path: Path, schema_path: Path) -> dict[str, Any]:
    scenario = _load_json(config_path)
    schema = _load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(scenario), key=lambda err: err.path)
    if errors:
        joined = "; ".join(error.message for error in errors)
        raise ScenarioValidationError(f"scenario validation failed: {joined}")
    return scenario


def load_scenario_config(config_path: Path, schema_path: Path) -> ScenarioConfig:
    data = validate_scenario(config_path, schema_path)
    return ScenarioConfig(
        scenario_id=str(data["scenario_id"]),
        name=str(data["name"]),
        map_name=str(data["map"]),
        ego_spawn=Pose2D(**data["ego_spawn"]),
        ego_goal=Point2D(**data["ego_goal"]),
        max_duration_s=float(data["max_duration_s"]),
        npcs=[
            ScenarioNpcConfig(
                model=str(item["model"]),
                behavior=str(item.get("behavior", "scripted")),
                spawn=Pose2D(**item["spawn"]),
                route=[Point2D(**point) for point in item.get("route", [])],
            )
            for item in data.get("npcs", [])
        ],
        props=[ScenarioPropConfig(**item) for item in data.get("props", [])],
        triggers=[
            ScenarioTrigger(
                type=str(item["type"]),
                at_s=item.get("at_s"),
                lane_id=item.get("lane_id"),
                metadata={k: v for k, v in item.items() if k not in {"type", "at_s", "lane_id"}},
            )
            for item in data.get("triggers", [])
        ],
        eval=ScenarioEvalCriteria(**data["eval"]),
    )

