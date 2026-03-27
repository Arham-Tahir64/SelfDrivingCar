from pathlib import Path

import pytest

from autonomy_demo.common.exceptions import ScenarioValidationError
from autonomy_demo.orchestration.scenario_loader import load_scenario_config


SCHEMA = Path("scenarios/schema/scenario.schema.json")


def test_load_valid_scenario() -> None:
    scenario = load_scenario_config(Path("fixtures/scenarios/valid_scenario.json"), SCHEMA)
    assert scenario.scenario_id == "SC-TEST"
    assert scenario.eval.max_collisions == 0


def test_invalid_scenario_raises() -> None:
    with pytest.raises(ScenarioValidationError):
        load_scenario_config(Path("fixtures/scenarios/invalid_scenario.json"), SCHEMA)

