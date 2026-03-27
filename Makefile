PYTHON ?= python
SCENARIO ?= scenarios/SC-01_highway_cruise.json
REPLAY ?= outputs/latest/replay.json

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m ruff format .

run:
	$(PYTHON) scripts/run_scenario.py --config $(SCENARIO) --record --visualize

validate:
	$(PYTHON) scripts/validate_scenario.py --config $(SCENARIO)

replay:
	$(PYTHON) scripts/replay_scenario.py --replay $(REPLAY)

