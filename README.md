# autonomy_demo

`autonomy_demo` is a replay-first autonomy research scaffold for CARLA 0.9.16. It mirrors the PRD pipeline:

`sim -> sensors -> perception -> localization -> mapping -> prediction -> planning -> control`

The repository intentionally ships only interfaces, deterministic stubs, scenario/config loading, replay hooks, and starter tests. It is designed so each layer can be developed and tested without CARLA running.

## What Is Included

- Typed dataclasses, enums, and layer protocols
- JSON scenario schema and seven starter scenarios
- YAML config loading for app and sensor settings
- In-process event bus with read-only visualization subscribers
- Stub simulation backend and one-tick/multi-tick pipeline execution
- Replay writer/reader abstraction with HDF5-oriented interface and JSON fallback
- Evaluation harness stub and starter tests

## Quick Start

```bash
python -m pip install -e ".[dev]"
python scripts/validate_scenario.py --config scenarios/SC-01_highway_cruise.json
python scripts/run_scenario.py --config scenarios/SC-01_highway_cruise.json --record --visualize
python scripts/replay_scenario.py --replay outputs/latest/replay.json
```

## Architecture

- `src/autonomy_demo/interfaces`: shared public types and contracts
- `src/autonomy_demo/orchestration`: config/scenario loading, event bus, runtime wiring
- `src/autonomy_demo/sim`: stub and CARLA backend boundaries
- `src/autonomy_demo/sensors`: synthetic sensor bundle generation and synchronization hooks
- `src/autonomy_demo/perception` through `src/autonomy_demo/control`: stubbed autonomy layers
- `src/autonomy_demo/visualization`: read-only visualization service and optional web stub
- `src/autonomy_demo/replay`: replay writer/reader
- `src/autonomy_demo/eval`: online metrics and run summaries

## Design Constraints From The PRD

- Simulation/research demo only, not a production AV system
- CARLA 0.9.16 target
- Python 3.12 target for live CARLA integration
- Python monorepo, no microservice split
- Typed layer boundaries with no non-adjacent layer imports
- Visualization remains decoupled and read-only
- Scenario definitions are JSON-driven
- Replay and evaluation are first-class from day one

## Next Implementation Steps

1. Replace the stub simulation backend with a real CARLA adapter.
2. Flesh out live sensor synchronization and multi-rate handling for slower side/rear cameras.
3. Swap stub perception modules for YOLOv8, PointPillars, lane detection, and segmentation adapters.
4. Add OpenDRIVE parsing, Frenet planning, Stanley/PID control, and a real dashboard renderer.
