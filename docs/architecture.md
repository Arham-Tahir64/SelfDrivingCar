# Architecture Overview

The scaffold follows the PRD pipeline exactly:

`sim -> sensors -> perception -> localization -> mapping -> prediction -> planning -> control`

Rules enforced by structure:

- Shared interfaces live in `interfaces`
- Cross-cutting helpers live in `common`
- Visualization is read-only
- Replay and evaluation subscribe to runtime outputs from day one
- Each layer can be exercised in isolation with fixtures and stub data

The current implementation is a foundation only. Each module contains TODO markers that point back to the relevant PRD sections for future implementation.

Live runtime notes:

- Stub mode remains the default and does not require CARLA.
- Live mode targets CARLA 0.9.16 with the local Python API wheel path configured in `configs/app.default.yaml`.
- The simulator must be started manually before invoking `scripts/run_scenario.py --backend carla ...`.
