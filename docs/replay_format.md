# Replay Format

The scaffold exposes an HDF5-oriented replay interface but currently writes a JSON artifact for simplicity and zero-friction bootstrapping.

Each replay frame stores:

- `tick_id`
- `sim_time_s`
- `topic_payloads`

The production replay format can replace the JSON implementation with compressed HDF5 while preserving the same reader/writer contracts.
