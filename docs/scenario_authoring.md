# Scenario Authoring

Add a new scenario without changing the core stack:

1. Copy an existing file in `scenarios/`.
2. Update `scenario_id`, `name`, map, ego spawn/goal, actors, props, triggers, and eval criteria.
3. Validate it with:

```bash
python scripts/validate_scenario.py --config scenarios/your_scenario.json
```

4. If a custom NPC behavior is needed, add a class under `scenario_agents/` and reference it in the JSON `behavior` field.

This mirrors the PRD goal that new scenarios should be authorable quickly and without changing autonomy-layer code.

