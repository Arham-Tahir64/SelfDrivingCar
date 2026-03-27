from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.orchestration.scenario_loader import validate_scenario



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a scenario JSON file.")
    parser.add_argument("--config", required=True, help="Path to scenario JSON.")
    parser.add_argument("--schema", default="scenarios/schema/scenario.schema.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_scenario(Path(args.config), Path(args.schema))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
