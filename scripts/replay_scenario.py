from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.replay.reader import JsonReplayReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a recorded autonomy demo run.")
    parser.add_argument("--replay", required=True, help="Path to replay artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = JsonReplayReader(Path(args.replay)).read()
    print(f"Loaded {len(frames)} replay frames from {args.replay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
