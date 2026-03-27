from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.common.logging import configure_logging, get_logger
from autonomy_demo.orchestration.config_loader import load_runtime_config
from autonomy_demo.sim.carla_runtime import ensure_carla_importable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check basic CARLA RPC connectivity.")
    parser.add_argument("--app-config", default="configs/app.default.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_config = load_runtime_config(Path(args.app_config))
    if args.host:
        runtime_config.carla_host = args.host
    if args.port is not None:
        runtime_config.carla_port = args.port
    if args.timeout is not None:
        runtime_config.carla_timeout_s = args.timeout

    configure_logging(Path("configs/logging.yaml"), runtime_config.log_level)
    logger = get_logger(__name__)

    try:
        carla = ensure_carla_importable(runtime_config.carla_python_api_wheel)
        client = carla.Client(runtime_config.carla_host, runtime_config.carla_port)
        client.set_timeout(runtime_config.carla_timeout_s)
        world = client.get_world()
        carla_map = world.get_map().name
        actor_count = len(world.get_actors())
        logger.info(
            "Connected to CARLA at %s:%s",
            runtime_config.carla_host,
            runtime_config.carla_port,
        )
        logger.info("Current map: %s", carla_map)
        logger.info("Actor count: %s", actor_count)
        return 0
    except RuntimeError as exc:
        raise CarlaRuntimeError(
            f"CARLA RPC did not respond at {runtime_config.carla_host}:{runtime_config.carla_port}: {exc}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
