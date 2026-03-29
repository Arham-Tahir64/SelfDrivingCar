from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy_demo.common.logging import configure_logging
from autonomy_demo.common.paths import ensure_directory
from autonomy_demo.orchestration.config_loader import load_runtime_config, load_sensor_config
from autonomy_demo.orchestration.scenario_loader import load_scenario_config
from autonomy_demo.orchestration.scenario_runner import ScenarioRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an autonomy demo scenario.")
    parser.add_argument("--config", required=True, help="Path to scenario JSON.")
    parser.add_argument("--app-config", default="configs/app.default.yaml")
    parser.add_argument("--sensor-config", default="configs/sensors.default.yaml")
    parser.add_argument("--backend", choices=["stub", "carla"], default=None)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--perception-mode", choices=["stub", "camera_v1", "lidar_v1", "fused_v1"], default=None)
    parser.add_argument("--perception-device", default=None)
    parser.add_argument("--perception-model-variant", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--lidar-view", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_config = load_runtime_config(Path(args.app_config))
    if args.backend:
        runtime_config.backend = args.backend
    if args.max_ticks is not None:
        runtime_config.max_ticks = args.max_ticks
    if args.perception_mode:
        runtime_config.perception_mode = args.perception_mode
    if args.perception_device:
        runtime_config.perception_device = args.perception_device
    if args.perception_model_variant:
        runtime_config.perception_model_variant = args.perception_model_variant
    if args.visualize or args.lidar_view:
        runtime_config.enable_visualization = True
    if args.record:
        runtime_config.record_replay = True
    configure_logging(Path("configs/logging.yaml"), runtime_config.log_level)
    sensor_config = load_sensor_config(Path(args.sensor_config))
    scenario = load_scenario_config(Path(args.config), Path("scenarios/schema/scenario.schema.json"))
    if args.validate:
        return 0
    output_dir = ensure_directory(runtime_config.output_dir / scenario.scenario_id)
    runner = ScenarioRunner(runtime_config, sensor_config, output_dir)
    runner.run(
        scenario=scenario,
        visualize=runtime_config.enable_visualization and not args.headless,
        record=runtime_config.record_replay,
        lidar_view=args.lidar_view and not args.headless,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
