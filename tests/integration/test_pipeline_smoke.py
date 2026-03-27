from pathlib import Path

from autonomy_demo.orchestration.config_loader import load_runtime_config, load_sensor_config
from autonomy_demo.orchestration.scenario_loader import load_scenario_config
from autonomy_demo.orchestration.scenario_runner import ScenarioRunner


def test_pipeline_smoke(tmp_path: Path) -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    runtime.output_dir = tmp_path
    sensor_config = load_sensor_config(Path("configs/sensors.default.yaml"))
    scenario = load_scenario_config(
        Path("fixtures/scenarios/valid_scenario.json"),
        Path("scenarios/schema/scenario.schema.json"),
    )
    runner = ScenarioRunner(runtime, sensor_config, tmp_path)
    result = runner.run(scenario, visualize=False, record=True)
    assert result.replay_path is not None
    assert result.replay_path.exists()
    assert result.evaluation_path.exists()
    assert result.metadata_path.exists()


def test_pipeline_smoke_camera_v1(tmp_path: Path) -> None:
    runtime = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    runtime.output_dir = tmp_path
    runtime.perception_mode = "camera_v1"
    sensor_config = load_sensor_config(Path("configs/sensors.default.yaml"))
    scenario = load_scenario_config(
        Path("fixtures/scenarios/valid_scenario.json"),
        Path("scenarios/schema/scenario.schema.json"),
    )
    runner = ScenarioRunner(runtime, sensor_config, tmp_path)
    result = runner.run(scenario, visualize=False, record=True)
    assert result.replay_path is not None
    assert result.replay_path.exists()
    assert result.evaluation_path.exists()
