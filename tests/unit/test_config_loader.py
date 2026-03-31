from pathlib import Path

from autonomy_demo.orchestration.config_loader import load_runtime_config, load_sensor_config


def test_load_runtime_config() -> None:
    config = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    assert config.backend == "stub"
    assert config.max_ticks == 2
    assert config.carla_sync_fps == 20
    assert config.carla_python_api_wheel.name == "carla-0.9.16-cp312-cp312-win_amd64.whl"
    assert config.perception_mode == "stub"
    assert config.perception_model_variant == "bootstrap"
    assert config.latency_budget_ms["perception"] == 60.0


def test_load_sensor_config() -> None:
    config = load_sensor_config(Path("fixtures/configs/sensors.test.yaml"))
    assert "front_camera" in config["sensors"]
