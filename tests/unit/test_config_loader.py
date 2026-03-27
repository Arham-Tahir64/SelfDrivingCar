from pathlib import Path

from autonomy_demo.orchestration.config_loader import load_runtime_config, load_sensor_config


def test_load_runtime_config() -> None:
    config = load_runtime_config(Path("fixtures/configs/app.test.yaml"))
    assert config.backend == "stub"
    assert config.max_ticks == 2
    assert config.latency_budget_ms["perception"] == 60.0


def test_load_sensor_config() -> None:
    config = load_sensor_config(Path("fixtures/configs/sensors.test.yaml"))
    assert "front_camera" in config["sensors"]

