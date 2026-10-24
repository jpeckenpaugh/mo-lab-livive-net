"""Configuration loading, rejection, and environment-override tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from modelops_lab.config import (
    ConfigError,
    RuntimeSettings,
    load_monitoring_config,
    load_promotion_config,
    load_training_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


@pytest.fixture(autouse=True)
def _clear_modelops_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("MODELOPS_"):
            monkeypatch.delenv(key, raising=False)


def read_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_training_config_loads_with_coherent_defaults() -> None:
    config = load_training_config(CONFIG_DIR / "training.yaml")
    assert config.dataset.kind == "synthetic"
    assert config.dataset.n_samples == 2000
    assert config.dataset.regime == "baseline"
    assert (
        config.split.train_fraction + config.split.validation_fraction + config.split.test_fraction
        == pytest.approx(1.0)
    )
    assert config.model.type == "logistic_regression"
    assert config.model.hyperparameters.solver == "lbfgs"
    assert config.tracking.experiment_name == "modelops-lab"


def test_unknown_dataset_regime_is_rejected(tmp_path: Path) -> None:
    data = read_yaml("training.yaml")
    data["dataset"]["regime"] = "sideways"
    path = write_yaml(tmp_path, "training.yaml", data)

    with pytest.raises(ConfigError) as error:
        load_training_config(path)

    assert "dataset.regime" in str(error.value)


def test_promotion_and_monitoring_configs_load() -> None:
    promotion = load_promotion_config(CONFIG_DIR / "promotion.yaml")
    monitoring = load_monitoring_config(CONFIG_DIR / "monitoring.yaml")

    assert promotion.auto_promote is False
    assert "model" in promotion.required_artifacts
    assert monitoring.min_sample_size == 100
    assert monitoring.current.window_size >= monitoring.min_sample_size
    assert monitoring.reference.window_size >= monitoring.min_sample_size


def test_default_paths_resolve_from_repository_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    assert load_training_config().dataset.seed == 42
    assert load_promotion_config().smoke_test.enabled is True
    assert load_monitoring_config().report_output_dir == "reports/drift"


def test_unknown_config_key_is_rejected_with_its_path(tmp_path: Path) -> None:
    data = read_yaml("training.yaml")
    data["dataset"]["surprise"] = 1
    path = write_yaml(tmp_path, "training.yaml", data)

    with pytest.raises(ConfigError) as error:
        load_training_config(path)

    message = str(error.value)
    assert "dataset.surprise" in message
    assert "surprise" in message


def test_bad_type_is_rejected_with_field_name(tmp_path: Path) -> None:
    data = read_yaml("training.yaml")
    data["dataset"]["n_samples"] = "many"
    path = write_yaml(tmp_path, "training.yaml", data)

    with pytest.raises(ConfigError) as error:
        load_training_config(path)

    message = str(error.value)
    assert "dataset.n_samples" in message
    assert "valid integer" in message


def test_split_fractions_must_sum_to_one(tmp_path: Path) -> None:
    data = read_yaml("training.yaml")
    data["split"]["test_fraction"] = 0.3
    path = write_yaml(tmp_path, "training.yaml", data)

    with pytest.raises(ConfigError, match="must sum to 1.0"):
        load_training_config(path)


def test_auto_promote_true_is_rejected(tmp_path: Path) -> None:
    data = read_yaml("promotion.yaml")
    data["auto_promote"] = True
    path = write_yaml(tmp_path, "promotion.yaml", data)

    with pytest.raises(ConfigError) as error:
        load_promotion_config(path)

    assert "auto_promote" in str(error.value)
    assert "reserved" in str(error.value)


def test_monitoring_samples_must_fit_windows(tmp_path: Path) -> None:
    data = read_yaml("monitoring.yaml")
    data["current"]["window_size"] = 50
    path = write_yaml(tmp_path, "monitoring.yaml", data)

    with pytest.raises(ConfigError) as error:
        load_monitoring_config(path)

    assert "current.window_size" in str(error.value)


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELOPS_DATASET__SEED", "7")
    monkeypatch.setenv("MODELOPS_MODEL__HYPERPARAMETERS__C", "0.25")

    config = load_training_config(CONFIG_DIR / "training.yaml")

    assert config.dataset.seed == 7
    assert config.model.hyperparameters.C == 0.25


def test_json_environment_override_for_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELOPS_REQUIRED_ARTIFACTS", '["model", "evaluation_report"]')

    config = load_promotion_config(CONFIG_DIR / "promotion.yaml")

    assert config.required_artifacts == ["model", "evaluation_report"]


def test_unknown_environment_override_under_known_section_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODELOPS_DATASET__SURPRISE", "yes")

    with pytest.raises(ConfigError, match="dataset.surprise"):
        load_training_config(CONFIG_DIR / "training.yaml")


def test_runtime_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELOPS_DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("MODELOPS_APP_PORT", "9001")

    settings = RuntimeSettings(_env_file=None)

    assert settings.database_url.endswith("/x")
    assert settings.app_port == 9001
    assert settings.mlflow_tracking_uri == "http://localhost:5001"


def test_invalid_runtime_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELOPS_APP_PORT", "70000")

    with pytest.raises(ValueError, match="app_port"):
        RuntimeSettings(_env_file=None)
