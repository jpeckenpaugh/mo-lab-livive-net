"""End-to-end MLflow run logging against a temporary file store (no Docker)."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from modelops_lab.config import TrainingConfig
from modelops_lab.data import LoadedDataset
from modelops_lab.tracking import (
    load_run_model,
    log_training_run,
    resolve_tracking_uri,
)
from modelops_lab.train import train_on_dataset


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # MLflow 3 refuses filesystem stores unless explicitly opted in. Unit tests
    # use a throwaway file store so they stay infrastructure-free.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


def file_store_uri(tmp_path: Path) -> str:
    return (tmp_path / "mlruns").as_uri()


def log_one_run(
    tmp_path: Path,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    *,
    name: str = "test-run",
):
    uri = file_store_uri(tmp_path)
    result = train_on_dataset(training_config, loaded_dataset)
    outcome = log_training_run(training_config, result, tracking_uri=uri, run_name=name)
    return uri, result, outcome


def test_complete_run_is_recorded_and_readable(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri, result, outcome = log_one_run(tmp_path, training_config, loaded_dataset)
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(outcome.run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["dataset.seed"] == "42"
    assert run.data.params["dataset.regime"] == "baseline"
    assert run.data.params["model.type"] == "logistic_regression"
    assert run.data.params["model_type"] == "logistic_regression"
    assert run.data.params["threshold"] == "0.5"
    assert run.data.params["model.hyperparameters.C"] == "1.0"

    manifest = loaded_dataset.manifest
    assert run.data.tags["dataset_fingerprint"] == manifest.fingerprint
    assert run.data.tags["dataset_version"] == manifest.dataset_version
    assert run.data.tags["regime"] == "baseline"
    for split in manifest.splits:
        assert run.data.tags[f"split_{split.name}_fingerprint"] == split.fingerprint
    for tag in (
        "git_revision",
        "uv_lock_sha256",
        "python_version",
        "sklearn_version",
        "mlflow_version",
        "numpy_version",
        "pandas_version",
    ):
        assert run.data.tags.get(tag)

    assert run.data.metrics["roc_auc"] == pytest.approx(result.metrics.roc_auc)
    assert run.data.metrics["confusion_matrix_tp"] == float(result.metrics.true_positive)
    assert outcome.model_id
    assert outcome.model_uri == run.data.tags["model_uri"]

    evaluation_path = Path(client.download_artifacts(outcome.run_id, "evaluation.json"))
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert payload["dataset_fingerprint"] == manifest.fingerprint
    assert payload["evaluation_split"] == "validation"
    assert payload["threshold"] == 0.5
    assert payload["features"] == list(manifest.feature_names)
    assert payload["confusion_matrix"]["true_positive"] == result.metrics.true_positive

    loaded = load_run_model(outcome.run_id, uri)
    assert loaded.model is not None
    assert loaded.model_uri == outcome.model_uri
    assert loaded.trained_dataset_fingerprint == manifest.fingerprint

    model = mlflow.models.Model.load(outcome.model_uri)
    assert model.signature is not None
    assert model.signature.inputs is not None
    assert model.signature.outputs is not None
    assert [spec.name for spec in model.signature.inputs.inputs] == list(manifest.feature_names)


def test_two_runs_are_deterministic_in_metrics_and_evaluation_artifact(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri, first_result, first = log_one_run(tmp_path, training_config, loaded_dataset, name="first")
    _, second_result, second = log_one_run(tmp_path, training_config, loaded_dataset, name="second")
    client = MlflowClient(tracking_uri=uri)

    assert first.run_id != second.run_id
    assert first_result.metrics == second_result.metrics
    assert client.get_run(first.run_id).data.metrics == client.get_run(second.run_id).data.metrics

    first_path = Path(client.download_artifacts(first.run_id, "evaluation.json"))
    second_path = Path(client.download_artifacts(second.run_id, "evaluation.json"))
    assert first_path.read_bytes() == second_path.read_bytes()


def test_tracking_uri_resolution_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri("http://config:5000") == "http://config:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    assert resolve_tracking_uri("http://config:5000") == "http://env:5000"
    assert resolve_tracking_uri("http://config:5000", "http://flag:5000") == "http://flag:5000"
