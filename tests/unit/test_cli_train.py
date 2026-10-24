"""CLI tests for ``modelops train`` and ``modelops evaluate``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from modelops_lab.cli import app
from modelops_lab.config import TrainingConfig
from modelops_lab.data import LoadedDataset, prepare_dataset
from modelops_lab.tracking import log_training_run
from modelops_lab.train import train_on_dataset

ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


def log_run(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> tuple[str, str]:
    uri = (tmp_path / "mlruns").as_uri()
    result = train_on_dataset(training_config, loaded_dataset)
    outcome = log_training_run(training_config, result, tracking_uri=uri, run_name="cli-source")
    return uri, outcome.run_id


def test_train_prints_run_id_metrics_and_model_uri(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri = (tmp_path / "mlruns").as_uri()
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(loaded_dataset.directory),
            "--tracking-uri",
            uri,
            "--run-name",
            "cli-run",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert "run_id:" in result.stdout
    assert "dataset_fingerprint:" in result.stdout
    assert loaded_dataset.manifest.fingerprint in result.stdout
    assert "roc_auc:" in result.stdout
    assert "model_uri: models:/" in result.stdout
    assert "Traceback" not in result.stderr

    run_id = next(
        line.split(":", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("run_id:")
    )
    run = MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.params["split.seed"] == "42"
    assert run.data.tags["dataset_fingerprint"] == loaded_dataset.manifest.fingerprint


def test_evaluate_reproduces_validation_metrics_exactly(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri, run_id = log_run(tmp_path, training_config, loaded_dataset)
    client = MlflowClient(tracking_uri=uri)
    logged = client.get_run(run_id)
    runs_before = client.search_runs(experiment_ids=[logged.info.experiment_id])

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(loaded_dataset.directory),
            "--run-id",
            run_id,
            "--tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert len(client.search_runs(experiment_ids=[logged.info.experiment_id])) == len(runs_before)
    summary = json.loads(result.stdout)
    assert summary["evaluation_split"] == "validation"
    assert summary["run_id"] == run_id
    assert summary["dataset_fingerprint"] == loaded_dataset.manifest.fingerprint
    assert "WARNING" not in result.stderr

    expected = {
        "roc_auc": logged.data.metrics["roc_auc"],
        "precision": logged.data.metrics["precision"],
        "recall": logged.data.metrics["recall"],
        "f1": logged.data.metrics["f1"],
        "accuracy": logged.data.metrics["accuracy"],
        "log_loss": logged.data.metrics["log_loss"],
    }
    assert summary["metrics"] == expected
    assert summary["confusion_matrix"] == {
        "true_negative": int(logged.data.metrics["confusion_matrix_tn"]),
        "false_positive": int(logged.data.metrics["confusion_matrix_fp"]),
        "false_negative": int(logged.data.metrics["confusion_matrix_fn"]),
        "true_positive": int(logged.data.metrics["confusion_matrix_tp"]),
    }


def test_evaluate_test_split_warns_loudly(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri, run_id = log_run(tmp_path, training_config, loaded_dataset)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(loaded_dataset.directory),
            "--run-id",
            run_id,
            "--split",
            "test",
            "--tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["evaluation_split"] == "test"
    assert "held-out test split" in result.stderr
    assert "Do not use these results" in result.stderr


def test_evaluate_warns_when_the_dataset_fingerprint_differs(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri, run_id = log_run(tmp_path, training_config, loaded_dataset)
    shifted = prepare_dataset(
        training_config.dataset.model_copy(update={"regime": "covariate_shift"}),
        training_config.split,
        out_root=tmp_path / "shifted",
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(shifted.directory),
            "--run-id",
            run_id,
            "--tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["dataset_fingerprint"] == shifted.manifest.fingerprint
    assert summary["dataset_fingerprint"] != loaded_dataset.manifest.fingerprint
    assert "trained on dataset fingerprint" in result.stderr
    assert "Comparisons are only valid on a shared slice" in result.stderr


def test_evaluate_corrupt_model_artifact_fails_cleanly(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    uri, run_id = log_run(tmp_path, training_config, loaded_dataset)
    client = MlflowClient(tracking_uri=uri)
    model_id = client.get_run(run_id).outputs.model_outputs[0].model_id
    artifact_location = client.get_logged_model(model_id).artifact_location
    shutil.rmtree(Path(unquote(urlparse(artifact_location).path)))

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(loaded_dataset.directory),
            "--run-id",
            run_id,
            "--tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 1
    assert "cannot load the model" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_train_missing_dataset_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(tmp_path / "does-not-exist"),
            "--tracking-uri",
            (tmp_path / "mlruns").as_uri(),
        ],
    )

    assert result.exit_code == 1
    assert "does not exist" in result.stderr
    assert "modelops train: error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_train_corrupt_artifact_fails_cleanly(tmp_path: Path, prepared_dataset: Path) -> None:
    corrupt = tmp_path / "dataset"
    shutil.copytree(prepared_dataset, corrupt)
    dataset = corrupt / "dataset.csv"
    dataset.write_bytes(dataset.read_bytes().replace(b"row_id,age", b"row_id,age,", 1))

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(corrupt),
            "--tracking-uri",
            (tmp_path / "mlruns").as_uri(),
        ],
    )

    assert result.exit_code == 1
    assert "fingerprint mismatch" in result.stderr
    assert "Traceback" not in result.stderr


def test_train_unreachable_tracking_uri_fails_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_dataset: LoadedDataset,
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(loaded_dataset.directory),
            "--tracking-uri",
            "http://127.0.0.1:1",
        ],
    )

    assert result.exit_code == 1
    assert "cannot record the run" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_evaluate_unknown_run_fails_cleanly(tmp_path: Path, loaded_dataset: LoadedDataset) -> None:
    uri = (tmp_path / "mlruns").as_uri()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(TRAINING_CONFIG),
            "--dataset",
            str(loaded_dataset.directory),
            "--run-id",
            "00000000000000000000000000000000",
            "--tracking-uri",
            uri,
        ],
    )

    assert result.exit_code == 1
    assert "cannot read run" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_required_dataset_option_is_a_usage_error() -> None:
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 2
    assert "Traceback" not in result.stderr
