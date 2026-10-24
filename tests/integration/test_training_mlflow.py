"""Integration test against the real Compose MLflow tracking server.

Skipped unless ``MODELOPS_TEST_TRACKING_URI`` points at a reachable server:

    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 uv run pytest -m integration
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from modelops_lab.config import TrainingConfig
from modelops_lab.data import LoadedDataset
from modelops_lab.tracking import load_run_model, log_training_run
from modelops_lab.train import train_on_dataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("MODELOPS_TEST_TRACKING_URI"),
        reason="MODELOPS_TEST_TRACKING_URI is not set",
    ),
]


def test_train_and_read_back_against_compose_mlflow(
    tmp_path: Path, training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    tracking_uri = os.environ["MODELOPS_TEST_TRACKING_URI"]
    result = train_on_dataset(training_config, loaded_dataset)
    outcome = log_training_run(
        training_config,
        result,
        tracking_uri=tracking_uri,
        run_name=f"integration-{uuid.uuid4().hex[:8]}",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(outcome.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["dataset_fingerprint"] == loaded_dataset.manifest.fingerprint
    assert run.data.tags["split_validation_fingerprint"] == next(
        split.fingerprint for split in loaded_dataset.manifest.splits if split.name == "validation"
    )
    assert run.data.metrics["roc_auc"] == pytest.approx(result.metrics.roc_auc)

    evaluation_path = Path(client.download_artifacts(outcome.run_id, "evaluation.json"))
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert payload["features"] == list(loaded_dataset.manifest.feature_names)

    loaded = load_run_model(outcome.run_id, tracking_uri)
    assert loaded.trained_dataset_fingerprint == loaded_dataset.manifest.fingerprint
    assert loaded.model is not None
