"""Integration coverage for serving, rollback, and reconciliation.

Requires the Compose PostgreSQL + MLflow stack:

    MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
      uv run pytest -m integration
"""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pytest
import yaml
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from modelops_lab.cli import app as cli_app
from modelops_lab.config import (
    PromotionConfig,
    RuntimeSettings,
    TrainingConfig,
    load_promotion_config,
)
from modelops_lab.data import (
    LoadedDataset,
    load_prepared_dataset,
    prepare_dataset,
    split_frame,
)
from modelops_lab.db import PredictionEventRow, upgrade_database
from modelops_lab.pointer import PointerStore
from modelops_lab.promotion import (
    CandidateRun,
    MlflowCandidateInspector,
    PromotionRequest,
    PromotionResult,
    perform_promotion,
    tag_candidate_run,
)
from modelops_lab.registry import MlflowModelRegistry
from modelops_lab.serving import (
    PointerModelProvider,
    ServingMetrics,
    SqlPredictionEventStore,
    create_app,
)
from modelops_lab.tracking import log_training_run
from modelops_lab.train import train_on_dataset

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("MODELOPS_TEST_DATABASE_URL")
        or not os.environ.get("MODELOPS_TEST_TRACKING_URI"),
        reason="MODELOPS_TEST_DATABASE_URL and MODELOPS_TEST_TRACKING_URI are required",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_CONFIG = ROOT / "configs" / "promotion.yaml"


@dataclass
class Environment:
    store: PointerStore
    registry: MlflowModelRegistry
    promotion_config: PromotionConfig
    config_path: Path
    improved_dataset: LoadedDataset


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ["MODELOPS_TEST_DATABASE_URL"]


@pytest.fixture(scope="module")
def tracking_uri() -> str:
    return os.environ["MODELOPS_TEST_TRACKING_URI"]


def _train(
    tracking_uri: str, config: TrainingConfig, loaded: LoadedDataset, run_name: str
) -> CandidateRun:
    result = train_on_dataset(config, loaded)
    outcome = log_training_run(config, result, tracking_uri=tracking_uri, run_name=run_name)
    return MlflowCandidateInspector(tracking_uri).inspect(outcome.run_id)


def _promote(
    *,
    store: PointerStore,
    registry: MlflowModelRegistry,
    config: PromotionConfig,
    dataset: LoadedDataset,
    candidate: CandidateRun,
    decision_dir: Path,
) -> PromotionResult:
    result = perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode="approve",
            reason="integration serving setup",
            actor="integration-test",
        ),
        promotion_config=config,
        dataset=dataset,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=decision_dir,
    )
    assert result.verdict.value == "approved", [
        check for check in result.checks if not check.passed
    ]
    if result.decision is not None:
        tag_candidate_run(registry.tracking_uri, candidate.run_id, result.decision)
    return result


@pytest.fixture(scope="module")
def environment(
    database_url: str,
    tracking_uri: str,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    tmp_path_factory: pytest.TempPathFactory,
) -> Environment:
    upgrade_database(database_url)
    store = PointerStore(database_url)
    store.reset()
    name = f"modelops-lab-serve-it-{uuid.uuid4().hex[:8]}"
    client = MlflowClient(tracking_uri=tracking_uri)
    promotion = load_promotion_config(PROMOTION_CONFIG)
    promotion = promotion.model_copy(
        update={"registry": promotion.registry.model_copy(update={"registered_model_name": name})}
    )
    registry = MlflowModelRegistry(
        tracking_uri=tracking_uri,
        registered_model_name=name,
        alias=promotion.registry.alias,
    )
    root = tmp_path_factory.mktemp("serving-it")
    improved_config = training_config.model_copy(
        update={"dataset": training_config.dataset.model_copy(update={"regime": "improved_signal"})}
    )
    prepared = prepare_dataset(improved_config.dataset, improved_config.split, out_root=root)
    improved_dataset = load_prepared_dataset(prepared.directory)

    baseline = _train(tracking_uri, training_config, loaded_dataset, "serve-it-baseline")
    _promote(
        store=store,
        registry=registry,
        config=promotion,
        dataset=loaded_dataset,
        candidate=baseline,
        decision_dir=root / "decisions",
    )
    improved = _train(tracking_uri, improved_config, improved_dataset, "serve-it-improved")
    second = _promote(
        store=store,
        registry=registry,
        config=promotion,
        dataset=improved_dataset,
        candidate=improved,
        decision_dir=root / "decisions",
    )
    assert second.active_after == "2"

    config_path = root / "promotion.yaml"
    config_path.write_text(
        yaml.safe_dump(promotion.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    try:
        yield Environment(
            store=store,
            registry=registry,
            promotion_config=promotion,
            config_path=config_path,
            improved_dataset=improved_dataset,
        )
    finally:
        with contextlib.suppress(Exception):
            client.delete_registered_model(name)


def _payload(loaded: LoadedDataset) -> dict[str, object]:
    row = split_frame(loaded, "validation").iloc[0]
    categorical = set(loaded.schema.categorical_features)
    return {
        name: str(row[name]) if name in categorical else float(row[name])
        for name in loaded.manifest.feature_names
    }


def _client(environment: Environment, database_url: str, tracking_uri: str) -> TestClient:
    metrics = ServingMetrics()
    event_store = SqlPredictionEventStore(database_url)
    pointer_store = PointerStore(database_url)

    def load(name: str, version: str) -> object:
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow.sklearn.load_model(f"models:/{name}/{version}")

    provider = PointerModelProvider(
        pointer_reader=pointer_store.active,
        loader=load,
        metrics=metrics,
        alias_reader=environment.registry.get_alias_version,
        decision_reader=pointer_store.decision,
        database_ping=event_store.ping,
        alias_check_interval_seconds=0.0,
    )
    app = create_app(
        settings=RuntimeSettings(database_url=database_url, mlflow_tracking_uri=tracking_uri),
        provider=provider,
        event_store=event_store,
        metrics=metrics,
    )
    return TestClient(app)


def test_serve_rollback_without_restart_and_prediction_events(
    environment: Environment,
    database_url: str,
    tracking_uri: str,
    tmp_path: Path,
) -> None:
    client = _client(environment, database_url, tracking_uri)
    payload = _payload(environment.improved_dataset)

    served = client.post("/predict", json=payload)
    assert served.status_code == 200, served.text
    assert served.json()["model_version"] == "2"
    assert client.get("/model").json()["model_version"] == "2"

    runner = CliRunner()
    rollback = runner.invoke(
        cli_app,
        [
            "rollback",
            "--to-version",
            "1",
            "--reason",
            "integration rollback proof",
            "--actor",
            "integration-test",
            "--database-url",
            database_url,
            "--tracking-uri",
            tracking_uri,
            "--config",
            str(environment.config_path),
            "--rollback-dir",
            str(tmp_path / "rollbacks"),
        ],
    )
    assert rollback.exit_code == 0, rollback.stderr
    assert "from_version: 2" in rollback.stdout
    assert "to_version: 1" in rollback.stdout
    assert "verification: PASS" in rollback.stdout

    after = client.post("/predict", json=payload)
    assert after.status_code == 200, after.text
    assert after.json()["model_version"] == "1"

    pointer = environment.store.active()
    assert pointer is not None and pointer.model_version == "1"
    assert environment.registry.get_alias_version() == "1"
    assert environment.store.transitions()[-1].operation == "rollback"
    assert len(list((tmp_path / "rollbacks").glob("*.json"))) == 1

    with environment.store.session() as session:
        rows = session.query(PredictionEventRow).order_by(PredictionEventRow.event_id).all()
    assert [row.model_version for row in rows] == ["2", "1"]
    assert all(row.request_id for row in rows)
    assert all(row.features["sex"] in {"female", "male"} for row in rows)
    assert all(row.probability >= 0.0 for row in rows)
    assert all(row.latency_ms >= 0.0 for row in rows)


def test_reconcile_repairs_a_deliberately_moved_alias(
    environment: Environment,
    database_url: str,
    tracking_uri: str,
) -> None:
    pointer = environment.store.active()
    assert pointer is not None
    other = "2" if pointer.model_version == "1" else "1"
    environment.registry.set_alias(other)

    runner = CliRunner()
    mismatch = runner.invoke(
        cli_app,
        [
            "registry",
            "reconcile",
            "--database-url",
            database_url,
            "--tracking-uri",
            tracking_uri,
            "--config",
            str(environment.config_path),
        ],
    )
    assert mismatch.exit_code == 1, mismatch.stdout
    assert f"pointer_version: {pointer.model_version}" in mismatch.stdout
    assert f"alias_version: {other}" in mismatch.stdout
    assert "mismatch" in mismatch.stdout

    fixed = runner.invoke(
        cli_app,
        [
            "registry",
            "reconcile",
            "--fix",
            "--database-url",
            database_url,
            "--tracking-uri",
            tracking_uri,
            "--config",
            str(environment.config_path),
        ],
    )
    assert fixed.exit_code == 0, fixed.stderr
    assert "reconciled" in fixed.stdout
    assert environment.registry.get_alias_version() == pointer.model_version
