"""Integration coverage for monitoring against real PostgreSQL and MLflow.

Requires the Compose stack:

    MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
      uv run pytest -m integration

The test registers a unique model, promotes a baseline version, replays the
three dataset regimes as real prediction events plus delayed labels, and checks
the three independent verdicts. Only the events it created are deleted.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from mlflow.tracking import MlflowClient
from sqlalchemy import delete
from typer.testing import CliRunner

from modelops_lab.cli import app as cli_app
from modelops_lab.config import PromotionConfig, TrainingConfig, load_promotion_config
from modelops_lab.contracts import DriftReport, DriftVerdict, PredictionEvent
from modelops_lab.data import (
    LoadedDataset,
    load_prepared_dataset,
    prepare_dataset,
    split_frame,
)
from modelops_lab.db import PredictionEventRow, PredictionLabelRow, upgrade_database
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
from modelops_lab.serving import SqlPredictionEventStore, SqlPredictionLabelStore
from modelops_lab.tracking import log_training_run
from modelops_lab.train import predict_probabilities, train_on_dataset

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
MONITORING_CONFIG = ROOT / "configs" / "monitoring.yaml"
WINDOW = 400


@dataclass
class Environment:
    store: PointerStore
    registry: MlflowModelRegistry
    promotion_config: PromotionConfig
    config_path: Path
    baseline_dataset: LoadedDataset
    baseline_dir: Path
    datasets: dict[str, LoadedDataset]
    model: object
    model_version: str
    root: Path
    event_ids: list[int] = field(default_factory=list)


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ["MODELOPS_TEST_DATABASE_URL"]


@pytest.fixture(scope="module")
def tracking_uri() -> str:
    return os.environ["MODELOPS_TEST_TRACKING_URI"]


def _train(tracking_uri: str, config: TrainingConfig, dataset: LoadedDataset) -> CandidateRun:
    result = train_on_dataset(config, dataset)
    outcome = log_training_run(config, result, tracking_uri=tracking_uri, run_name="monitor-it")
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
            reason="integration monitoring setup",
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
    root = tmp_path_factory.mktemp("monitoring-it")
    name = f"modelops-lab-monitor-it-{uuid.uuid4().hex[:8]}"
    client = MlflowClient(tracking_uri=tracking_uri)
    promotion = load_promotion_config(PROMOTION_CONFIG)
    promotion = promotion.model_copy(
        update={"registry": promotion.registry.model_copy(update={"registered_model_name": name})}
    )
    config_path = root / "promotion.yaml"
    config_path.write_text(
        yaml.safe_dump(promotion.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    registry = MlflowModelRegistry(
        tracking_uri=tracking_uri,
        registered_model_name=name,
        alias=promotion.registry.alias,
    )

    datasets: dict[str, LoadedDataset] = {"baseline": loaded_dataset}
    for regime in ("covariate_shift", "concept_shift"):
        config = training_config.model_copy(
            update={"dataset": training_config.dataset.model_copy(update={"regime": regime})}
        )
        prepared = prepare_dataset(config.dataset, config.split, out_root=root / regime)
        datasets[regime] = load_prepared_dataset(prepared.directory)

    training = train_on_dataset(training_config, loaded_dataset)
    outcome = log_training_run(
        training_config, training, tracking_uri=tracking_uri, run_name="monitor-it-baseline"
    )
    candidate = MlflowCandidateInspector(tracking_uri).inspect(outcome.run_id)
    promoted = _promote(
        store=store,
        registry=registry,
        config=promotion,
        dataset=loaded_dataset,
        candidate=candidate,
        decision_dir=root / "decisions",
    )
    assert promoted.active_after == "1"

    env = Environment(
        store=store,
        registry=registry,
        promotion_config=promotion,
        config_path=config_path,
        baseline_dataset=loaded_dataset,
        baseline_dir=Path(loaded_dataset.directory),
        datasets=datasets,
        model=training.pipeline,
        model_version="1",
        root=root,
    )
    try:
        yield env
    finally:
        _clear_events(env)
        with contextlib.suppress(Exception):
            client.delete_registered_model(name)


def _clear_events(environment: Environment) -> None:
    if not environment.event_ids:
        return
    with environment.store.session() as session, session.begin():
        session.execute(
            delete(PredictionLabelRow).where(PredictionLabelRow.event_id.in_(environment.event_ids))
        )
        session.execute(
            delete(PredictionEventRow).where(PredictionEventRow.event_id.in_(environment.event_ids))
        )
    environment.event_ids.clear()


def _record_events(environment: Environment, dataset: LoadedDataset) -> None:
    frame = split_frame(dataset, "validation").sort_values("row_id").reset_index(drop=True)
    features = list(dataset.schema.feature_names)
    probabilities = predict_probabilities(environment.model, frame, features)
    event_store = SqlPredictionEventStore(environment.store.database_url)
    label_store = SqlPredictionLabelStore(environment.store.database_url)
    started = datetime.now(UTC)
    for index, row in frame.iterrows():
        probability = float(probabilities[index])
        event_id = event_store.record(
            PredictionEvent(
                request_id=f"monitor-it-{environment.model_version}-{index}",
                timestamp=started + timedelta(milliseconds=index),
                features={
                    name: (
                        float(row[name])
                        if name in dataset.schema.numeric_features
                        else str(row[name])
                    )
                    for name in features
                },
                prediction=int(probability >= 0.5),
                probability=probability,
                model_version=environment.model_version,
                latency_ms=1.0,
            )
        )
        environment.event_ids.append(event_id)
        label_store.upsert(event_id, int(row["target"]))


def _monitor(
    environment: Environment,
    database_url: str,
    tracking_uri: str,
    name: str,
    *,
    fail_on_drift: bool = False,
    expect_code: int = 0,
) -> tuple:
    output = environment.root / f"reports-{name}"
    runner = CliRunner()
    arguments = [
        "monitor",
        "--config",
        str(MONITORING_CONFIG),
        "--reference-dataset",
        str(environment.baseline_dir),
        "--window-size",
        str(WINDOW),
        "--output-dir",
        str(output),
        "--database-url",
        database_url,
        "--tracking-uri",
        tracking_uri,
        "--promotion-config",
        str(environment.config_path),
    ]
    if fail_on_drift:
        arguments.append("--fail-on-drift")
    result = runner.invoke(cli_app, arguments)
    assert result.exit_code == expect_code, result.stderr
    json_files = sorted(output.glob("drift-*.json"))
    html_files = sorted(output.glob("drift-*.html"))
    assert len(json_files) == 1
    assert len(html_files) == 1
    report = DriftReport.model_validate(json.loads(json_files[0].read_text(encoding="utf-8")))
    assert report.current_window.row_count == WINDOW
    assert report.performance is not None
    assert report.performance.labeled_sample_count == WINDOW
    assert report.reference_window.row_count == WINDOW
    return report, result


def test_baseline_replay_is_stable(
    environment: Environment, database_url: str, tracking_uri: str
) -> None:
    _clear_events(environment)
    _record_events(environment, environment.datasets["baseline"])

    report, _ = _monitor(environment, database_url, tracking_uri, "baseline")

    assert report.data_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.NOT_DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance is not None
    assert report.performance.delta == pytest.approx(0.0, abs=1e-6)


def test_covariate_shift_is_confounded_not_concept_drift(
    environment: Environment, database_url: str, tracking_uri: str
) -> None:
    _clear_events(environment)
    _record_events(environment, environment.datasets["covariate_shift"])

    report, resolved = _monitor(
        environment,
        database_url,
        tracking_uri,
        "covariate",
        fail_on_drift=True,
        expect_code=3,
    )

    assert report.data_drift.verdict is DriftVerdict.DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.NOT_EVALUATED
    assert "confounded" in report.concept_drift.detail
    assert report.performance is not None
    assert report.performance.delta < -0.05
    assert "data_drift: drifted" in resolved.stdout


def test_concept_shift_is_concept_drift_with_stable_inputs(
    environment: Environment, database_url: str, tracking_uri: str
) -> None:
    _clear_events(environment)
    _record_events(environment, environment.datasets["concept_shift"])

    report, _ = _monitor(environment, database_url, tracking_uri, "concept")

    assert report.data_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.DRIFTED
    assert report.performance is not None
    assert report.performance.delta < -0.05
