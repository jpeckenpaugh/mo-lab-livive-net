"""Integration coverage for promotion against real PostgreSQL and MLflow.

Skipped unless both variables point at the Compose stack:

    MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
    uv run pytest -m integration

Each test uses a unique registered-model name so aliases cannot leak between
tests, and truncates the three promotion tables before it starts.
"""

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from modelops_lab.cli import app
from modelops_lab.config import (
    PromotionConfig,
    TrainingConfig,
    load_promotion_config,
    load_training_config,
)
from modelops_lab.data import LoadedDataset, load_prepared_dataset, prepare_dataset
from modelops_lab.db import upgrade_database
from modelops_lab.pointer import PointerStore, PromotionConflict
from modelops_lab.promotion import (
    CandidateRun,
    MlflowCandidateInspector,
    PromotionRequest,
    PromotionResult,
    perform_promotion,
    tag_candidate_run,
)
from modelops_lab.registry import MlflowModelRegistry
from modelops_lab.tracking import log_training_run
from modelops_lab.train import (
    evaluate_loaded_model,
    train_on_dataset,
)

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
INFERIOR_CONFIG = ROOT / "configs" / "training_inferior.yaml"


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ["MODELOPS_TEST_DATABASE_URL"]


@pytest.fixture(scope="module")
def tracking_uri() -> str:
    return os.environ["MODELOPS_TEST_TRACKING_URI"]


@pytest.fixture(scope="module")
def store(database_url: str) -> PointerStore:
    upgrade_database(database_url)
    return PointerStore(database_url)


@pytest.fixture(autouse=True)
def _clean_promotion_tables(store: PointerStore) -> None:
    store.reset()


@pytest.fixture()
def model_name(tracking_uri: str) -> Iterator[str]:
    name = f"modelops-lab-it-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(Exception):
        MlflowClient(tracking_uri=tracking_uri).delete_registered_model(name)


@pytest.fixture()
def promotion_config(model_name: str) -> PromotionConfig:
    config = load_promotion_config(PROMOTION_CONFIG)
    registry = config.registry.model_copy(update={"registered_model_name": model_name})
    return config.model_copy(update={"registry": registry})


@pytest.fixture()
def registry(tracking_uri: str, promotion_config: PromotionConfig) -> MlflowModelRegistry:
    return MlflowModelRegistry(
        tracking_uri=tracking_uri,
        registered_model_name=promotion_config.registry.registered_model_name,
        alias=promotion_config.registry.alias,
    )


@pytest.fixture(scope="module")
def improved_dataset(
    tmp_path_factory: pytest.TempPathFactory, training_config: TrainingConfig
) -> LoadedDataset:
    root = tmp_path_factory.mktemp("improved-datasets")
    improved = training_config.model_copy(
        update={"dataset": training_config.dataset.model_copy(update={"regime": "improved_signal"})}
    )
    prepared = prepare_dataset(improved.dataset, improved.split, out_root=root)
    return load_prepared_dataset(prepared.directory)


def train_and_inspect(
    *,
    tracking_uri: str,
    config: TrainingConfig,
    loaded: LoadedDataset,
    run_name: str,
) -> CandidateRun:
    result = train_on_dataset(config, loaded)
    outcome = log_training_run(config, result, tracking_uri=tracking_uri, run_name=run_name)
    return MlflowCandidateInspector(tracking_uri).inspect(outcome.run_id)


def _default_evaluator(model, loaded, split):
    return evaluate_loaded_model(model, loaded, split_name=split)


def promote(
    *,
    store: PointerStore,
    registry: MlflowModelRegistry,
    config: PromotionConfig,
    dataset: LoadedDataset,
    candidate: CandidateRun,
    decision_dir: Path,
    evaluator=_default_evaluator,
    mode: str = "approve",
    reason: str = "integration promotion",
) -> PromotionResult:
    def evaluate(model, loaded, split):
        return evaluator(model, loaded, split)

    result = perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode=mode,  # type: ignore[arg-type]
            reason=reason,
            actor="integration-test",
        ),
        promotion_config=config,
        dataset=dataset,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=decision_dir,
        evaluator=evaluate,
    )
    if result.decision is not None:
        tag_candidate_run(registry.tracking_uri, candidate.run_id, result.decision)
    return result


def test_full_lifecycle_baseline_then_improved(
    store: PointerStore,
    registry: MlflowModelRegistry,
    promotion_config: PromotionConfig,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    improved_dataset: LoadedDataset,
    tracking_uri: str,
    database_url: str,
    tmp_path: Path,
) -> None:
    baseline = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config,
        loaded=loaded_dataset,
        run_name="it-baseline",
    )
    first = promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=loaded_dataset,
        candidate=baseline,
        decision_dir=tmp_path / "decisions",
    )

    assert first.verdict.value == "approved"
    assert first.active_before is None
    assert first.active_after == "1"
    assert store.active() is not None
    assert store.active().model_version == "1"
    assert registry.get_alias_version() == "1"
    assert [(t.operation, t.to_model_version) for t in store.transitions()] == [("bootstrap", "1")]
    tags = MlflowClient(tracking_uri=tracking_uri).get_run(baseline.run_id).data.tags
    assert tags["promotion_decision_id"] == first.decision.decision_id  # type: ignore[union-attr]
    assert tags["promotion_verdict"] == "approved"
    assert tags["promotion_actor"] == "integration-test"
    assert tags["dataset_fingerprint"] == loaded_dataset.manifest.fingerprint

    improved = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config.model_copy(
            update={
                "dataset": training_config.dataset.model_copy(update={"regime": "improved_signal"})
            }
        ),
        loaded=improved_dataset,
        run_name="it-improved",
    )
    second = promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=improved_dataset,
        candidate=improved,
        decision_dir=tmp_path / "decisions",
    )

    assert second.verdict.value == "approved", [
        check for check in second.checks if not check.passed
    ]
    assert second.active_before == "1"
    assert second.active_after == "2"
    pointer = store.active()
    assert pointer is not None
    assert pointer.model_version == "2"
    assert pointer.mlflow_run_id == improved.run_id
    assert pointer.dataset_fingerprint == improved_dataset.manifest.fingerprint
    assert registry.get_alias_version() == "2"
    assert [
        (t.operation, t.from_model_version, t.to_model_version) for t in store.transitions()
    ] == [
        ("bootstrap", None, "1"),
        ("promotion", "1", "2"),
    ]
    assert len(store.decisions()) == 2
    assert len(list((tmp_path / "decisions").glob("*.json"))) == 2

    regression = next(check for check in second.checks if check.name == "regression_vs_active")
    assert "candidate" in regression.detail and "active" in regression.detail

    cli_config = tmp_path / "promotion-cli.yaml"
    cli_config.write_text(
        yaml.safe_dump(promotion_config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    decisions_before = len(store.decisions())
    transitions_before = len(store.transitions())
    noop = CliRunner().invoke(
        app,
        [
            "promote",
            "--candidate-run-id",
            improved.run_id,
            "--approve",
            "--reason",
            "re-promote the active candidate",
            "--config",
            str(cli_config),
            "--training-config",
            str(ROOT / "configs" / "training.yaml"),
            "--tracking-uri",
            tracking_uri,
            "--database-url",
            database_url,
        ],
    )

    assert noop.exit_code == 0, noop.output + noop.stderr
    assert "already active" in noop.stdout
    assert "no decision or transition written" in noop.stdout
    assert len(store.decisions()) == decisions_before
    assert len(store.transitions()) == transitions_before


def test_inferior_candidate_is_rejected_and_active_stays(
    store: PointerStore,
    registry: MlflowModelRegistry,
    promotion_config: PromotionConfig,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    improved_dataset: LoadedDataset,
    tracking_uri: str,
    tmp_path: Path,
) -> None:
    baseline = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config,
        loaded=loaded_dataset,
        run_name="it-baseline",
    )
    promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=loaded_dataset,
        candidate=baseline,
        decision_dir=tmp_path / "decisions",
    )
    improved = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config.model_copy(
            update={
                "dataset": training_config.dataset.model_copy(update={"regime": "improved_signal"})
            }
        ),
        loaded=improved_dataset,
        run_name="it-improved",
    )
    promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=improved_dataset,
        candidate=improved,
        decision_dir=tmp_path / "decisions",
    )
    assert store.active() is not None
    assert store.active().model_version == "2"
    transitions_before = len(store.transitions())

    inferior_config = load_training_config(INFERIOR_CONFIG)
    inferior = train_and_inspect(
        tracking_uri=tracking_uri,
        config=inferior_config,
        loaded=loaded_dataset,
        run_name="it-inferior",
    )
    rejected = promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=loaded_dataset,
        candidate=inferior,
        decision_dir=tmp_path / "decisions",
    )

    assert rejected.verdict.value == "rejected"
    assert rejected.decision is not None
    failed = {check.name for check in rejected.checks if not check.passed}
    assert "quality_floor" in failed
    assert rejected.decision.rejection_reasons
    assert any(
        reason.startswith("quality_floor:") for reason in rejected.decision.rejection_reasons
    )
    assert store.active() is not None and store.active().model_version == "2"
    assert registry.get_alias_version() == "2"
    assert len(store.transitions()) == transitions_before
    assert len(store.decisions()) == 3
    inferior_tags = MlflowClient(tracking_uri=tracking_uri).get_run(inferior.run_id).data.tags
    assert inferior_tags["promotion_verdict"] == "rejected"
    assert inferior_tags["promotion_decision_id"] == rejected.decision.decision_id


def test_concurrent_promotions_produce_exactly_one_winner(
    store: PointerStore,
    registry: MlflowModelRegistry,
    promotion_config: PromotionConfig,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    improved_dataset: LoadedDataset,
    tracking_uri: str,
    tmp_path: Path,
) -> None:
    baseline = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config,
        loaded=loaded_dataset,
        run_name="it-concurrency-baseline",
    )
    promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=loaded_dataset,
        candidate=baseline,
        decision_dir=tmp_path / "decisions",
    )
    assert store.active() is not None and store.active().model_version == "1"

    candidates = [
        train_and_inspect(
            tracking_uri=tracking_uri,
            config=training_config.model_copy(
                update={
                    "dataset": training_config.dataset.model_copy(
                        update={"regime": "improved_signal"}
                    ),
                    "model": training_config.model.model_copy(
                        update={"random_state": random_state}
                    ),
                }
            ),
            loaded=improved_dataset,
            run_name=f"it-concurrency-{random_state}",
        )
        for random_state in (42, 7)
    ]

    barrier = threading.Barrier(2)
    local = threading.local()

    def evaluator(model, loaded, split):
        if not getattr(local, "waited", False):
            local.waited = True
            barrier.wait(timeout=60)
        return evaluate_loaded_model(model, loaded, split_name=split)

    results: list[PromotionResult] = []
    conflicts: list[PromotionConflict] = []
    errors: list[BaseException] = []

    def run(candidate: CandidateRun) -> None:
        try:
            results.append(
                promote(
                    store=store,
                    registry=registry,
                    config=promotion_config,
                    dataset=improved_dataset,
                    candidate=candidate,
                    decision_dir=tmp_path / "decisions",
                    evaluator=evaluator,
                )
            )
        except PromotionConflict as exc:
            conflicts.append(exc)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(candidate,)) for candidate in candidates]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert len(results) == 1
    assert len(conflicts) == 1
    winner = results[0]
    assert winner.verdict.value == "approved"

    pointer = store.active()
    assert pointer is not None
    assert pointer.model_version == winner.candidate_version
    assert pointer.model_version in {"2", "3"}
    assert registry.get_alias_version() == pointer.model_version
    assert len(store.transitions()) == 2
    assert [t.operation for t in store.transitions()] == ["bootstrap", "promotion"]
    assert len(store.decisions()) == 2
    assert conflicts[0].expected == "1"
    assert conflicts[0].observed == pointer.model_version
