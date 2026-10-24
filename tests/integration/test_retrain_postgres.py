"""Integration coverage for retraining, same-run registration, and rollback races.

Requires the Compose stack:

    MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops_test \
    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
      uv run pytest -m integration
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from modelops_lab.cli import app as cli_app
from modelops_lab.config import (
    PromotionConfig,
    TrainingConfig,
    load_promotion_config,
    load_training_config,
)
from modelops_lab.data import LoadedDataset, load_prepared_dataset, prepare_dataset
from modelops_lab.db import upgrade_database, validate_test_database_url
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
from modelops_lab.retrain import RetrainRequest, perform_retraining
from modelops_lab.rollback import (
    MlflowRollbackResolver,
    RollbackResult,
    perform_rollback,
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
INFERIOR_CONFIG = ROOT / "configs" / "training_inferior.yaml"


@pytest.fixture(scope="module")
def database_url() -> str:
    return validate_test_database_url(os.environ.get("MODELOPS_TEST_DATABASE_URL"))


@pytest.fixture(scope="module")
def tracking_uri() -> str:
    return os.environ["MODELOPS_TEST_TRACKING_URI"]


@pytest.fixture(scope="module")
def store(database_url: str) -> PointerStore:
    upgrade_database(database_url)
    return PointerStore(database_url)


@pytest.fixture(autouse=True)
def _clean_state(store: PointerStore) -> None:
    store.reset()


@pytest.fixture()
def model_name(tracking_uri: str) -> Iterator[str]:
    name = f"modelops-lab-retrain-it-{uuid.uuid4().hex[:10]}"
    yield name
    with contextlib.suppress(Exception):
        MlflowClient(tracking_uri=tracking_uri).delete_registered_model(name)


@pytest.fixture()
def promotion_config(model_name: str) -> PromotionConfig:
    config = load_promotion_config(PROMOTION_CONFIG)
    registry = config.registry.model_copy(update={"registered_model_name": model_name})
    return config.model_copy(update={"registry": registry})


@pytest.fixture()
def registry(
    tracking_uri: str, promotion_config: PromotionConfig, database_url: str
) -> MlflowModelRegistry:
    return MlflowModelRegistry(
        tracking_uri=tracking_uri,
        registered_model_name=promotion_config.registry.registered_model_name,
        alias=promotion_config.registry.alias,
        lock_database_url=database_url,
    )


@pytest.fixture(scope="module")
def improved_dataset(
    tmp_path_factory: pytest.TempPathFactory, training_config: TrainingConfig
) -> LoadedDataset:
    root = tmp_path_factory.mktemp("retrain-improved")
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


def promote(
    *,
    store: PointerStore,
    registry: MlflowModelRegistry,
    config: PromotionConfig,
    dataset: LoadedDataset,
    candidate: CandidateRun,
    decision_dir: Path,
    evaluator=None,
) -> PromotionResult:
    kwargs: dict[str, Any] = {}
    if evaluator is not None:
        kwargs["evaluator"] = evaluator
    result = perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode="approve",
            reason="integration promotion",
            actor="integration-test",
        ),
        promotion_config=config,
        dataset=dataset,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=decision_dir,
        **kwargs,
    )
    if result.decision is not None:
        tag_candidate_run(registry.tracking_uri, candidate.run_id, result.decision)
    return result


def _retrain(
    *,
    store: PointerStore,
    registry: MlflowModelRegistry,
    training_config: TrainingConfig,
    promotion_config: PromotionConfig,
    dataset: LoadedDataset,
    tracking_uri: str,
    tmp_path: Path,
    run_name: str,
    drift_report_path: Path | None = None,
    trigger: str = "manual",
):
    return perform_retraining(
        RetrainRequest(
            dataset_dir=dataset.directory,
            trigger=trigger,  # type: ignore[arg-type]
            drift_report_path=drift_report_path,
            actor="integration-test",
            reason="integration retraining",
            run_name=run_name,
        ),
        training_config=training_config,
        promotion_config=promotion_config,
        tracking_uri=tracking_uri,
        store=store,
        registry=registry,
        decision_dir=tmp_path / "decisions",
        retrain_dir=tmp_path / "retraining",
    )


def test_improved_retrain_is_eligible_then_promotion_moves_pointer(
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
        run_name="retrain-it-baseline",
    )
    assert (
        promote(
            store=store,
            registry=registry,
            config=promotion_config,
            dataset=loaded_dataset,
            candidate=baseline,
            decision_dir=tmp_path / "decisions",
        ).active_after
        == "1"
    )
    pointer_before = store.active()
    decisions_before = len(store.decisions())
    transitions_before = len(store.transitions())
    assert pointer_before is not None and pointer_before.model_version == "1"
    assert registry.get_alias_version() == "1"
    decision_files_before = set((tmp_path / "decisions").glob("*.json"))

    result = _retrain(
        store=store,
        registry=registry,
        training_config=training_config,
        promotion_config=promotion_config,
        dataset=improved_dataset,
        tracking_uri=tracking_uri,
        tmp_path=tmp_path,
        run_name="retrain-it-improved",
    )

    assert result.verdict == "eligible", [c for c in result.checks if not c.passed]
    assert result.candidate_version == "2"
    assert result.active_before == "1"
    assert result.dataset_regime == "improved_signal"
    assert result.dataset_fingerprint == improved_dataset.manifest.fingerprint
    assert result.drift_report_id is None
    assert result.audit_path.exists()
    assert set((tmp_path / "decisions").glob("*.json")) == decision_files_before
    assert len(store.decisions()) == decisions_before
    assert len(store.transitions()) == transitions_before
    assert store.active().model_version == "1"
    assert registry.get_alias_version() == "1"

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["verdict"] == "eligible"
    assert audit["dataset_version"] == improved_dataset.manifest.dataset_version
    assert audit["dataset_regime"] == "improved_signal"
    assert audit["candidate_version"] == "2"
    assert audit["active_version_before"] == "1"
    assert audit["next_command"].startswith(
        f"modelops promote --candidate-run-id {result.run_id} --approve"
    )

    approved = promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=improved_dataset,
        candidate=MlflowCandidateInspector(tracking_uri).inspect(result.run_id),
        decision_dir=tmp_path / "decisions",
    )
    assert approved.verdict.value == "approved", [c for c in approved.checks if not c.passed]
    assert approved.active_before == "1"
    assert approved.active_after == "2"
    pointer_after = store.active()
    assert pointer_after is not None and pointer_after.model_version == "2"
    assert pointer_after.dataset_fingerprint == improved_dataset.manifest.fingerprint
    assert registry.get_alias_version() == "2"
    assert [
        (transition.operation, transition.from_model_version, transition.to_model_version)
        for transition in store.transitions()
    ] == [
        ("bootstrap", None, "1"),
        ("promotion", "1", "2"),
    ]


def test_inferior_retrain_is_rejected_and_changes_nothing(
    store: PointerStore,
    registry: MlflowModelRegistry,
    promotion_config: PromotionConfig,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    tracking_uri: str,
    tmp_path: Path,
) -> None:
    baseline = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config,
        loaded=loaded_dataset,
        run_name="retrain-it-baseline-reject",
    )
    promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=loaded_dataset,
        candidate=baseline,
        decision_dir=tmp_path / "decisions",
    )
    decisions_before = len(store.decisions())
    transitions_before = len(store.transitions())

    result = _retrain(
        store=store,
        registry=registry,
        training_config=load_training_config(INFERIOR_CONFIG),
        promotion_config=promotion_config,
        dataset=loaded_dataset,
        tracking_uri=tracking_uri,
        tmp_path=tmp_path,
        run_name="retrain-it-inferior",
    )

    assert result.verdict == "rejected"
    failed = {check.name for check in result.checks if not check.passed}
    assert "quality_floor" in failed
    assert store.active().model_version == "1"
    assert registry.get_alias_version() == "1"
    assert len(store.transitions()) == transitions_before
    assert len(store.decisions()) == decisions_before
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["verdict"] == "rejected"
    assert audit["active_version_before"] == "1"
    assert "quality_floor" in {check["name"] for check in audit["checks"] if not check["passed"]}


def test_same_run_concurrent_registration_creates_one_version(
    registry: MlflowModelRegistry,
    promotion_config: PromotionConfig,
    training_config: TrainingConfig,
    loaded_dataset: LoadedDataset,
    tracking_uri: str,
    database_url: str,
    tmp_path: Path,
) -> None:
    candidate = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config,
        loaded=loaded_dataset,
        run_name="retrain-it-register-once",
    )
    assert candidate.model_id is not None
    threads_count = 6
    barrier = threading.Barrier(threads_count)
    versions: list[str] = []
    errors: list[BaseException] = []

    def register() -> None:
        try:
            local = MlflowModelRegistry(
                tracking_uri=tracking_uri,
                registered_model_name=promotion_config.registry.registered_model_name,
                alias=promotion_config.registry.alias,
                lock_database_url=database_url,
            )
            barrier.wait(timeout=60)
            versions.append(local.register_run(candidate.run_id, candidate.model_id or ""))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=register) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert len(versions) == threads_count
    assert len(set(versions)) == 1
    client = MlflowClient(tracking_uri=tracking_uri)
    matches = [
        version
        for version in client.search_model_versions(
            f"name = '{promotion_config.registry.registered_model_name}'"
        )
        if version.run_id == candidate.run_id
    ]
    assert len(matches) == 1
    assert str(matches[0].version) == versions[0]
    assert registry.find_version_for_run(candidate.run_id) == versions[0]


class RecordingRegistry:
    """Delegate registry that records every alias write."""

    def __init__(self, inner: MlflowModelRegistry) -> None:
        self.inner = inner
        self.alias_writes: list[str] = []

    def find_version_for_run(self, run_id: str) -> str | None:
        return self.inner.find_version_for_run(run_id)

    def register_run(self, run_id: str, model_id: str) -> str:
        return self.inner.register_run(run_id, model_id)

    def set_alias(self, version: str) -> None:
        self.alias_writes.append(version)
        self.inner.set_alias(version)

    def delete_alias(self) -> None:
        self.alias_writes.append("<deleted>")
        self.inner.delete_alias()

    def get_alias_version(self) -> str | None:
        return self.inner.get_alias_version()

    def load_version(self, version: str) -> Any:
        return self.inner.load_version(version)

    def version_uri(self, version: str) -> str:
        return self.inner.version_uri(version)


def test_concurrent_rollback_and_promotion_have_one_winner(
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
        run_name="retrain-it-race-baseline",
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
        run_name="retrain-it-race-improved",
    )
    promote(
        store=store,
        registry=registry,
        config=promotion_config,
        dataset=improved_dataset,
        candidate=improved,
        decision_dir=tmp_path / "decisions",
    )
    assert store.active() is not None and store.active().model_version == "2"
    transitions_before = len(store.transitions())

    third = train_and_inspect(
        tracking_uri=tracking_uri,
        config=training_config.model_copy(
            update={
                "dataset": training_config.dataset.model_copy(update={"regime": "improved_signal"}),
                "model": training_config.model.model_copy(update={"random_state": 7}),
            }
        ),
        loaded=improved_dataset,
        run_name="retrain-it-race-third",
    )
    recording = RecordingRegistry(registry)
    barrier = threading.Barrier(2)
    local = threading.local()

    def evaluator(model: Any, loaded: LoadedDataset, split: str):
        if not getattr(local, "waited", False):
            local.waited = True
            barrier.wait(timeout=60)
        from modelops_lab.train import evaluate_loaded_model

        return evaluate_loaded_model(model, loaded, split_name=split)

    def smoke(model: Any):
        barrier.wait(timeout=60)
        from modelops_lab.rollback import smoke_predict

        return smoke_predict(model)

    promotions: list[PromotionResult] = []
    rollbacks: list[RollbackResult] = []
    conflicts: list[PromotionConflict] = []
    errors: list[BaseException] = []

    def run_promotion() -> None:
        try:
            promotions.append(
                promote(
                    store=store,
                    registry=recording,  # type: ignore[arg-type]
                    config=promotion_config,
                    dataset=improved_dataset,
                    candidate=third,
                    decision_dir=tmp_path / "decisions",
                    evaluator=evaluator,
                )
            )
        except PromotionConflict as exc:
            conflicts.append(exc)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    def run_rollback() -> None:
        try:
            rollbacks.append(
                perform_rollback(
                    target_version="1",
                    reason="integration race rollback",
                    actor="integration-test",
                    store=store,
                    resolver=MlflowRollbackResolver(tracking_uri),
                    registry=recording,  # type: ignore[arg-type]
                    rollback_dir=tmp_path / "rollbacks",
                    smoke=smoke,
                )
            )
        except PromotionConflict as exc:
            conflicts.append(exc)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run_promotion), threading.Thread(target=run_rollback)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)

    assert errors == []
    assert len(conflicts) == 1
    assert len(promotions) + len(rollbacks) == 1

    pointer = store.active()
    assert pointer is not None
    if promotions:
        winner = promotions[0]
        assert winner.verdict.value == "approved"
        assert pointer.model_version == winner.candidate_version
        assert store.transitions()[-1].operation == "promotion"
    else:
        assert rollbacks[0].no_op is False
        assert pointer.model_version == "1"
        assert store.transitions()[-1].operation == "rollback"
    assert len(store.transitions()) == transitions_before + 1
    assert registry.get_alias_version() == pointer.model_version
    assert recording.alias_writes == [pointer.model_version]
    assert conflicts[0].expected == "2"
    assert conflicts[0].observed == pointer.model_version

    config_path = tmp_path / "promotion-race.yaml"
    config_path.write_text(
        yaml.safe_dump(promotion_config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    corrupted = "2" if pointer.model_version != "2" else "1"
    registry.set_alias(corrupted)
    runner = CliRunner()
    repaired = runner.invoke(
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
            str(config_path),
        ],
    )
    assert repaired.exit_code == 0, repaired.stdout + repaired.stderr
    assert "reconciled" in repaired.stdout
    assert registry.get_alias_version() == pointer.model_version
