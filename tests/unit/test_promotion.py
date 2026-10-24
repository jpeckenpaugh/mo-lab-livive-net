"""Unit tests for the promotion gate, pointer transaction, and decisions.

Everything here runs on SQLite with a fake registry and an injected evaluator,
so the default test tier stays Docker-free (ADR-008).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from modelops_lab.config import PromotionConfig, load_promotion_config
from modelops_lab.contracts import (
    ModelDescriptor,
    PromotionCheck,
    PromotionDecision,
    PromotionVerdict,
)
from modelops_lab.data import LoadedDataset
from modelops_lab.db import ActiveModelRow
from modelops_lab.pointer import (
    ActiveModel,
    DecisionRecord,
    PointerStore,
    PromotionConflict,
)
from modelops_lab.promotion import (
    CandidateRun,
    PromotionError,
    PromotionRequest,
    build_decision,
    build_model_descriptor,
    decision_json_writer,
    decision_tags,
    perform_promotion,
)
from modelops_lab.registry import RegistryError, reconcile
from modelops_lab.train import Metrics

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_CONFIG = ROOT / "configs" / "promotion.yaml"
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
APPROVED_CHECK = PromotionCheck(name="run_finished", passed=True, detail="FINISHED")


def make_metrics(
    roc_auc: float = 0.88,
    precision: float = 0.78,
    recall: float = 0.75,
    f1: float = 0.76,
) -> Metrics:
    return Metrics(
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=0.78,
        log_loss=0.45,
        true_negative=20,
        false_positive=5,
        false_negative=5,
        true_positive=10,
        support=15,
        predicted_positive=15,
        sample_count=40,
    )


class FakeModel:
    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics


class FakeRegistry:
    """In-memory registry implementing the ModelRegistry protocol."""

    def __init__(self, *, alias: str = "champion") -> None:
        self.registered_model_name = "modelops-lab-classifier"
        self.alias = alias
        self.alias_version: str | None = None
        self.versions: dict[str, str] = {}
        self.models: dict[str, Any] = {}
        self.registered: list[tuple[str, str]] = []
        self.alias_writes: list[str] = []
        self._next = 0

    def find_version_for_run(self, run_id: str) -> str | None:
        return self.versions.get(run_id)

    def register_run(self, run_id: str, model_id: str) -> str:
        existing = self.versions.get(run_id)
        if existing is not None:
            return existing
        self._next += 1
        version = str(self._next)
        self.versions[run_id] = version
        self.registered.append((run_id, model_id))
        self.models[version] = FakeModel(make_metrics())
        return version

    def seed_version(self, run_id: str, metrics: Metrics) -> str:
        version = self.register_run(run_id, f"m-{run_id}")
        self.models[version] = FakeModel(metrics)
        return version

    def set_alias(self, version: str) -> None:
        if version not in self.models:
            raise RegistryError(f"unknown version {version}")
        self.alias_version = version
        self.alias_writes.append(version)

    def delete_alias(self) -> None:
        self.alias_version = None
        self.alias_writes.append("<deleted>")

    def get_alias_version(self) -> str | None:
        return self.alias_version

    def load_version(self, version: str) -> Any:
        if version not in self.models:
            raise RegistryError(f"cannot load version {version}")
        return self.models[version]

    def version_uri(self, version: str) -> str:
        return f"models:/{self.registered_model_name}/{version}"


def evaluation_payload(dataset: LoadedDataset, metrics: Metrics) -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "dataset_fingerprint": dataset.manifest.fingerprint,
        "dataset_version": dataset.manifest.dataset_version,
        "metrics": {
            "roc_auc": metrics.roc_auc,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
    }


def make_candidate(
    dataset: LoadedDataset,
    metrics: Metrics,
    *,
    run_id: str = "run-candidate",
    status: str = "FINISHED",
    model_id: str | None = "m-candidate",
    evaluation: dict[str, Any] | None = None,
    evaluation_error: str | None = None,
    fingerprint: str | None = None,
    model_error: str | None = None,
) -> CandidateRun:
    model = FakeModel(metrics) if model_id is not None else None
    payload = evaluation if evaluation is not None else evaluation_payload(dataset, metrics)
    return CandidateRun(
        run_id=run_id,
        status=status,
        tags={
            "dataset_version": dataset.manifest.dataset_version,
            "dataset_fingerprint": fingerprint or dataset.manifest.fingerprint,
            "evaluation_split": "validation",
        },
        model_id=model_id,
        model_uri=f"models:/{model_id}" if model_id else None,
        model=model if model_error is None else None,
        model_error=model_error,
        evaluation=None if evaluation_error is not None else payload,
        evaluation_error=evaluation_error,
    )


def predictor_ok(model: Any, frame: Any, features: list[str]) -> np.ndarray:
    return np.full(len(frame), 0.5)


def evaluator(model: Any, dataset: LoadedDataset, split: str) -> Metrics:
    return model.metrics


@pytest.fixture(scope="module")
def promotion_config() -> PromotionConfig:
    return load_promotion_config(PROMOTION_CONFIG)


@pytest.fixture()
def store(tmp_path: Path) -> PointerStore:
    store = PointerStore(f"sqlite:///{tmp_path / 'promotion.db'}")
    store.create_schema()
    return store


def approve(
    store: PointerStore,
    registry: FakeRegistry,
    config: PromotionConfig,
    dataset: LoadedDataset,
    candidate: CandidateRun,
    *,
    decision_dir: Path,
    evaluator_fn=evaluator,
    predictor=predictor_ok,
):
    return perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode="approve",
            reason="checks passed",
            actor="tester",
        ),
        promotion_config=config,
        dataset=dataset,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=decision_dir,
        evaluator=evaluator_fn,
        predictor=predictor,
        now=NOW,
    )


def make_active(version: str, run_id: str, decision_id: str) -> ActiveModel:
    return ActiveModel(
        model_name="modelops-lab-classifier",
        model_version=version,
        mlflow_run_id=run_id,
        mlflow_model_id=f"m-{run_id}",
        dataset_version="synthetic-v1",
        dataset_fingerprint="a" * 64,
        decision_id=decision_id,
        promoted_at=NOW,
        updated_at=NOW,
    )


def make_record(decision_id: str, version: str, run_id: str, before: str | None) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        candidate_run_id=run_id,
        candidate_model_version=version,
        active_model_version_before=before,
        verdict="approved",
        checks=[APPROVED_CHECK.model_dump(mode="json")],
        rejection_reasons=[],
        actor="tester",
        reason="test",
        decided_at=NOW,
        dataset_version="synthetic-v1",
        dataset_fingerprint="a" * 64,
        evaluation_split="validation",
    )


def make_decision(decision_id: str, version: str, before: str | None) -> PromotionDecision:
    return build_decision(
        decision_id=decision_id,
        candidate_version=version,
        active_version=before,
        checks=[APPROVED_CHECK],
        verdict=PromotionVerdict.APPROVED,
        rejection_reasons=[],
        actor="tester",
        reason="test",
        decided_at=NOW,
    )


def test_all_checks_pass_and_bootstrap_writes_state(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    assert result.verdict is PromotionVerdict.APPROVED
    assert result.no_op is False
    assert result.active_before is None
    assert result.active_after == "1"
    assert [check.name for check in result.checks] == [
        "run_finished",
        "model_logged",
        "evaluation_artifact",
        "dataset_identity",
        "required_artifacts",
        "quality_floor",
        "regression_vs_active",
        "smoke_test",
        "registry_consistency",
    ]
    assert all(check.passed for check in result.checks)
    assert result.checks[6].detail == "no active model (baseline promotion)"
    assert len(result.checks[5].detail.splitlines()) == 4

    pointer = store.active()
    assert pointer is not None
    assert pointer.model_version == "1"
    assert pointer.mlflow_run_id == candidate.run_id
    assert result.decision is not None
    assert pointer.decision_id == result.decision.decision_id
    transitions = store.transitions()
    assert [
        (item.operation, item.from_model_version, item.to_model_version) for item in transitions
    ] == [("bootstrap", None, "1")]
    assert registry.alias_version == "1"

    decisions = store.decisions()
    assert len(decisions) == 1
    assert decisions[0].verdict == "approved"
    assert decisions[0].evaluation_split == "validation"

    path = result.decision_path
    assert path is not None and path.exists()
    restored = PromotionDecision.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored == result.decision
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "1.1.0"


def test_run_not_finished_fails_and_blocks_approval(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics(), status="RUNNING")

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    assert result.verdict is PromotionVerdict.REJECTED
    assert result.active_after is None
    failed = [check.name for check in result.checks if not check.passed]
    assert "run_finished" in failed
    assert result.decision is not None
    assert any(reason.startswith("run_finished:") for reason in result.decision.rejection_reasons)
    assert store.active() is None
    assert registry.alias_version is None
    assert store.transitions() == []


def test_missing_logged_model_fails_model_and_required_artifacts(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(
        loaded_dataset,
        make_metrics(),
        model_id=None,
        model_error="the run has no logged model output",
    )

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    failed = {check.name for check in result.checks if not check.passed}
    assert {"model_logged", "required_artifacts", "quality_floor", "smoke_test"} <= failed
    assert result.verdict is PromotionVerdict.REJECTED
    assert result.candidate_version is None


def test_missing_evaluation_artifact_fails(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(
        loaded_dataset,
        make_metrics(),
        evaluation_error="evaluation.json unavailable: not found",
    )

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    failed = {check.name for check in result.checks if not check.passed}
    assert {"evaluation_artifact", "dataset_identity", "required_artifacts"} <= failed


def test_dataset_fingerprint_mismatch_fails(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics(), fingerprint="0" * 64)

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    check = next(item for item in result.checks if item.name == "dataset_identity")
    assert not check.passed
    assert "does not match artifact" in check.detail


def test_evaluation_split_must_be_validation(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    payload = evaluation_payload(loaded_dataset, make_metrics())
    payload["evaluation_split"] = "test"
    candidate = make_candidate(loaded_dataset, make_metrics(), evaluation=payload)

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    check = next(item for item in result.checks if item.name == "dataset_identity")
    assert not check.passed
    assert "requires 'validation'" in check.detail


def test_quality_floor_failure_records_every_metric(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    weak = make_metrics(roc_auc=0.61, precision=0.20, recall=0.20, f1=0.20)
    candidate = make_candidate(loaded_dataset, weak)

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    check = next(item for item in result.checks if item.name == "quality_floor")
    assert not check.passed
    assert len(check.detail.splitlines()) == 4
    assert "roc_auc 0.6100 < floor 0.8300" in check.detail
    assert check.observed_value == pytest.approx(0.20)
    assert check.threshold_value == pytest.approx(0.70)
    assert result.verdict is PromotionVerdict.REJECTED
    assert store.active() is None
    assert registry.alias_version is None


def test_regression_against_active_uses_the_candidate_slice(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    active_metrics = make_metrics(roc_auc=0.92, precision=0.86, recall=0.84, f1=0.85)
    registry.seed_version("run-active", active_metrics)
    bootstrap = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        make_candidate(loaded_dataset, active_metrics, run_id="run-active"),
        decision_dir=tmp_path / "decisions",
    )
    assert bootstrap.verdict is PromotionVerdict.APPROVED
    assert registry.alias_version == "1"

    candidate_metrics = make_metrics(roc_auc=0.84, precision=0.78, recall=0.76, f1=0.77)
    candidate = make_candidate(loaded_dataset, candidate_metrics, run_id="run-worse")

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )

    check = next(item for item in result.checks if item.name == "regression_vs_active")
    assert not check.passed
    assert "candidate 0.8400 vs active 0.9200" in check.detail
    assert result.verdict is PromotionVerdict.REJECTED
    assert store.active() is not None and store.active().model_version == "1"
    assert len(store.transitions()) == 1


def test_improved_candidate_passes_regression_and_promotes(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    registry.seed_version("run-active", make_metrics())
    approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        make_candidate(loaded_dataset, make_metrics(), run_id="run-active"),
        decision_dir=tmp_path / "decisions",
    )

    improved = make_metrics(roc_auc=0.95, precision=0.90, recall=0.88, f1=0.89)
    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        make_candidate(loaded_dataset, improved, run_id="run-improved"),
        decision_dir=tmp_path / "decisions",
    )

    assert result.verdict is PromotionVerdict.APPROVED
    assert result.active_before == "1"
    assert result.active_after == "2"
    assert store.active() is not None and store.active().model_version == "2"
    assert [
        (t.operation, t.from_model_version, t.to_model_version) for t in store.transitions()
    ] == [("bootstrap", None, "1"), ("promotion", "1", "2")]
    assert registry.alias_version == "2"


def test_smoke_test_failure_blocks_approval(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
        predictor=lambda model, frame, features: np.full(len(frame), np.nan),
    )

    check = next(item for item in result.checks if item.name == "smoke_test")
    assert not check.passed
    assert "not all finite" in check.detail
    assert result.verdict is PromotionVerdict.REJECTED
    assert store.active() is None


def test_smoke_test_rejects_out_of_range_probabilities(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
        predictor=lambda model, frame, features: np.full(len(frame), 1.5),
    )

    check = next(item for item in result.checks if item.name == "smoke_test")
    assert not check.passed
    assert "outside [0, 1]" in check.detail


def test_alias_mismatch_fails_with_reconciliation_message(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    registry.seed_version("run-active", make_metrics())
    approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        make_candidate(loaded_dataset, make_metrics(), run_id="run-active"),
        decision_dir=tmp_path / "decisions",
    )
    registry.seed_version("run-other", make_metrics())
    registry.alias_version = "2"  # simulate drift between pointer and alias

    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        make_candidate(loaded_dataset, make_metrics(roc_auc=0.95), run_id="run-new"),
        decision_dir=tmp_path / "decisions",
    )

    check = next(item for item in result.checks if item.name == "registry_consistency")
    assert not check.passed
    assert "reconcile" in check.detail.lower()
    assert result.verdict is PromotionVerdict.REJECTED


def test_human_reject_with_all_checks_passing_is_recorded(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())

    result = perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode="reject",
            reason="holding for a manual fairness review",
            actor="reviewer",
        ),
        promotion_config=promotion_config,
        dataset=loaded_dataset,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=tmp_path / "decisions",
        evaluator=evaluator,
        predictor=predictor_ok,
        now=NOW,
    )

    assert all(check.passed for check in result.checks)
    assert result.verdict is PromotionVerdict.REJECTED
    assert result.decision is not None
    assert result.decision.rejection_reasons == ["holding for a manual fairness review"]
    assert result.decision.actor == "reviewer"
    assert store.active() is None
    assert registry.alias_version is None
    assert len(store.decisions()) == 1
    assert store.transitions() == []


def test_dry_run_writes_nothing_and_does_not_register(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())
    decision_dir = tmp_path / "decisions"

    result = perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode="dry-run",
            reason="preview",
            actor="tester",
        ),
        promotion_config=promotion_config,
        dataset=loaded_dataset,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=decision_dir,
        evaluator=evaluator,
        predictor=predictor_ok,
        now=NOW,
    )

    assert result.dry_run is True
    assert result.verdict is PromotionVerdict.APPROVED
    assert result.decision is None
    assert result.decision_path is None
    assert result.candidate_version is None
    assert registry.registered == []
    assert registry.alias_version is None
    assert store.active() is None
    assert store.decisions() == []
    assert store.transitions() == []
    assert not decision_dir.exists()


def test_repromoting_the_active_candidate_is_a_noop(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())
    decision_dir = tmp_path / "decisions"
    first = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=decision_dir,
    )
    assert first.verdict is PromotionVerdict.APPROVED

    second = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=decision_dir,
    )

    assert second.no_op is True
    assert second.verdict is PromotionVerdict.APPROVED
    assert second.decision is None
    assert second.active_before == second.active_after == "1"
    assert len(store.decisions()) == 1
    assert len(store.transitions()) == 1
    assert registry.alias_writes == ["1"]


def test_stale_expected_version_conflicts_and_writes_nothing(
    tmp_path: Path,
    store: PointerStore,
) -> None:
    registry = FakeRegistry()
    registry.seed_version("run-one", make_metrics())
    registry.seed_version("run-two", make_metrics())
    decision_dir = tmp_path / "decisions"

    store.apply_promotion(
        decision=make_record("dec-one", "1", "run-one", None),
        expected_active_version=None,
        operation="bootstrap",
        target=make_active("1", "run-one", "dec-one"),
        set_alias=lambda: registry.set_alias("1"),
        restore_alias=lambda: registry.delete_alias(),
        write_decision_json=decision_json_writer(make_decision("dec-one", "1", None), decision_dir),
    )
    transitions_before = len(store.transitions())
    decisions_before = len(store.decisions())
    alias_writes_before = list(registry.alias_writes)

    with pytest.raises(PromotionConflict) as error:
        store.apply_promotion(
            decision=make_record("dec-two", "2", "run-two", "1"),
            expected_active_version="stale-version",
            operation="promotion",
            target=make_active("2", "run-two", "dec-two"),
            set_alias=lambda: registry.set_alias("2"),
            restore_alias=lambda: registry.set_alias("1"),
            write_decision_json=decision_json_writer(
                make_decision("dec-two", "2", "1"), decision_dir
            ),
        )

    assert error.value.observed == "1"
    assert store.active() is not None and store.active().model_version == "1"
    assert len(store.transitions()) == transitions_before
    assert len(store.decisions()) == decisions_before
    assert registry.alias_writes == alias_writes_before
    assert registry.alias_version == "1"
    assert not (decision_dir / "dec-two.json").exists()


def test_decision_json_matches_database_record(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    decision_dir = tmp_path / "decisions"
    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        make_candidate(loaded_dataset, make_metrics()),
        decision_dir=decision_dir,
    )

    assert result.decision is not None and result.decision_path is not None
    payload = json.loads(result.decision_path.read_text(encoding="utf-8"))
    record = store.decisions()[0]
    assert payload["decision_id"] == record.decision_id
    assert payload["candidate_version"] == record.candidate_model_version
    assert payload["verdict"] == record.verdict
    assert payload["reason"] == record.reason
    assert payload["checks"] == record.checks
    text = result.decision_path.read_text(encoding="utf-8").lower()
    assert "password" not in text
    assert "modelops:modelops@" not in text


def test_model_descriptor_is_built_with_approval_metadata(
    tmp_path: Path,
    store: PointerStore,
    promotion_config: PromotionConfig,
    loaded_dataset: LoadedDataset,
) -> None:
    registry = FakeRegistry()
    candidate = make_candidate(loaded_dataset, make_metrics())
    result = approve(
        store,
        registry,
        promotion_config,
        loaded_dataset,
        candidate,
        decision_dir=tmp_path / "decisions",
    )
    assert result.decision is not None
    assert result.candidate_version is not None

    descriptor = build_model_descriptor(
        candidate=candidate,
        metrics=make_metrics(),
        model_version=result.candidate_version,
        dataset_fingerprint=loaded_dataset.manifest.fingerprint,
        decision=result.decision,
        registry_alias=promotion_config.registry.alias,
        artifact_uri="models:/m-candidate",
    )

    assert descriptor.model_version == "1"
    assert descriptor.mlflow_run_id == candidate.run_id
    assert descriptor.evaluation_split == "validation"
    assert descriptor.approval is not None
    assert descriptor.approval.decision_id == result.decision.decision_id
    assert descriptor.approval.approved_by == "tester"
    assert descriptor.approval.registry_alias == "champion"
    assert descriptor.metrics.sample_count == 40
    assert descriptor.metrics.confusion_matrix.true_positive == 10
    assert ModelDescriptor.model_validate(descriptor.model_dump(mode="json")) == descriptor


def test_active_model_table_enforces_the_singleton(store: PointerStore) -> None:
    def row(identifier: int) -> ActiveModelRow:
        return ActiveModelRow(
            id=identifier,
            model_name="modelops-lab-classifier",
            model_version=str(identifier),
            mlflow_run_id=f"run-{identifier}",
            mlflow_model_id=f"m-{identifier}",
            dataset_version="synthetic-v1",
            dataset_fingerprint="a" * 64,
            decision_id=f"dec-{identifier}",
            promoted_at=NOW,
            updated_at=NOW,
        )

    with store.session() as session, session.begin():
        session.add(row(1))

    with pytest.raises(IntegrityError), store.session() as session, session.begin():
        session.add(row(2))


def test_decision_mlflow_tags_link_the_decision() -> None:
    decision = make_decision("dec-tags", "3", "2")

    tags = decision_tags(decision)

    assert tags == {
        "promotion_decision_id": "dec-tags",
        "promotion_verdict": "approved",
        "promotion_actor": "tester",
    }


def test_build_decision_refuses_approved_with_failed_check() -> None:
    with pytest.raises(PromotionError, match="failed checks"):
        build_decision(
            decision_id="dec-x",
            candidate_version="1",
            active_version=None,
            checks=[PromotionCheck(name="quality_floor", passed=False, detail="below")],
            verdict=PromotionVerdict.APPROVED,
            rejection_reasons=[],
            actor="tester",
            reason="test",
            decided_at=NOW,
        )


def test_reconcile_reports_mismatch_and_agreement() -> None:
    registry = FakeRegistry()
    registry.seed_version("run-one", make_metrics())
    registry.alias_version = "1"

    consistent = reconcile(registry, "1")
    assert consistent.consistent is True

    mismatched = reconcile(registry, "2")
    assert mismatched.consistent is False
    assert "reconcile" in mismatched.detail.lower()


def test_conflict_detected_before_the_alias_move_never_sets_it(tmp_path: Path) -> None:
    store = PointerStore(f"sqlite:///{tmp_path / 'conflict.db'}")
    store.create_schema()
    calls: list[str] = []

    with pytest.raises(PromotionConflict):
        store.apply_promotion(
            decision=make_record("dec-never", "9", "run-never", None),
            expected_active_version="1",
            operation="promotion",
            target=make_active("9", "run-never", "dec-never"),
            set_alias=lambda: calls.append("set"),
            restore_alias=lambda: calls.append("restore"),
            write_decision_json=lambda: tmp_path / "never.json",
        )

    assert calls == []
    assert store.active() is None
    assert store.decisions() == []
    assert store.transitions() == []
