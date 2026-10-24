"""Docker-free rollback and reconciliation tests (SQLite + fake registry)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import update

from modelops_lab.db import ActiveModelRow, ModelTransitionRow
from modelops_lab.pointer import PointerStore
from modelops_lab.registry import RegistryError
from modelops_lab.rollback import (
    RollbackArtifact,
    RollbackError,
    perform_reconcile,
    perform_rollback,
)
from modelops_lab.serving import positive_probability

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
FINGERPRINT = "a" * 64


class FakeModel:
    def __init__(self, probability: float = 0.6) -> None:
        self.probability = probability
        self.feature_names_in_ = np.array(["age", "sex"], dtype=object)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, frame: Any) -> np.ndarray:
        probability = positive_probability(self, [[0.4, self.probability]])
        return np.array([[1.0 - probability, probability]])


class BrokenModel(FakeModel):
    def predict_proba(self, frame: Any) -> np.ndarray:
        raise RuntimeError("artifact is corrupt")


class FakeResolver:
    def __init__(self, *, mutate: Any | None = None) -> None:
        self.loaded: list[str] = []
        self.mutate = mutate
        self.fail_load = False

    def resolve(self, model_name: str, version: str) -> RollbackArtifact:
        return RollbackArtifact(
            run_id=f"run-{version}",
            model_id=f"m-{version}",
            dataset_version=f"synthetic-v1.0.0-{version}",
            dataset_fingerprint=FINGERPRINT,
        )

    def load(self, model_name: str, version: str) -> Any:
        self.loaded.append(version)
        if self.mutate is not None:
            self.mutate()
        if self.fail_load:
            raise RegistryError(f"cannot load version {version}")
        return BrokenModel() if version == "broken" else FakeModel()


class FakeRegistry:
    def __init__(self, alias_version: str | None = "2") -> None:
        self.alias_version = alias_version
        self.alias_writes: list[str] = []
        self.deleted = 0

    def get_alias_version(self) -> str | None:
        return self.alias_version

    def set_alias(self, version: str) -> None:
        self.alias_version = version
        self.alias_writes.append(version)

    def delete_alias(self) -> None:
        self.alias_version = None
        self.deleted += 1


def seed(
    store: PointerStore,
    *,
    active_version: str = "2",
    history: tuple[tuple[str, str | None, str, str], ...] = (
        ("bootstrap", None, "1", "dec-1"),
        ("promotion", "1", "2", "dec-2"),
    ),
) -> None:
    with store.session() as session, session.begin():
        session.add(
            ActiveModelRow(
                id=1,
                model_name="test-model",
                model_version=active_version,
                mlflow_run_id=f"run-{active_version}",
                mlflow_model_id=f"m-{active_version}",
                dataset_version="synthetic-v1",
                dataset_fingerprint=FINGERPRINT,
                decision_id=f"dec-{active_version}",
                promoted_at=NOW,
                updated_at=NOW,
            )
        )
        for operation, source, target, decision_id in history:
            session.add(
                ModelTransitionRow(
                    operation=operation,
                    model_name="test-model",
                    from_model_version=source,
                    to_model_version=target,
                    run_id=f"run-{target}",
                    decision_id=decision_id,
                    actor="tester",
                    reason="seed",
                    occurred_at=NOW,
                )
            )


@pytest.fixture()
def store(tmp_path: Path) -> PointerStore:
    store = PointerStore(f"sqlite:///{tmp_path / 'rollback.db'}")
    store.create_schema()
    seed(store)
    return store


def test_successful_rollback_moves_alias_appends_transition_and_writes_audit(
    store: PointerStore, tmp_path: Path
) -> None:
    registry = FakeRegistry()
    resolver = FakeResolver()

    result = perform_rollback(
        target_version="1",
        reason="bad predictions after promote",
        actor="on-call",
        store=store,
        resolver=resolver,
        registry=registry,
        rollback_dir=tmp_path / "rollbacks",
        now=NOW,
    )

    assert result.no_op is False
    assert result.from_version == "2"
    assert result.to_version == "1"
    assert resolver.loaded == ["1"]
    assert registry.alias_version == "1"
    pointer = store.active()
    assert pointer is not None
    assert pointer.model_version == "1"
    assert pointer.mlflow_run_id == "run-1"
    assert pointer.decision_id == "dec-1"
    assert pointer.dataset_fingerprint == FINGERPRINT
    transitions = store.transitions()
    assert (transitions[-1].operation, transitions[-1].from_model_version) == ("rollback", "2")
    assert transitions[-1].to_model_version == "1"
    assert transitions[-1].actor == "on-call"
    assert transitions[-1].reason == "bad predictions after promote"

    assert result.audit_path is not None
    payload = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert payload["rollback_id"] == result.rollback_id
    assert payload["from_model_version"] == "2"
    assert payload["to_model_version"] == "1"
    assert payload["actor"] == "on-call"
    assert payload["reason"] == "bad predictions after promote"
    assert payload["authorizing_decision_id"] == "dec-1"
    assert payload["verification"]["passed"] is True
    assert "probability" in payload["verification"]["detail"]


def test_never_approved_version_is_refused(store: PointerStore, tmp_path: Path) -> None:
    registry = FakeRegistry()
    resolver = FakeResolver()
    before = len(store.transitions())

    with pytest.raises(RollbackError, match="never active"):
        perform_rollback(
            target_version="9",
            reason="try an unapproved version",
            actor="tester",
            store=store,
            resolver=resolver,
            registry=registry,
            rollback_dir=tmp_path / "rollbacks",
        )

    assert resolver.loaded == []
    assert registry.alias_writes == []
    assert len(store.transitions()) == before
    assert store.active() is not None and store.active().model_version == "2"
    assert not (tmp_path / "rollbacks").exists()


def test_stale_pointer_conflict_moves_nothing(store: PointerStore, tmp_path: Path) -> None:
    registry = FakeRegistry()

    def move_pointer() -> None:
        with store.session() as session, session.begin():
            session.execute(
                update(ActiveModelRow).where(ActiveModelRow.id == 1).values(model_version="3")
            )

    from modelops_lab.pointer import PromotionConflict

    with pytest.raises(PromotionConflict):
        perform_rollback(
            target_version="1",
            reason="race with another operator",
            actor="tester",
            store=store,
            resolver=FakeResolver(mutate=move_pointer),
            registry=registry,
            rollback_dir=tmp_path / "rollbacks",
        )

    assert registry.alias_writes == []
    assert store.active() is not None and store.active().model_version == "3"
    assert [transition.to_model_version for transition in store.transitions()] == ["1", "2"]
    assert not (tmp_path / "rollbacks").exists()


def test_same_version_is_a_noop_with_no_writes(store: PointerStore, tmp_path: Path) -> None:
    registry = FakeRegistry()
    resolver = FakeResolver()
    before = len(store.transitions())

    result = perform_rollback(
        target_version="2",
        reason="already there",
        actor="tester",
        store=store,
        resolver=resolver,
        registry=registry,
        rollback_dir=tmp_path / "rollbacks",
    )

    assert result.no_op is True
    assert resolver.loaded == []
    assert registry.alias_writes == []
    assert len(store.transitions()) == before
    assert not (tmp_path / "rollbacks").exists()


def test_failed_smoke_prediction_aborts_before_any_move(
    store: PointerStore, tmp_path: Path
) -> None:
    registry = FakeRegistry()
    resolver = FakeResolver()
    resolver.load = lambda name, version: BrokenModel()  # type: ignore[method-assign]

    with pytest.raises(RollbackError, match="smoke"):
        perform_rollback(
            target_version="1",
            reason="target is corrupt",
            actor="tester",
            store=store,
            resolver=resolver,
            registry=registry,
            rollback_dir=tmp_path / "rollbacks",
        )

    assert registry.alias_writes == []
    assert store.active() is not None and store.active().model_version == "2"
    assert [transition.to_model_version for transition in store.transitions()] == ["1", "2"]
    assert not (tmp_path / "rollbacks").exists()


def test_reconcile_consistent_mismatch_and_fix(store: PointerStore) -> None:
    registry = FakeRegistry(alias_version="2")
    consistent = perform_reconcile(store=store, registry=registry)
    assert consistent.consistent is True
    assert consistent.fixed is False

    registry.alias_version = "1"
    mismatch = perform_reconcile(store=store, registry=registry)
    assert mismatch.consistent is False
    assert mismatch.pointer_version == "2"
    assert mismatch.alias_version == "1"
    assert "pointer" in mismatch.detail
    assert registry.alias_writes == []

    fixed = perform_reconcile(store=store, registry=registry, fix=True)
    assert fixed.consistent is True
    assert fixed.fixed is True
    assert registry.alias_version == "2"
    assert registry.alias_writes == ["2"]


def test_reconcile_without_a_pointer_removes_a_stray_alias(tmp_path: Path) -> None:
    store = PointerStore(f"sqlite:///{tmp_path / 'empty.db'}")
    store.create_schema()
    registry = FakeRegistry(alias_version="4")

    mismatch = perform_reconcile(store=store, registry=registry)
    assert mismatch.consistent is False
    assert mismatch.pointer_version is None

    fixed = perform_reconcile(store=store, registry=registry, fix=True)
    assert fixed.fixed is True
    assert registry.deleted == 1
    assert registry.alias_version is None
