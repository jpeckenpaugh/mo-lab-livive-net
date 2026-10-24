"""Transactional active-model pointer, transition log, and decision records.

Approval is the only operation that moves the pointer. It runs in a single
database transaction that:

1. takes a transaction-scoped PostgreSQL advisory lock (constant key);
2. re-reads the pointer and aborts with :class:`PromotionConflict` if another
   promotion moved it since the gate captured its expected version;
3. sets the MLflow alias, inserts the decision and transition rows, and updates
   the pointer last (ADR-006);
4. commits.

The alias move is an external registry call made inside the database transaction
immediately after the conflict check, so a commit failure after the move attempts
to restore the previous alias and reports that reconciliation may be required.
SQLite (the unit-test tier) has no advisory lock; the expected-version check
still prevents a lost update.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from modelops_lab.db import (
    ADVISORY_LOCK_KEY,
    ActiveModelRow,
    Base,
    ModelTransitionRow,
    PromotionDecisionRow,
    _session_factory,
    create_engine_for,
    utc,
)

DecisionJsonWriter = Callable[[], Path]


class PointerError(RuntimeError):
    """Raised when the active pointer cannot be read or written."""


class PromotionConflict(PointerError):
    """Raised when the active pointer moved after the gate captured its version."""

    def __init__(self, expected: str | None, observed: str | None) -> None:
        self.expected = expected
        self.observed = observed
        super().__init__(
            "active model changed during promotion: expected version "
            f"{expected or '<none>'}, found {observed or '<none>'}. "
            "Nothing was written; re-run the promotion gate against the current pointer."
        )


class PointerTransactionError(PointerError):
    """Raised when the pointer transaction failed after the alias moved."""


@dataclass(frozen=True)
class ActiveModel:
    """Materialized active_model row."""

    model_name: str
    model_version: str
    mlflow_run_id: str
    mlflow_model_id: str
    dataset_version: str
    dataset_fingerprint: str
    decision_id: str
    promoted_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DecisionRecord:
    """Database representation of one promotion decision."""

    decision_id: str
    candidate_run_id: str
    candidate_model_version: str
    active_model_version_before: str | None
    verdict: str
    checks: list[dict[str, Any]]
    rejection_reasons: list[str]
    actor: str
    reason: str
    decided_at: datetime
    dataset_version: str
    dataset_fingerprint: str
    evaluation_split: str


@dataclass(frozen=True)
class TransitionRecord:
    """Materialized append-only transition row."""

    id: int
    operation: str
    model_name: str
    from_model_version: str | None
    to_model_version: str
    run_id: str
    decision_id: str
    actor: str
    reason: str
    occurred_at: datetime


def _active_from_row(row: ActiveModelRow) -> ActiveModel:
    return ActiveModel(
        model_name=row.model_name,
        model_version=row.model_version,
        mlflow_run_id=row.mlflow_run_id,
        mlflow_model_id=row.mlflow_model_id,
        dataset_version=row.dataset_version,
        dataset_fingerprint=row.dataset_fingerprint,
        decision_id=row.decision_id,
        promoted_at=utc(row.promoted_at),
        updated_at=utc(row.updated_at),
    )


def _decision_row(record: DecisionRecord) -> PromotionDecisionRow:
    return PromotionDecisionRow(
        decision_id=record.decision_id,
        candidate_run_id=record.candidate_run_id,
        candidate_model_version=record.candidate_model_version,
        active_model_version_before=record.active_model_version_before,
        verdict=record.verdict,
        checks=record.checks,
        rejection_reasons=record.rejection_reasons,
        actor=record.actor,
        reason=record.reason,
        decided_at=record.decided_at,
        dataset_version=record.dataset_version,
        dataset_fingerprint=record.dataset_fingerprint,
        evaluation_split=record.evaluation_split,
    )


class PointerStore:
    """Postgres/SQLite-backed active pointer plus its audit tables."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = create_engine_for(database_url)
        self._session_factory = _session_factory(self.engine)
        self._is_postgres = self.engine.dialect.name == "postgresql"

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
        finally:
            session.close()

    def create_schema(self) -> None:
        """Create the tables directly (unit tests; migrations are authoritative)."""
        Base.metadata.create_all(self.engine)

    def active(self) -> ActiveModel | None:
        """Read the singleton pointer, or ``None`` before the first promotion."""
        try:
            with self.session() as session:
                row = session.get(ActiveModelRow, 1)
                return _active_from_row(row) if row is not None else None
        except Exception as exc:
            raise PointerError(f"cannot read the active model pointer: {exc}") from exc

    def decisions(self) -> list[DecisionRecord]:
        with self.session() as session:
            rows = session.query(PromotionDecisionRow).order_by(
                PromotionDecisionRow.decided_at, PromotionDecisionRow.decision_id
            )
            return [
                DecisionRecord(
                    decision_id=row.decision_id,
                    candidate_run_id=row.candidate_run_id,
                    candidate_model_version=row.candidate_model_version,
                    active_model_version_before=row.active_model_version_before,
                    verdict=row.verdict,
                    checks=list(row.checks),
                    rejection_reasons=list(row.rejection_reasons),
                    actor=row.actor,
                    reason=row.reason,
                    decided_at=utc(row.decided_at),
                    dataset_version=row.dataset_version,
                    dataset_fingerprint=row.dataset_fingerprint,
                    evaluation_split=row.evaluation_split,
                )
                for row in rows
            ]

    def decision(self, decision_id: str) -> DecisionRecord | None:
        """Return one decision row by id, or ``None`` when it is absent."""
        with self.session() as session:
            row = session.get(PromotionDecisionRow, decision_id)
            if row is None:
                return None
            return DecisionRecord(
                decision_id=row.decision_id,
                candidate_run_id=row.candidate_run_id,
                candidate_model_version=row.candidate_model_version,
                active_model_version_before=row.active_model_version_before,
                verdict=row.verdict,
                checks=list(row.checks),
                rejection_reasons=list(row.rejection_reasons),
                actor=row.actor,
                reason=row.reason,
                decided_at=utc(row.decided_at),
                dataset_version=row.dataset_version,
                dataset_fingerprint=row.dataset_fingerprint,
                evaluation_split=row.evaluation_split,
            )

    def transitions(self) -> list[TransitionRecord]:
        with self.session() as session:
            rows = session.query(ModelTransitionRow).order_by(ModelTransitionRow.id)
            return [
                TransitionRecord(
                    id=row.id,
                    operation=row.operation,
                    model_name=row.model_name,
                    from_model_version=row.from_model_version,
                    to_model_version=row.to_model_version,
                    run_id=row.run_id,
                    decision_id=row.decision_id,
                    actor=row.actor,
                    reason=row.reason,
                    occurred_at=utc(row.occurred_at),
                )
                for row in rows
            ]

    def reset(self) -> None:
        """Delete every row. Integration-test helper; never used by the CLI."""
        with self.engine.begin() as connection:
            if self._is_postgres:
                connection.execute(
                    text(
                        "TRUNCATE TABLE prediction_labels, prediction_events, "
                        "promotion_decisions, model_transitions, active_model"
                    )
                )
            else:
                connection.execute(text("DELETE FROM prediction_labels"))
                connection.execute(text("DELETE FROM prediction_events"))
                connection.execute(text("DELETE FROM promotion_decisions"))
                connection.execute(text("DELETE FROM model_transitions"))
                connection.execute(text("DELETE FROM active_model"))

    def record_decision(
        self, record: DecisionRecord, *, write_decision_json: DecisionJsonWriter
    ) -> Path:
        """Persist a rejected decision (and its JSON artifact) without moving state."""
        path = write_decision_json()
        try:
            with self.session() as session, session.begin():
                session.add(_decision_row(record))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def apply_promotion(
        self,
        *,
        decision: DecisionRecord,
        expected_active_version: str | None,
        operation: str,
        target: ActiveModel,
        set_alias: Callable[[], None],
        restore_alias: Callable[[], None],
        write_decision_json: DecisionJsonWriter,
    ) -> Path:
        """Atomically move the pointer and record the decision and transition.

        ``set_alias`` is invoked inside the critical section after the conflict
        check. If anything after that point fails (including the commit), the
        alias is restored and a :class:`PointerTransactionError` is raised.
        """
        session = self._session_factory()
        alias_moved = False
        decision_path: Path | None = None
        try:
            with session.begin():
                if self._is_postgres:
                    session.execute(
                        text("SELECT pg_advisory_xact_lock(:key)"),
                        {"key": ADVISORY_LOCK_KEY},
                    )
                row = session.get(ActiveModelRow, 1)
                observed = row.model_version if row is not None else None
                if observed != expected_active_version:
                    raise PromotionConflict(expected_active_version, observed)

                set_alias()
                alias_moved = True
                session.add(_decision_row(decision))
                session.add(
                    ModelTransitionRow(
                        operation=operation,
                        model_name=target.model_name,
                        from_model_version=expected_active_version,
                        to_model_version=target.model_version,
                        run_id=target.mlflow_run_id,
                        decision_id=decision.decision_id,
                        actor=decision.actor,
                        reason=decision.reason,
                        occurred_at=decision.decided_at,
                    )
                )
                decision_path = write_decision_json()

                if row is None:
                    session.add(
                        ActiveModelRow(
                            id=1,
                            model_name=target.model_name,
                            model_version=target.model_version,
                            mlflow_run_id=target.mlflow_run_id,
                            mlflow_model_id=target.mlflow_model_id,
                            dataset_version=target.dataset_version,
                            dataset_fingerprint=target.dataset_fingerprint,
                            decision_id=target.decision_id,
                            promoted_at=target.promoted_at,
                            updated_at=target.updated_at,
                        )
                    )
                else:
                    row.model_name = target.model_name
                    row.model_version = target.model_version
                    row.mlflow_run_id = target.mlflow_run_id
                    row.mlflow_model_id = target.mlflow_model_id
                    row.dataset_version = target.dataset_version
                    row.dataset_fingerprint = target.dataset_fingerprint
                    row.decision_id = target.decision_id
                    row.promoted_at = target.promoted_at
                    row.updated_at = target.updated_at
        except PromotionConflict:
            raise
        except Exception as exc:
            if decision_path is not None:
                decision_path.unlink(missing_ok=True)
            if alias_moved:
                try:
                    restore_alias()
                except Exception as restore_exc:
                    raise PointerTransactionError(
                        "promotion failed after the registry alias moved "
                        f"({exc}); restoring the previous alias also failed ({restore_exc}). "
                        "Reconciliation is required: the alias may point at the candidate."
                    ) from exc
                raise PointerTransactionError(
                    f"promotion failed after the registry alias moved ({exc}); the previous "
                    "alias was restored and no database state was committed."
                ) from exc
            raise
        finally:
            session.close()
        assert decision_path is not None
        return decision_path

    def apply_rollback(
        self,
        *,
        expected_active_version: str,
        target: ActiveModel,
        transition_decision_id: str,
        actor: str,
        reason: str,
        occurred_at: datetime,
        set_alias: Callable[[], None],
        restore_alias: Callable[[], None],
        write_rollback_json: Callable[[], Path],
    ) -> Path:
        """Atomically restore a previously active version.

        Reuses the promotion critical section: advisory lock, re-read of the
        pointer, expected-version conflict, alias move, transition append, and
        the pointer update last. A conflict aborts before the alias is touched.
        """
        session = self._session_factory()
        alias_moved = False
        audit_path: Path | None = None
        try:
            with session.begin():
                if self._is_postgres:
                    session.execute(
                        text("SELECT pg_advisory_xact_lock(:key)"),
                        {"key": ADVISORY_LOCK_KEY},
                    )
                row = session.get(ActiveModelRow, 1)
                observed = row.model_version if row is not None else None
                if observed != expected_active_version:
                    raise PromotionConflict(expected_active_version, observed)

                set_alias()
                alias_moved = True
                session.add(
                    ModelTransitionRow(
                        operation="rollback",
                        model_name=target.model_name,
                        from_model_version=expected_active_version,
                        to_model_version=target.model_version,
                        run_id=target.mlflow_run_id,
                        decision_id=transition_decision_id,
                        actor=actor,
                        reason=reason,
                        occurred_at=occurred_at,
                    )
                )
                audit_path = write_rollback_json()
                assert row is not None  # expected_active_version is a concrete version
                row.model_name = target.model_name
                row.model_version = target.model_version
                row.mlflow_run_id = target.mlflow_run_id
                row.mlflow_model_id = target.mlflow_model_id
                row.dataset_version = target.dataset_version
                row.dataset_fingerprint = target.dataset_fingerprint
                row.decision_id = target.decision_id
                row.promoted_at = target.promoted_at
                row.updated_at = target.updated_at
        except PromotionConflict:
            raise
        except Exception as exc:
            if audit_path is not None:
                audit_path.unlink(missing_ok=True)
            if alias_moved:
                try:
                    restore_alias()
                except Exception as restore_exc:
                    raise PointerTransactionError(
                        "rollback failed after the registry alias moved "
                        f"({exc}); restoring the previous alias also failed ({restore_exc}). "
                        "Reconciliation is required: the alias may point at the rollback target."
                    ) from exc
                raise PointerTransactionError(
                    f"rollback failed after the registry alias moved ({exc}); the previous "
                    "alias was restored and no database state was committed."
                ) from exc
            raise
        finally:
            session.close()
        assert audit_path is not None
        return audit_path

    def apply_alias_reconciliation(
        self,
        *,
        expected_active_version: str | None,
        set_alias: Callable[[], None],
        delete_alias: Callable[[], None],
    ) -> None:
        """Move the alias onto the pointer under the promotion lock.

        Used by ``modelops registry reconcile --fix``. The pointer is re-read
        inside the transaction; a concurrent move aborts with
        :class:`PromotionConflict` before the alias is touched.
        """
        with self.session() as session, session.begin():
            if self._is_postgres:
                session.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": ADVISORY_LOCK_KEY},
                )
            row = session.get(ActiveModelRow, 1)
            observed = row.model_version if row is not None else None
            if observed != expected_active_version:
                raise PromotionConflict(expected_active_version, observed)
            if observed is None:
                delete_alias()
            else:
                set_alias()


def read_decision_file(path: Path) -> dict[str, Any]:
    """Read a decision JSON artifact (used by tests and the status page later)."""
    return json.loads(path.read_text(encoding="utf-8"))
