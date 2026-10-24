"""Retention pruning for prediction events and their delayed labels.

The retention window is ``MODELOPS_EVENT_RETENTION_DAYS`` (default 30). A
label is removed only when its own event is removed, so cleanup can never
delete a label for a surviving event. ``--dry-run`` counts what would be
deleted without changing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from modelops_lab.db import PredictionEventRow, PredictionLabelRow, create_engine_for


class PruneError(RuntimeError):
    """Raised when pruning cannot reach the database."""


@dataclass(frozen=True)
class PruneResult:
    cutoff: datetime
    events: int
    labels: int
    dry_run: bool

    def describe(self) -> list[str]:
        action = "would delete" if self.dry_run else "deleted"
        lines = [
            f"retention_cutoff: {self.cutoff.isoformat()}",
            f"events: {self.events} {action}",
            f"labels: {self.labels} {action}",
        ]
        if self.dry_run:
            lines.append("dry run: no rows were changed")
        return lines


def prune_events(
    *,
    database_url: str,
    days: int,
    dry_run: bool = False,
    now: datetime | None = None,
) -> PruneResult:
    """Delete events older than ``days`` and the labels attached to them."""
    if days < 1:
        raise PruneError(f"retention days must be at least 1, got {days}")
    cutoff = (now or datetime.now(UTC)).astimezone(UTC) - timedelta(days=days)
    engine = create_engine_for(database_url)
    factory: sessionmaker[Session] = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    old_events = select(PredictionEventRow.event_id).where(PredictionEventRow.timestamp < cutoff)
    try:
        with factory() as session, session.begin():
            if dry_run:
                events = int(
                    session.execute(
                        select(func.count())
                        .select_from(PredictionEventRow)
                        .where(PredictionEventRow.timestamp < cutoff)
                    ).scalar_one()
                )
                labels = int(
                    session.execute(
                        select(func.count())
                        .select_from(PredictionLabelRow)
                        .where(PredictionLabelRow.event_id.in_(old_events))
                    ).scalar_one()
                )
                return PruneResult(cutoff=cutoff, events=events, labels=labels, dry_run=True)
            labels = int(
                session.execute(
                    delete(PredictionLabelRow).where(PredictionLabelRow.event_id.in_(old_events))
                ).rowcount
                or 0
            )
            events = int(
                session.execute(
                    delete(PredictionEventRow).where(PredictionEventRow.timestamp < cutoff)
                ).rowcount
                or 0
            )
            return PruneResult(cutoff=cutoff, events=events, labels=labels, dry_run=False)
    except PruneError:
        raise
    except Exception as exc:
        raise PruneError(f"cannot prune prediction events: {exc}") from exc
