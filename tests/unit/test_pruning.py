"""Retention pruning deletes old events and their labels only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from modelops_lab.db import (
    Base,
    PredictionEventRow,
    PredictionLabelRow,
    create_engine_for,
)
from modelops_lab.pruning import prune_events

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
FEATURES = {"age": 54.0, "bmi": 27.3, "sex": "female"}


def _seed(database_url: str) -> None:
    engine = create_engine_for(database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        for index, age_days in enumerate((90, 45, 10, 1)):
            session.add(
                PredictionEventRow(
                    request_id=f"req-{index}",
                    timestamp=NOW - timedelta(days=age_days),
                    features=FEATURES,
                    prediction=index % 2,
                    probability=0.5,
                    model_version="1",
                    latency_ms=1.0,
                )
            )
        session.flush()
        session.add_all(
            [
                PredictionLabelRow(event_id=1, label=0, labeled_at=NOW - timedelta(days=89)),
                PredictionLabelRow(event_id=2, label=1, labeled_at=NOW - timedelta(days=44)),
                PredictionLabelRow(event_id=3, label=0, labeled_at=NOW - timedelta(days=9)),
                PredictionLabelRow(event_id=4, label=1, labeled_at=NOW - timedelta(days=1)),
            ]
        )


def _counts(database_url: str) -> tuple[int, int]:
    factory = sessionmaker(bind=create_engine_for(database_url))
    with factory() as session:
        return (
            session.query(PredictionEventRow).count(),
            session.query(PredictionLabelRow).count(),
        )


def test_prune_deletes_old_events_and_labels_and_keeps_recent(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'prune.db'}"
    _seed(database_url)

    result = prune_events(database_url=database_url, days=30, now=NOW)

    assert result.events == 2
    assert result.labels == 2
    assert result.dry_run is False
    assert _counts(database_url) == (2, 2)
    factory = sessionmaker(bind=create_engine_for(database_url))
    with factory() as session:
        remaining = [row.event_id for row in session.query(PredictionEventRow).all()]
        labels = [row.event_id for row in session.query(PredictionLabelRow).all()]
    assert remaining == [3, 4]
    assert labels == [3, 4]


def test_dry_run_counts_without_changing_anything(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'prune.db'}"
    _seed(database_url)

    result = prune_events(database_url=database_url, days=30, dry_run=True, now=NOW)

    assert result.events == 2
    assert result.labels == 2
    assert result.dry_run is True
    assert _counts(database_url) == (4, 4)


def test_prune_keeps_the_label_of_a_surviving_boundary_event(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'prune.db'}"
    _seed(database_url)

    prune_events(database_url=database_url, days=1, now=NOW)

    assert _counts(database_url) == (1, 1)
    factory = sessionmaker(bind=create_engine_for(database_url))
    with factory() as session:
        remaining = [row.event_id for row in session.query(PredictionEventRow).all()]
        labels = [row.event_id for row in session.query(PredictionLabelRow).all()]
    assert remaining == [4]
    assert labels == [4]
