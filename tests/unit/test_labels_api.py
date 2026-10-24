"""POST /labels: success, unknown event, invalid body, and idempotent relabel."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.orm import sessionmaker

from modelops_lab.config import RuntimeSettings
from modelops_lab.contracts import PredictionEvent
from modelops_lab.db import Base, PredictionEventRow, PredictionLabelRow, create_engine_for
from modelops_lab.pointer import ActiveModel
from modelops_lab.serving import (
    PointerModelProvider,
    ServingMetrics,
    SqlPredictionEventStore,
    SqlPredictionLabelStore,
    create_app,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
FEATURES = {"age": 54.0, "bmi": 27.3, "sex": "female"}


def _pipeline() -> object:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.DataFrame({"sex": ["female", "male"]})
    )

    class FakePipeline:
        feature_names_in_ = np.array(["age", "bmi", "sex"], dtype=object)
        named_steps = {"preprocess": SimpleNamespace(transformers_=[("cat", encoder, ["sex"])])}
        classes_ = np.array([0, 1])

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            values = np.full(len(frame), 0.5)
            return np.column_stack([1.0 - values, values])

    return FakePipeline()


def _pointer() -> ActiveModel:
    return ActiveModel(
        model_name="test-model",
        model_version="1",
        mlflow_run_id="run-1",
        mlflow_model_id="m-1",
        dataset_version="synthetic-v1.0.0-test",
        dataset_fingerprint="a" * 64,
        decision_id="dec-1",
        promoted_at=NOW,
        updated_at=NOW,
    )


def _client(tmp_path: Path) -> tuple[TestClient, str, int]:
    database_url = f"sqlite:///{tmp_path / 'labels.db'}"
    engine = create_engine_for(database_url)
    Base.metadata.create_all(engine)
    event_id = SqlPredictionEventStore(database_url).record(
        PredictionEvent(
            request_id="req-1",
            timestamp=NOW,
            features=FEATURES,
            prediction=1,
            probability=0.8,
            model_version="1",
            latency_ms=2.0,
        )
    )
    metrics = ServingMetrics()
    provider = PointerModelProvider(
        pointer_reader=_pointer,
        loader=lambda _name, _version: _pipeline(),
        metrics=metrics,
        alias_check_interval_seconds=0.0,
    )
    app = create_app(
        settings=RuntimeSettings(database_url=database_url),
        provider=provider,
        event_store=SqlPredictionEventStore(database_url),
        label_store=SqlPredictionLabelStore(database_url),
        metrics=metrics,
    )
    return TestClient(app), database_url, event_id


def test_label_upsert_is_idempotent_and_relabels(tmp_path: Path) -> None:
    client, database_url, event_id = _client(tmp_path)

    first = client.post("/labels", json={"event_id": event_id, "label": 1})
    assert first.status_code == 200, first.text
    assert first.json() == {
        "event_id": event_id,
        "label": 1,
        "labeled_at": first.json()["labeled_at"],
        "updated": False,
    }

    second = client.post("/labels", json={"event_id": event_id, "label": 0})
    assert second.status_code == 200
    assert second.json()["label"] == 0
    assert second.json()["updated"] is True

    with sessionmaker(bind=create_engine_for(database_url))() as session:
        rows = session.query(PredictionLabelRow).all()
    assert len(rows) == 1
    assert rows[0].event_id == event_id
    assert rows[0].label == 0


def test_label_for_unknown_event_is_404(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)

    response = client.post("/labels", json={"event_id": 999999, "label": 1})

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "unknown_event"


def test_invalid_label_bodies_are_422(tmp_path: Path) -> None:
    client, _, event_id = _client(tmp_path)

    for body in (
        {"event_id": event_id},
        {"event_id": event_id, "label": 2},
        {"event_id": event_id, "label": True},
        {"event_id": 0, "label": 1},
        {"event_id": "one", "label": 1},
        {"event_id": event_id, "label": 1, "extra": "nope"},
    ):
        response = client.post("/labels", json=body)
        assert response.status_code == 422, (body, response.text)


def test_event_id_is_returned_by_predict(tmp_path: Path) -> None:
    client, database_url, _ = _client(tmp_path)

    response = client.post("/predict", json=FEATURES)

    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["event_id"], int)
    with sessionmaker(bind=create_engine_for(database_url))() as session:
        event = session.get(PredictionEventRow, body["event_id"])
    assert event is not None
    assert event.request_id == body["request_id"]


def test_event_id_is_null_when_event_persistence_fails(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'labels.db'}"

    class FailingStore:
        def record(self, event: PredictionEvent) -> int:
            raise RuntimeError("database is down")

        def ping(self) -> bool:
            return False

    metrics = ServingMetrics()
    provider = PointerModelProvider(
        pointer_reader=_pointer,
        loader=lambda _name, _version: _pipeline(),
        metrics=metrics,
        alias_check_interval_seconds=0.0,
    )
    app = create_app(
        settings=RuntimeSettings(database_url=database_url),
        provider=provider,
        event_store=FailingStore(),
        label_store=SqlPredictionLabelStore(database_url),
        metrics=metrics,
    )
    client = TestClient(app)

    response = client.post("/predict", json=FEATURES)

    assert response.status_code == 200
    assert response.json()["event_id"] is None
    assert response.json()["prediction"] == 1
