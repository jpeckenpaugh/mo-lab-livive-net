"""Docker-free tests for the inference app, provider, and event logging."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from modelops_lab.config import RuntimeSettings
from modelops_lab.contracts import PredictionEvent
from modelops_lab.data import LoadedDataset
from modelops_lab.db import Base, PredictionEventRow, create_engine_for
from modelops_lab.pointer import ActiveModel, DecisionRecord
from modelops_lab.serving import (
    PointerModelProvider,
    ServingMetrics,
    SqlPredictionEventStore,
    _request_id,
    create_app,
    feature_spec_from_model,
)
from modelops_lab.train import train_on_dataset

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
FEATURES = {"age": 54.0, "bmi": 27.3, "sex": "female"}


def make_pipeline(positive_probability: float) -> Any:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.DataFrame({"sex": ["female", "male"]})
    )
    preprocess = SimpleNamespace(transformers_=[("categorical", encoder, ["sex"])])

    class FakePipeline:
        feature_names_in_ = np.array(["age", "bmi", "sex"], dtype=object)
        named_steps = {"preprocess": preprocess}
        classes_ = np.array([0, 1])

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            assert list(frame.columns) == ["age", "bmi", "sex"]
            values = np.full(len(frame), positive_probability, dtype=float)
            return np.column_stack([1.0 - values, values])

    return FakePipeline()


def make_pointer(version: str = "2", model_name: str = "test-model") -> ActiveModel:
    return ActiveModel(
        model_name=model_name,
        model_version=version,
        mlflow_run_id=f"run-{version}",
        mlflow_model_id=f"m-{version}",
        dataset_version="synthetic-v1.0.0-test",
        dataset_fingerprint="a" * 64,
        decision_id="dec-1",
        promoted_at=NOW,
        updated_at=NOW,
    )


class FakeEventStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[Any] = []
        self.fail = fail

    def record(self, event: Any) -> None:
        if self.fail:
            raise RuntimeError("database is down")
        self.events.append(event)

    def ping(self) -> bool:
        return not self.fail


class Harness:
    def __init__(self, *, fail_events: bool = False) -> None:
        self.metrics = ServingMetrics()
        self.events = FakeEventStore(fail=fail_events)
        self.pointer: ActiveModel | None = make_pointer("2")
        self.models = {"1": make_pipeline(0.2), "2": make_pipeline(0.9)}
        self.loads: list[str] = []
        self.load_error: Exception | None = None
        self.alias_version: str | None = "2"
        self.decision: DecisionRecord | None = DecisionRecord(
            decision_id="dec-1",
            candidate_run_id="run-2",
            candidate_model_version="2",
            active_model_version_before="1",
            verdict="approved",
            checks=[],
            rejection_reasons=[],
            actor="release-manager",
            reason="passed the gate",
            decided_at=NOW,
            dataset_version="synthetic-v1.0.0-test",
            dataset_fingerprint="a" * 64,
            evaluation_split="validation",
        )

        def loader(name: str, version: str) -> Any:
            self.loads.append(version)
            if self.load_error is not None:
                raise self.load_error
            return self.models[version]

        def decision_reader(decision_id: str) -> DecisionRecord | None:
            return self.decision

        self.provider = PointerModelProvider(
            pointer_reader=lambda: self.pointer,
            loader=loader,
            metrics=self.metrics,
            alias_reader=lambda: self.alias_version,
            decision_reader=decision_reader,
            database_ping=self.events.ping,
            alias_check_interval_seconds=0.0,
        )
        self.app = create_app(
            settings=RuntimeSettings(database_url="sqlite://"),
            provider=self.provider,
            event_store=self.events,
            metrics=self.metrics,
            logger=logging.getLogger("test.serving"),
        )
        self.client = TestClient(self.app)

    def predict(self, features: dict[str, Any] | None = None, **kwargs: Any):
        return self.client.post("/predict", json=features or FEATURES, **kwargs)


@pytest.fixture()
def harness() -> Harness:
    return Harness()


def test_feature_spec_comes_from_the_fitted_pipeline(
    training_config: Any, loaded_dataset: LoadedDataset
) -> None:
    result = train_on_dataset(training_config, loaded_dataset)

    spec = feature_spec_from_model(result.pipeline)

    assert list(spec.names) == list(loaded_dataset.manifest.feature_names)
    assert set(spec.categorical) == {"sex", "chest_pain_type", "smoker"}
    assert "age" in spec.numeric
    assert spec.categorical["sex"] == ("female", "male")


def test_health_reports_no_model_without_failing(harness: Harness) -> None:
    harness.pointer = None

    response = harness.client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database_reachable"] is True
    assert body["model_loaded"] is False
    assert body["active_model_version"] is None


def test_health_reports_loaded_model_after_a_request(harness: Harness) -> None:
    assert harness.client.get("/health").json()["model_loaded"] is False

    harness.predict()

    health = harness.client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["active_model_version"] == "2"


def test_model_descriptor_includes_approval_metadata(harness: Harness) -> None:
    response = harness.client.get("/model")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "2"
    assert body["mlflow_run_id"] == "run-2"
    assert body["dataset_version"] == "synthetic-v1.0.0-test"
    assert body["dataset_fingerprint"] == "a" * 64
    assert body["approval"]["decision_id"] == "dec-1"
    assert body["approval"]["approved_by"] == "release-manager"
    assert body["features"]["features"] == ["age", "bmi", "sex"]


def test_model_and_predict_are_503_without_an_active_model(harness: Harness) -> None:
    harness.pointer = None

    model_response = harness.client.get("/model")
    predict_response = harness.predict()

    assert model_response.status_code == 503
    assert "no model is active" in model_response.json()["detail"]["message"]
    assert predict_response.status_code == 503
    assert predict_response.json()["detail"]["error"] == "model_unavailable"


def test_predict_happy_path_and_request_id_echo(harness: Harness) -> None:
    response = harness.predict(headers={"X-Request-ID": "req-fixed-1"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-fixed-1"
    body = response.json()
    assert body["request_id"] == "req-fixed-1"
    assert body["model_version"] == "2"
    assert body["prediction"] == 1
    assert body["probability"] == pytest.approx(0.9)


def test_predict_generates_a_uuid_request_id(harness: Harness) -> None:
    response = harness.predict()

    request_id = response.json()["request_id"]
    assert response.headers["X-Request-ID"] == request_id
    UUID(request_id)


def test_predict_rejects_an_overlong_request_id(harness: Harness) -> None:
    response = harness.predict(headers={"X-Request-ID": "a" * 129})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "invalid_request_id"
    assert detail["errors"][0]["field"] == "X-Request-ID"


def test_request_id_rejects_non_printable_characters() -> None:
    request = Request({"type": "http", "headers": [(b"x-request-id", b"retry\x01id")]})

    with pytest.raises(HTTPException) as error:
        _request_id(request)

    assert error.value.status_code == 422
    assert error.value.detail["errors"][0]["field"] == "X-Request-ID"


def test_duplicate_request_ids_are_both_persisted(tmp_path: Any) -> None:
    database_url = f"sqlite:///{tmp_path / 'events.db'}"
    engine = create_engine_for(database_url)
    Base.metadata.create_all(engine)
    store = SqlPredictionEventStore(database_url)
    event = PredictionEvent(
        request_id="client-retry-1",
        timestamp=NOW,
        features=FEATURES,
        prediction=0,
        probability=0.42,
        model_version="2",
        latency_ms=3.5,
    )

    store.record(event)
    store.record(event)

    with sessionmaker(bind=engine)() as session:
        rows = session.query(PredictionEventRow).order_by(PredictionEventRow.event_id).all()
    assert [row.request_id for row in rows] == ["client-retry-1", "client-retry-1"]
    assert rows[0].event_id != rows[1].event_id
    assert [row.probability for row in rows] == [0.42, 0.42]


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"age": 54.0, "sex": "female"}, "bmi"),
        ({**FEATURES, "unexpected": 1.0}, "unexpected"),
        ({**FEATURES, "age": "54"}, "age"),
        ({**FEATURES, "sex": "unknown"}, "sex"),
        ({**FEATURES, "sex": 1}, "sex"),
    ],
)
def test_predict_422_names_the_invalid_field(
    harness: Harness, payload: dict[str, Any], field: str
) -> None:
    response = harness.client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "invalid_features"
    assert field in {error["field"] for error in detail["errors"]}


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_values_without_coercion(harness: Harness, literal: str) -> None:
    response = harness.client.post(
        "/predict",
        content=f'{{"age": {literal}, "bmi": 27.3, "sex": "female"}}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert "age" in {error["field"] for error in response.json()["detail"]["errors"]}


def test_predict_rejects_an_integer_too_large_for_a_float(harness: Harness) -> None:
    huge = "1" + "0" * 400
    response = harness.client.post(
        "/predict",
        content=f'{{"age": {huge}, "bmi": 27.3, "sex": "female"}}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert "age" in {error["field"] for error in response.json()["detail"]["errors"]}


def test_predict_rejects_a_non_object_body(harness: Harness) -> None:
    response = harness.client.post("/predict", json=[1, 2, 3])

    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["field"] == "<body>"


def test_event_is_recorded_with_version_request_and_latency(harness: Harness) -> None:
    response = harness.predict(headers={"X-Request-ID": "req-event-1"})

    assert response.status_code == 200
    assert len(harness.events.events) == 1
    event = harness.events.events[0]
    assert event.request_id == "req-event-1"
    assert event.model_version == "2"
    assert event.prediction == 1
    assert event.probability == pytest.approx(0.9)
    assert event.latency_ms >= 0.0
    assert event.features == FEATURES
    assert event.schema_version == "1.1.0"


def test_event_insert_failure_never_fails_inference() -> None:
    harness = Harness(fail_events=True)

    response = harness.predict()

    assert response.status_code == 200
    assert response.json()["model_version"] == "2"
    assert harness.events.events == []
    exposition = generate_latest(harness.metrics.registry).decode("utf-8")
    assert "modelops_event_log_failures_total 1.0" in exposition


def test_pointer_move_serves_the_new_version_without_rebuilding(harness: Harness) -> None:
    harness.pointer = make_pointer("1")
    first = harness.predict()
    assert first.json()["model_version"] == "1"
    assert first.json()["prediction"] == 0

    harness.pointer = make_pointer("2")
    second = harness.predict()

    assert second.json()["model_version"] == "2"
    assert second.json()["prediction"] == 1
    assert harness.loads == ["1", "2"]


def test_pointer_wins_when_the_alias_disagrees(harness: Harness) -> None:
    harness.pointer = make_pointer("1")
    harness.alias_version = "2"

    response = harness.predict()

    assert response.json()["model_version"] == "1"
    exposition = generate_latest(harness.metrics.registry).decode("utf-8")
    assert "modelops_registry_alias_mismatch_total 1.0" in exposition


def test_metrics_expose_the_documented_series(harness: Harness) -> None:
    harness.predict()
    harness.predict()

    response = harness.client.get("/metrics")

    assert response.status_code == 200
    text = response.text
    assert 'modelops_predictions_total{model_version="2",outcome="success"} 2.0' in text
    assert "modelops_prediction_latency_seconds_bucket" in text
    assert "modelops_model_load_failures_total" in text
    assert 'modelops_active_model_info{model_name="test-model",model_version="2"' in text
    assert "modelops_event_writes_total" in text


def test_structured_prediction_log_line_has_no_features(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    harness.predict(headers={"X-Request-ID": "req-log-1"})

    lines = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    success = next(line for line in lines if line.get("status") == "success")
    assert success["request_id"] == "req-log-1"
    assert success["model_version"] == "2"
    assert "latency_ms" in success
    assert "features" not in success
    assert "sex" not in json.dumps(success)
