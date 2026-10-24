"""Monitoring degrades gracefully when the model cannot be loaded or scored."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from modelops_lab.config import load_monitoring_config
from modelops_lab.contracts import DriftVerdict, PredictionEvent
from modelops_lab.data import LoadedDataset, split_frame
from modelops_lab.db import Base, create_engine_for
from modelops_lab.monitoring import describe_result, run_monitoring
from modelops_lab.serving import SqlPredictionEventStore, SqlPredictionLabelStore

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _seed_events(database_url: str, dataset: LoadedDataset) -> None:
    Base.metadata.create_all(create_engine_for(database_url))
    frame = split_frame(dataset, "validation").sort_values("row_id").reset_index(drop=True)
    features = list(dataset.schema.feature_names)
    event_store = SqlPredictionEventStore(database_url)
    label_store = SqlPredictionLabelStore(database_url)
    for index, row in frame.iterrows():
        event_id = event_store.record(
            PredictionEvent(
                request_id=f"degraded-{index}",
                timestamp=NOW + timedelta(milliseconds=index),
                features={
                    name: (
                        float(row[name])
                        if name in dataset.schema.numeric_features
                        else str(row[name])
                    )
                    for name in features
                },
                prediction=0,
                probability=0.5,
                model_version="1",
                latency_ms=1.0,
            )
        )
        label_store.upsert(event_id, int(row["target"]))


def test_unloadable_model_still_produces_a_data_drift_report(
    tmp_path: Path, loaded_dataset: LoadedDataset
) -> None:
    database_url = f"sqlite:///{tmp_path / 'degraded.db'}"
    _seed_events(database_url, loaded_dataset)
    output = tmp_path / "reports"

    def failing_loader(version: str) -> object:
        raise RuntimeError(f"registry is offline for version {version}")

    run = run_monitoring(
        config=load_monitoring_config("configs/monitoring.yaml"),
        database_url=database_url,
        dataset=loaded_dataset,
        model_version="1",
        model_loader=failing_loader,
        output_dir=output,
        now=NOW,
    )

    assert run.json_path.exists() and run.html_path.exists()
    payload = json.loads(run.json_path.read_text(encoding="utf-8"))
    assert payload["data_drift"]["verdict"] == DriftVerdict.NOT_DRIFTED.value
    assert payload["performance_degradation"]["verdict"] == DriftVerdict.NOT_EVALUATED.value
    assert payload["concept_drift"]["verdict"] == DriftVerdict.NOT_EVALUATED.value
    assert payload["performance"] is None
    assert "registry is offline" in payload["performance_degradation"]["detail"]
    assert "registry is offline" in payload["concept_drift"]["detail"]
    assert payload["current_window"]["row_count"] == 400

    summary = "\n".join(describe_result(run))
    assert "data_drift: not_drifted" in summary
    assert "performance_degradation: not_evaluated" in summary
    assert "concept_drift: not_evaluated" in summary
    assert "registry is offline" in summary

    html = run.html_path.read_text(encoding="utf-8")
    assert "registry is offline" in html
    assert "Traceback" not in html


def test_operator_error_is_one_redacted_line() -> None:
    from modelops_lab.monitoring import operator_error

    message = operator_error(
        RuntimeError("cannot reach postgresql://user:secret@db:5432/modelops\nsecond line")
    )

    assert message.startswith("RuntimeError: ")
    assert "secret" not in message
    assert "***" in message
    assert "\n" not in message
    assert len(message) <= 320


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, KeyError])
def test_operator_error_names_the_exception_type(failure: type[Exception]) -> None:
    from modelops_lab.monitoring import operator_error

    message = operator_error(failure("boom"))
    assert message.startswith(f"{failure.__name__}:")
    assert "boom" in message
