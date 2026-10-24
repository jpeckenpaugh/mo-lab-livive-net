"""Drift report contract round-trip and JSON/HTML artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from modelops_lab.config import load_monitoring_config
from modelops_lab.contracts import DriftReport, DriftVerdict, WindowSummary
from modelops_lab.monitoring import analyze_windows, load_report, write_drift_report


def _result() -> object:
    config = load_monitoring_config("configs/monitoring.yaml")
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "value": rng.normal(0.0, 1.0, config.min_sample_size),
            "category": rng.choice(["a", "b"], size=config.min_sample_size),
        }
    )
    return analyze_windows(
        model_version="7",
        reference_frame=frame,
        current_frame=frame,
        reference_window=WindowSummary(
            label="reference",
            source="dataset:synthetic-test:validation_split",
            row_count=len(frame),
        ),
        current_window=WindowSummary(
            label="current", source="prediction_events:model_version=7", row_count=len(frame)
        ),
        config=config,
        numeric_features=["value"],
        categorical_features=["category"],
        reference_auc=0.87,
        current_labels=[0, 1] * (len(frame) // 2),
        current_scores=[0.1, 0.9] * (len(frame) // 2),
        now=datetime(2026, 10, 24, 12, 0, tzinfo=UTC),
        report_id="fixed-report-id",
    )


def test_drift_report_round_trips_through_the_contract() -> None:
    result = _result()
    report = result.report

    parsed = DriftReport.model_validate_json(report.model_dump_json())
    assert parsed == report
    assert parsed.schema_version == "1.1.0"
    assert parsed.data_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert parsed.performance is not None
    assert parsed.performance.labeled_sample_count > 0


def test_json_and_html_are_written_and_html_contains_all_three_verdicts(
    tmp_path: Path,
) -> None:
    result = _result()

    json_path, html_path = write_drift_report(result, tmp_path)

    assert json_path.exists() and html_path.exists()
    assert json_path.name == "drift-fixed-report-id.json"
    loaded = load_report(json_path)
    assert loaded.report_id == "fixed-report-id"
    html = html_path.read_text(encoding="utf-8")
    for verdict in (
        result.report.data_drift.verdict.value,
        result.report.performance_degradation.verdict.value,
        result.report.concept_drift.verdict.value,
    ):
        assert verdict in html
    assert "Data drift" in html
    assert "Concept drift" in html
    assert "<script" not in html.lower()
