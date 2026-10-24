"""Static status page: active model, decisions, and latest drift verdicts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from modelops_lab.config import load_monitoring_config
from modelops_lab.contracts import WindowSummary
from modelops_lab.db import ActiveModelRow, ModelTransitionRow, PromotionDecisionRow
from modelops_lab.monitoring import analyze_windows, write_drift_report
from modelops_lab.pointer import PointerStore
from modelops_lab.status_page import gather_status, write_status_page

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _seed(database_url: str) -> None:
    store = PointerStore(database_url)
    store.create_schema()
    with store.session() as session, session.begin():
        session.add(
            ActiveModelRow(
                id=1,
                model_name="modelops-lab-classifier",
                model_version="3",
                mlflow_run_id="run-3",
                mlflow_model_id="m-3",
                dataset_version="synthetic-v1.0.0-test",
                dataset_fingerprint="a" * 64,
                decision_id="dec-approved",
                promoted_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            ModelTransitionRow(
                operation="promotion",
                model_name="modelops-lab-classifier",
                from_model_version="2",
                to_model_version="3",
                run_id="run-3",
                decision_id="dec-approved",
                actor="release-manager",
                reason="passed the gate",
                occurred_at=NOW,
            )
        )
        session.add(
            PromotionDecisionRow(
                decision_id="dec-approved",
                candidate_run_id="run-3",
                candidate_model_version="3",
                active_model_version_before="2",
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
        )


def _write_report(report_dir: Path) -> None:
    config = load_monitoring_config("configs/monitoring.yaml")
    rng = np.random.default_rng(21)
    frame = pd.DataFrame({"value": rng.normal(0, 1, config.min_sample_size)})
    result = analyze_windows(
        model_version="3",
        reference_frame=frame,
        current_frame=frame,
        reference_window=WindowSummary(
            label="reference", source="dataset:test:validation_split", row_count=len(frame)
        ),
        current_window=WindowSummary(
            label="current", source="prediction_events", row_count=len(frame)
        ),
        config=config,
        numeric_features=["value"],
        categorical_features=[],
        reference_auc=0.9,
        current_labels=[0, 1] * (len(frame) // 2),
        current_scores=[0.2, 0.8] * (len(frame) // 2),
        now=NOW,
    )
    write_drift_report(result, report_dir)


def test_status_page_contains_active_version_decision_and_verdicts(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'status.db'}"
    report_dir = tmp_path / "reports"
    _seed(database_url)
    _write_report(report_dir)

    snapshot = gather_status(database_url=database_url, report_dir=report_dir, now=NOW)
    html_path, json_path = write_status_page(snapshot, tmp_path / "status" / "status.html")

    html = html_path.read_text(encoding="utf-8")
    assert "modelops-lab-classifier v3" in html
    assert "dec-approved" in html
    assert "not_drifted" in html
    assert "Data drift" in html
    assert "Latest drift verdicts" in html
    assert "<script" not in html.lower()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["active"]["model_version"] == "3"
    assert payload["approval"]["decision_id"] == "dec-approved"
    assert payload["reports"][0]["model_version"] == "3"
    assert payload["reports"][0]["data_drift"] == "not_drifted"
    assert payload["event_counts"]["total"] == 0
