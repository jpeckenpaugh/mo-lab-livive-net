"""End-to-end monitoring proof against the Compose API.

Skipped unless ``MODELOPS_TEST_API_URL`` points at a running api container.

    MODELOPS_TEST_API_URL=http://localhost:8000 \
    MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
      uv run pytest -m e2e
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from sqlalchemy import delete, func, select
from typer.testing import CliRunner

from modelops_lab.cli import app as cli_app
from modelops_lab.contracts import DriftReport, DriftVerdict
from modelops_lab.db import PredictionEventRow, PredictionLabelRow, create_engine_for

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("MODELOPS_TEST_API_URL"),
        reason="MODELOPS_TEST_API_URL is not set",
    ),
]

API_URL = os.environ.get("MODELOPS_TEST_API_URL", "http://localhost:8000").rstrip("/")
DATABASE_URL = os.environ.get(
    "MODELOPS_TEST_DATABASE_URL",
    "postgresql+psycopg://modelops:modelops@localhost:5432/modelops",
)
TRACKING_URI = os.environ.get("MODELOPS_TEST_TRACKING_URI", "http://localhost:5001")
ROOT = Path(__file__).resolve().parents[2]
COUNT = 120


def _active_model() -> dict:
    response = httpx.get(f"{API_URL}/model", timeout=10)
    assert response.status_code == 200, response.text
    return response.json()


def test_simulated_traffic_produces_a_clean_data_drift_report(tmp_path: Path) -> None:
    model = _active_model()
    version = model["model_version"]
    dataset_dir = ROOT / "artifacts" / "datasets" / model["dataset_version"]
    if not (dataset_dir / "manifest.json").exists():
        pytest.skip(f"dataset artifact {dataset_dir} is not present locally")

    engine = create_engine_for(DATABASE_URL)
    with engine.connect() as connection:
        before = int(
            connection.execute(
                select(func.coalesce(func.max(PredictionEventRow.event_id), 0))
            ).scalar_one()
        )

    runner = CliRunner()
    try:
        simulation = runner.invoke(
            cli_app,
            [
                "simulate-traffic",
                "--dataset",
                str(dataset_dir),
                "--split",
                "validation",
                "--count",
                str(COUNT),
                "--api-url",
                API_URL,
                "--with-labels",
                "--seed",
                "7",
            ],
        )
        assert simulation.exit_code == 0, simulation.stderr
        assert f"sent: {COUNT}" in simulation.stdout
        assert f"labeled: {COUNT}" in simulation.stdout

        output = tmp_path / "reports"
        monitored = runner.invoke(
            cli_app,
            [
                "monitor",
                "--config",
                str(ROOT / "configs" / "monitoring.yaml"),
                "--reference-dataset",
                str(dataset_dir),
                "--window-size",
                str(COUNT),
                "--output-dir",
                str(output),
                "--database-url",
                DATABASE_URL,
                "--tracking-uri",
                TRACKING_URI,
            ],
        )
        assert monitored.exit_code == 0, monitored.stderr
        json_files = sorted(output.glob("drift-*.json"))
        assert len(json_files) == 1
        assert len(sorted(output.glob("drift-*.html"))) == 1
        report = DriftReport.model_validate(json.loads(json_files[0].read_text(encoding="utf-8")))
        assert report.model_version == version
        assert report.current_window.row_count == COUNT
        assert report.data_drift.verdict is DriftVerdict.NOT_DRIFTED
        assert report.performance is not None
        assert report.performance.labeled_sample_count == COUNT
    finally:
        with engine.begin() as connection:
            new_ids = select(PredictionEventRow.event_id).where(
                PredictionEventRow.event_id > before,
                PredictionEventRow.model_version == version,
            )
            connection.execute(
                delete(PredictionLabelRow).where(PredictionLabelRow.event_id.in_(new_ids))
            )
            connection.execute(
                delete(PredictionEventRow).where(
                    PredictionEventRow.event_id > before,
                    PredictionEventRow.model_version == version,
                )
            )
