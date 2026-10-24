"""End-to-end serving and rollback proof against the Compose api service.

Skipped unless ``MODELOPS_TEST_API_URL`` points at a running api container:

    MODELOPS_TEST_API_URL=http://localhost:8000 \
    MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
    MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
      uv run pytest -m e2e

The rollback target is taken from the transition history, so the test does not
assume a particular version number. It still requires a previously active
version different from the current one.
"""

from __future__ import annotations

import os

import httpx
import pytest
from typer.testing import CliRunner

from modelops_lab.cli import app as cli_app
from modelops_lab.pointer import PointerStore

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


def _features() -> dict[str, object]:
    model = httpx.get(f"{API_URL}/model", timeout=10)
    assert model.status_code == 200, model.text
    described = model.json()["features"]
    payload: dict[str, object] = {name: 50.0 for name in described["numeric_features"]}
    for name, levels in described["categorical_features"].items():
        payload[name] = levels[0]
    return payload


def test_api_health_predict_and_rollback_without_restart(tmp_path) -> None:
    health = httpx.get(f"{API_URL}/health", timeout=10)
    assert health.status_code == 200
    assert health.json()["database_reachable"] is True

    payload = _features()
    before = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
    assert before.status_code == 200, before.text
    served_version = before.json()["model_version"]

    store = PointerStore(DATABASE_URL)
    previous = {transition.to_model_version for transition in store.transitions()} - {
        served_version
    }
    if not previous:
        pytest.skip("no previously active version is available to roll back to")
    target = sorted(previous, key=int)[-1]

    runner = CliRunner()
    rollback = runner.invoke(
        cli_app,
        [
            "rollback",
            "--to-version",
            target,
            "--reason",
            "e2e rollback proof",
            "--actor",
            "e2e-test",
            "--database-url",
            DATABASE_URL,
            "--tracking-uri",
            TRACKING_URI,
            "--rollback-dir",
            str(tmp_path / "rollbacks"),
        ],
    )
    assert rollback.exit_code == 0, rollback.stderr

    after = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
    assert after.status_code == 200, after.text
    assert after.json()["model_version"] == target
    assert after.json()["model_version"] != served_version

    model = httpx.get(f"{API_URL}/model", timeout=10)
    assert model.status_code == 200
    assert model.json()["model_version"] == target
