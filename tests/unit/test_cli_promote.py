"""CLI tests for ``modelops promote`` usage and ``modelops db upgrade``."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from typer.testing import CliRunner

from modelops_lab.cli import _print_promotion_result, app
from modelops_lab.contracts import PromotionCheck, PromotionVerdict
from modelops_lab.db import Base
from modelops_lab.promotion import PromotionResult

ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


def test_promote_requires_exactly_one_mode(tmp_path: Path) -> None:
    base = [
        "promote",
        "--candidate-run-id",
        "abc",
        "--reason",
        "because",
        "--database-url",
        f"sqlite:///{tmp_path / 'state.db'}",
    ]

    missing = runner.invoke(app, base)
    assert missing.exit_code == 2
    assert "choose exactly one" in missing.stderr

    both = runner.invoke(app, [*base, "--approve", "--dry-run"])
    assert both.exit_code == 2
    assert "choose exactly one" in both.stderr


def test_promote_requires_a_reason() -> None:
    result = runner.invoke(app, ["promote", "--candidate-run-id", "abc", "--dry-run"])

    assert result.exit_code == 2
    assert "Traceback" not in result.stderr


def test_db_upgrade_applies_the_migration_to_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "migrated.db"

    result = runner.invoke(app, ["db", "upgrade", "--database-url", f"sqlite:///{database}"])

    assert result.exit_code == 0, result.stderr
    assert "database upgraded to head" in result.stdout
    tables = set(inspect(create_engine(f"sqlite:///{database}")).get_table_names())
    assert tables >= {
        "active_model",
        "model_transitions",
        "promotion_decisions",
        "modelops_alembic_version",
    }
    assert tables >= set(Base.metadata.tables)


def test_db_upgrade_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "migrated.db"
    command = ["db", "upgrade", "--database-url", f"sqlite:///{database}"]

    first = runner.invoke(app, command)
    second = runner.invoke(app, command)

    assert first.exit_code == 0, first.stderr
    assert second.exit_code == 0, second.stderr


def test_promote_unknown_run_fails_without_a_traceback(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "promote",
            "--candidate-run-id",
            "0" * 32,
            "--dry-run",
            "--reason",
            "look before leaping",
            "--training-config",
            str(TRAINING_CONFIG),
            "--tracking-uri",
            (tmp_path / "mlruns").as_uri(),
            "--database-url",
            f"sqlite:///{tmp_path / 'state.db'}",
        ],
    )

    assert result.exit_code == 1
    assert "cannot read candidate run" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_noop_output_is_explicit_about_no_writes(capsys: pytest.CaptureFixture[str]) -> None:
    result = PromotionResult(
        checks=[PromotionCheck(name="run_finished", passed=True, detail="FINISHED")],
        candidate_version="2",
        active_before="2",
        active_after="2",
        verdict=PromotionVerdict.APPROVED,
        dry_run=False,
        no_op=True,
    )

    _print_promotion_result(result, "run-1")

    output = capsys.readouterr().out
    assert "active: 2 (unchanged, no-op)" in output
    assert "candidate version 2 is already active" in output
    assert "no decision or transition written" in output
