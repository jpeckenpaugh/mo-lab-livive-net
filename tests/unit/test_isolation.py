"""Guards that keep tests away from the demo database and repository artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelops_lab.config import RuntimeSettings
from modelops_lab.db import IntegrationDatabaseGuardError, validate_test_database_url

ROOT = Path(__file__).resolve().parents[2]


def test_test_database_guard_accepts_only_a_test_database() -> None:
    accepted = "postgresql+psycopg://modelops:modelops@localhost:5432/modelops_test"
    assert validate_test_database_url(accepted) == accepted

    with pytest.raises(IntegrationDatabaseGuardError, match="must end in '_test'"):
        validate_test_database_url("postgresql+psycopg://modelops:modelops@localhost:5432/modelops")
    with pytest.raises(IntegrationDatabaseGuardError, match="must end in '_test'"):
        validate_test_database_url("sqlite:///tmp/modelops.db")
    with pytest.raises(IntegrationDatabaseGuardError, match="required"):
        validate_test_database_url(None)


def test_runtime_artifact_root_is_outside_the_repository(
    isolated_artifact_root: Path,
) -> None:
    settings = RuntimeSettings()

    assert settings.artifact_root == isolated_artifact_root
    assert ROOT not in settings.artifact_root.parents
    assert settings.artifact_root != ROOT


def test_cli_artifact_directories_derive_from_artifact_root(
    isolated_artifact_root: Path,
) -> None:
    root = RuntimeSettings().artifact_root

    for name in ("decisions", "retraining", "rollbacks", "status", "datasets"):
        assert (root / name).is_relative_to(isolated_artifact_root)
        assert not (root / name).is_relative_to(ROOT)
