"""CLI integration for ``modelops data prepare`` and ``modelops data validate``."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from modelops_lab.cli import app
from modelops_lab.contracts import DatasetManifest

ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"
runner = CliRunner()


def version_directories(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def test_prepare_writes_three_files_and_parses(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--config",
            str(TRAINING_CONFIG),
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    (directory,) = version_directories(tmp_path)
    for name in ("dataset.csv", "manifest.json", "splits.json"):
        assert (directory / name).is_file()
    manifest = DatasetManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.dataset_version == directory.name
    assert manifest.row_count == 2000
    assert "prepared:" in result.stdout


def test_prepare_is_idempotent(tmp_path: Path) -> None:
    arguments = [
        "data",
        "prepare",
        "--config",
        str(TRAINING_CONFIG),
        "--out",
        str(tmp_path),
    ]
    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "already prepared and verified" in second.stdout
    assert len(version_directories(tmp_path)) == 1


def test_prepare_regime_override_changes_the_version(tmp_path: Path) -> None:
    baseline = runner.invoke(
        app,
        ["data", "prepare", "--config", str(TRAINING_CONFIG), "--out", str(tmp_path)],
    )
    shifted = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--config",
            str(TRAINING_CONFIG),
            "--out",
            str(tmp_path),
            "--regime",
            "covariate_shift",
        ],
    )

    assert baseline.exit_code == 0, baseline.stderr
    assert shifted.exit_code == 0, shifted.stderr
    assert len(version_directories(tmp_path)) == 2


def test_validate_good_artifact(tmp_path: Path) -> None:
    runner.invoke(
        app,
        ["data", "prepare", "--config", str(TRAINING_CONFIG), "--out", str(tmp_path)],
    )
    (directory,) = version_directories(tmp_path)

    result = runner.invoke(app, ["data", "validate", str(directory)])

    assert result.exit_code == 0, result.stderr
    assert "verified:" in result.stdout
    assert "fingerprint" in result.stdout


def test_validate_corrupted_artifact_fails_cleanly(tmp_path: Path) -> None:
    runner.invoke(
        app,
        ["data", "prepare", "--config", str(TRAINING_CONFIG), "--out", str(tmp_path)],
    )
    (directory,) = version_directories(tmp_path)
    dataset = directory / "dataset.csv"
    dataset.write_bytes(dataset.read_bytes().replace(b"row_id,age", b"row_id,age,", 1))

    result = runner.invoke(app, ["data", "validate", str(directory)])

    assert result.exit_code == 1
    assert "fingerprint mismatch" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


def test_prepare_refuses_a_mismatched_existing_artifact_and_force_rewrites(
    tmp_path: Path,
) -> None:
    arguments = [
        "data",
        "prepare",
        "--config",
        str(TRAINING_CONFIG),
        "--out",
        str(tmp_path),
    ]
    runner.invoke(app, arguments)
    (directory,) = version_directories(tmp_path)
    dataset = directory / "dataset.csv"
    dataset.write_bytes(dataset.read_bytes() + b"garbage\n")

    refused = runner.invoke(app, arguments)
    forced = runner.invoke(app, [*arguments, "--force"])
    validated = runner.invoke(app, ["data", "validate", str(directory)])

    assert refused.exit_code == 1
    assert "does not match a regenerated artifact" in refused.stderr
    assert "Traceback" not in refused.stderr
    assert forced.exit_code == 0, forced.stderr
    assert validated.exit_code == 0, validated.stderr


def test_validate_missing_directory_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "validate", str(tmp_path / "absent")])

    assert result.exit_code == 1
    assert "does not exist" in result.stderr
    assert "Traceback" not in result.stderr


def test_validate_rejects_a_parent_directory_with_actionable_message(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["data", "validate", str(tmp_path)])

    assert result.exit_code == 1
    assert "manifest.json is missing" in result.stderr


def test_invalid_regime_is_a_clean_data_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--config",
            str(TRAINING_CONFIG),
            "--out",
            str(tmp_path),
            "--regime",
            "sideways",
        ],
    )

    assert result.exit_code == 1
    assert "--regime must be one of" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_config_is_a_clean_data_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "prepare",
            "--config",
            str(tmp_path / "absent.yaml"),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "cannot read configuration file" in result.stderr
    assert "Traceback" not in result.stderr
