"""Typer command-line entry point for ModelOps Lab.

W1 implements ``modelops data prepare`` and ``modelops data validate``. The
remaining lifecycle commands stay stubs that exit with status 2 and name the
wave that will implement them. User-facing data errors are reported as one
actionable stderr line with exit status 1, never a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from modelops_lab.config import ConfigError, load_training_config
from modelops_lab.contracts import DatasetManifest
from modelops_lab.data import (
    DEFAULT_DATASET_ROOT,
    REGIMES,
    DataError,
    prepare_dataset,
    verify_dataset,
)
from modelops_lab.validation import DatasetValidationError

app = typer.Typer(
    name="modelops",
    help=(
        "ModelOps Lab lifecycle commands. `data prepare` and `data validate` are "
        "implemented; later waves remain stubs."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

data_app = typer.Typer(
    name="data",
    help="Prepare and validate deterministic, versioned dataset artifacts.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")


def _not_implemented(command: str, wave: str, scope: str) -> NoReturn:
    typer.echo(
        f"modelops {command}: not implemented until {wave} ({scope}).",
        err=True,
    )
    raise typer.Exit(code=2)


def _fail(command: str, message: str) -> NoReturn:
    typer.echo(f"modelops {command}: error: {message}", err=True)
    raise typer.Exit(code=1)


def _summary_lines(dataset_version: str, fingerprint: str, manifest: DatasetManifest) -> list[str]:
    counts = ", ".join(f"{split.name} {split.row_count}" for split in manifest.splits)
    return [
        f"  dataset_version: {dataset_version}",
        f"  fingerprint: {fingerprint}",
        f"  rows: {manifest.row_count} ({counts})",
    ]


@data_app.command("prepare")
def data_prepare(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Training YAML supplying dataset and split settings.",
            show_default=True,
        ),
    ] = Path("configs/training.yaml"),
    regime: Annotated[
        str | None,
        typer.Option(
            "--regime",
            help="Override dataset.regime: baseline | covariate_shift | improved_signal.",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Root directory that will contain <dataset_version>/.",
            show_default=True,
        ),
    ] = DEFAULT_DATASET_ROOT,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Rewrite the version directory if it already exists.",
        ),
    ] = False,
) -> None:
    """Generate or load the dataset, validate it, and write a versioned artifact."""
    try:
        training = load_training_config(config)
        if regime is not None:
            normalized = regime.strip().lower()
            if normalized not in REGIMES:
                raise ConfigError(f"--regime must be one of {', '.join(REGIMES)}; got {regime!r}")
            training = training.model_copy(
                update={"dataset": training.dataset.model_copy(update={"regime": normalized})}
            )
        result = prepare_dataset(training.dataset, training.split, out_root=out, force=force)
    except (ConfigError, DataError, DatasetValidationError, OSError) as exc:
        _fail("data prepare", str(exc))

    manifest = result.manifest
    if result.already_prepared:
        typer.echo(
            f"already prepared and verified: {manifest.dataset_version} at {result.directory}"
        )
        typer.echo("  dataset.csv, manifest.json, and splits.json match the regenerated data")
        typer.echo("  re-run with --force to rewrite the artifact")
    else:
        typer.echo(f"prepared: {manifest.dataset_version} at {result.directory}")
    for line in _summary_lines(manifest.dataset_version, manifest.fingerprint, manifest):
        typer.echo(line)
    typer.echo("  source: " + manifest.source_kind)


@data_app.command("validate")
def data_validate(
    dataset_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing dataset.csv, manifest.json, and splits.json.",
        ),
    ],
) -> None:
    """Re-verify an existing artifact: fingerprints, canonical bytes, and splits."""
    try:
        verification = verify_dataset(dataset_dir)
    except (DataError, DatasetValidationError, OSError) as exc:
        _fail("data validate", str(exc))

    manifest = verification.manifest
    typer.echo(f"dataset: {manifest.dataset_version}")
    typer.echo(f"  path: {verification.directory.resolve()}")
    typer.echo(f"  source: {manifest.source_kind}")
    typer.echo(f"  schema: {manifest.data_schema_version}")
    typer.echo(f"  seed: {manifest.seed}, split_seed: {manifest.split_seed}")
    typer.echo(f"  rows: {manifest.row_count}")
    typer.echo(f"  fingerprint: {manifest.fingerprint}")
    typer.echo("  splits:")
    for split in manifest.splits:
        typer.echo(
            f"    {split.name:<10} {split.row_count:>5} rows  "
            f"fingerprint {split.fingerprint[:12]}  "
            f"class_counts {split.class_counts}"
        )
    typer.echo("verified: manifest, dataset fingerprint, canonical bytes, and split manifests")


@app.command()
def train() -> None:
    """Train a candidate and record an MLflow run (arrives in W2)."""
    _not_implemented("train", "W2", "training pipeline and MLflow tracking")


@app.command()
def evaluate() -> None:
    """Evaluate a candidate on held-out data (arrives in W2)."""
    _not_implemented("evaluate", "W2", "held-out evaluation and metric reporting")


@app.command()
def promote() -> None:
    """Apply the promotion gate and record a decision (arrives in W3)."""
    _not_implemented("promote", "W3", "registry writes and the promotion gate")


@app.command()
def serve() -> None:
    """Run the inference API (arrives in W4)."""
    _not_implemented("serve", "W4", "FastAPI serving and prediction events")


@app.command()
def monitor() -> None:
    """Compare recent traffic with the reference window (arrives in W5)."""
    _not_implemented("monitor", "W5", "drift statistics and report generation")


@app.command()
def retrain() -> None:
    """Run the closed retraining loop (arrives in W6)."""
    _not_implemented("retrain", "W6", "the retraining workflow")


@app.command()
def rollback() -> None:
    """Restore a previously approved model version (arrives in W4)."""
    _not_implemented("rollback", "W4", "serving and rollback of the active pointer")


@app.command()
def status() -> None:
    """Generate the static lifecycle status page (arrives in W5)."""
    _not_implemented("status", "W5", "the generated status page")


if __name__ == "__main__":
    app()
