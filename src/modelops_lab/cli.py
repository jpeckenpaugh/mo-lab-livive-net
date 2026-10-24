"""Typer command-line entry point for ModelOps Lab.

W0 defines the command surface only. Every lifecycle command exits with status
2 and names the wave that will implement it, so the skeleton cannot be mistaken
for working functionality.
"""

from __future__ import annotations

from typing import NoReturn

import typer

app = typer.Typer(
    name="modelops",
    help="ModelOps Lab lifecycle commands. Lifecycle behavior is not implemented yet.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _not_implemented(command: str, wave: str, scope: str) -> NoReturn:
    typer.echo(
        f"modelops {command}: not implemented until {wave} ({scope}).",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def data() -> None:
    """Generate or load and validate the dataset (arrives in W1)."""
    _not_implemented("data", "W1", "dataset generation, fingerprinting, and validation")


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
