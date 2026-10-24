"""Typer command-line entry point for ModelOps Lab.

W1 implements ``modelops data prepare`` and ``modelops data validate``; W2 adds
``modelops train`` and ``modelops evaluate``. The remaining lifecycle commands
stay stubs that exit with status 2 and name the wave that will implement them.
User-facing configuration, data, training, and tracking errors are reported as
one actionable stderr line with exit status 1, never a traceback.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from modelops_lab.config import (
    ConfigError,
    load_promotion_config,
    load_runtime_settings,
    load_training_config,
)
from modelops_lab.contracts import DatasetManifest, PromotionVerdict
from modelops_lab.data import (
    DEFAULT_DATASET_ROOT,
    REGIMES,
    DataError,
    load_prepared_dataset,
    prepare_dataset,
    verify_dataset,
)
from modelops_lab.db import DatabaseError, upgrade_database
from modelops_lab.pointer import PointerError, PointerStore, PromotionConflict
from modelops_lab.promotion import (
    DEFAULT_DECISION_DIR,
    MlflowCandidateInspector,
    PromotionError,
    PromotionRequest,
    PromotionResult,
    perform_promotion,
    registry_for,
    tag_candidate_run,
)
from modelops_lab.registry import RegistryError
from modelops_lab.tracking import (
    TrackingError,
    load_run_model,
    log_training_run,
    redact_uri,
    resolve_tracking_uri,
)
from modelops_lab.train import (
    EVALUATION_SPLITS,
    TrainingError,
    evaluate_loaded_model,
    evaluation_payload,
    train_on_dataset,
)
from modelops_lab.validation import DatasetValidationError

app = typer.Typer(
    name="modelops",
    help=(
        "ModelOps Lab lifecycle commands. `data`, `train`, `evaluate`, `promote`, "
        "and `db` are implemented; later waves remain stubs."
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

db_app = typer.Typer(
    name="db",
    help="Apply and inspect database schema migrations.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


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
def train(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            help=(
                "Verified dataset artifact directory containing dataset.csv, "
                "manifest.json, and splits.json."
            ),
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Training YAML supplying model, split, and tracking settings.",
            show_default=True,
        ),
    ] = Path("configs/training.yaml"),
    run_name: Annotated[
        str | None,
        typer.Option("--run-name", help="Optional human-readable MLflow run name."),
    ] = None,
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            "--tracking-uri",
            help="Override tracking.uri and MLFLOW_TRACKING_URI for this invocation.",
        ),
    ] = None,
) -> None:
    """Fit on the train split and record one complete MLflow run."""
    try:
        training = load_training_config(config)
        loaded = load_prepared_dataset(dataset)
        result = train_on_dataset(training, loaded)
        uri = resolve_tracking_uri(training.tracking.uri, tracking_uri)
        outcome = log_training_run(
            training,
            result,
            tracking_uri=uri,
            run_name=run_name.strip() if run_name and run_name.strip() else None,
        )
    except (
        ConfigError,
        DataError,
        DatasetValidationError,
        TrainingError,
        TrackingError,
        OSError,
        ValueError,
    ) as exc:
        _fail("train", str(exc))

    typer.echo(f"run_id: {outcome.run_id}")
    typer.echo(f"experiment: {outcome.experiment_name} ({outcome.experiment_id})")
    typer.echo(f"dataset_version: {loaded.manifest.dataset_version}")
    typer.echo(f"dataset_fingerprint: {loaded.manifest.fingerprint}")
    typer.echo("validation:")
    for name in ("roc_auc", "precision", "recall", "f1", "accuracy", "log_loss"):
        typer.echo(f"  {name}: {getattr(outcome.metrics, name):.6f}")
    matrix = outcome.metrics.confusion_dict()
    typer.echo(
        "  confusion_matrix: "
        f"tn={matrix['true_negative']} fp={matrix['false_positive']} "
        f"fn={matrix['false_negative']} tp={matrix['true_positive']}"
    )
    typer.echo(f"  sample_count: {outcome.metrics.sample_count}")
    typer.echo(f"model_uri: {outcome.model_uri}")


@app.command()
def evaluate(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            help="Verified dataset artifact directory to evaluate on.",
        ),
    ],
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="MLflow run id whose logged model should be evaluated."),
    ],
    split: Annotated[
        Literal["validation", "test"],
        typer.Option(
            "--split",
            help="Evaluation partition; validation is the default. Test is held out.",
        ),
    ] = "validation",
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            "--tracking-uri",
            help="Override tracking.uri and MLFLOW_TRACKING_URI for this invocation.",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Training YAML supplying the default tracking destination.",
            show_default=True,
        ),
    ] = Path("configs/training.yaml"),
) -> None:
    """Evaluate a logged model on one verified split. Logs nothing."""
    if split not in EVALUATION_SPLITS:  # defensive; typer constrains the option
        _fail("evaluate", f"--split must be one of {', '.join(EVALUATION_SPLITS)}")
    try:
        training = load_training_config(config)
        uri = resolve_tracking_uri(training.tracking.uri, tracking_uri)
        loaded = load_prepared_dataset(dataset)
        run_model = load_run_model(run_id, uri)
        metrics = evaluate_loaded_model(run_model.model, loaded, split_name=split)
    except (
        ConfigError,
        DataError,
        DatasetValidationError,
        TrainingError,
        TrackingError,
        OSError,
        ValueError,
    ) as exc:
        _fail("evaluate", str(exc))

    summary = evaluation_payload(loaded, split, metrics)
    summary["model_uri"] = run_model.model_uri
    summary["run_id"] = run_id

    if split == "test":
        typer.echo(
            "WARNING: evaluating on the held-out test split. Do not use these results "
            "for tuning, model selection, or promotion decisions.",
            err=True,
        )
    trained_fingerprint = run_model.trained_dataset_fingerprint
    if trained_fingerprint is None:
        typer.echo(
            "WARNING: this run records no dataset fingerprint tag; "
            "the evaluation slice cannot be checked against its training data.",
            err=True,
        )
    elif trained_fingerprint != loaded.manifest.fingerprint:
        typer.echo(
            "WARNING: the model was trained on dataset fingerprint "
            f"{trained_fingerprint}, but this evaluation uses {loaded.manifest.fingerprint}. "
            "Comparisons are only valid on a shared slice with recorded fingerprints.",
            err=True,
        )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@db_app.command("upgrade")
def db_upgrade(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Override MODELOPS_DATABASE_URL for this invocation.",
        ),
    ] = None,
) -> None:
    """Apply every pending Alembic migration."""
    url = (
        database_url.strip()
        if database_url and database_url.strip()
        else load_runtime_settings().database_url
    )
    try:
        upgrade_database(url)
    except DatabaseError as exc:
        _fail("db upgrade", str(exc))
    typer.echo(f"database upgraded to head: {redact_uri(url)}")


@app.command()
def promote(
    candidate_run_id: Annotated[
        str,
        typer.Option(
            "--candidate-run-id",
            help="MLflow run id of the candidate to evaluate.",
        ),
    ],
    reason: Annotated[
        str,
        typer.Option(
            "--reason",
            help="Why this decision is being made; recorded verbatim.",
        ),
    ],
    approve: Annotated[
        bool,
        typer.Option("--approve", help="Ask the gate to approve the candidate."),
    ] = False,
    reject: Annotated[
        bool,
        typer.Option("--reject", help="Reject the candidate even if every check passes."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Evaluate and print checks without writing a decision, pointer, or alias.",
        ),
    ] = False,
    actor: Annotated[
        str | None,
        typer.Option("--actor", help="Human or service recording the decision."),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            help="Dataset artifact to evaluate on; defaults to the run's dataset_version.",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Promotion policy YAML.",
            show_default=True,
        ),
    ] = Path("configs/promotion.yaml"),
    training_config: Annotated[
        Path,
        typer.Option(
            "--training-config",
            help="Training YAML supplying the default tracking destination.",
            show_default=True,
        ),
    ] = Path("configs/training.yaml"),
    tracking_uri: Annotated[
        str | None,
        typer.Option(
            "--tracking-uri",
            help="Override tracking.uri and MLFLOW_TRACKING_URI for this invocation.",
        ),
    ] = None,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Override MODELOPS_DATABASE_URL for this invocation.",
        ),
    ] = None,
) -> None:
    """Evaluate the promotion gate and record an auditable decision."""
    modes = [
        name
        for name, chosen in (
            ("approve", approve),
            ("reject", reject),
            ("dry-run", dry_run),
        )
        if chosen
    ]
    if len(modes) != 1:
        typer.echo(
            "modelops promote: error: choose exactly one of --approve, --reject, --dry-run",
            err=True,
        )
        raise typer.Exit(code=2)
    if not reason.strip():
        typer.echo("modelops promote: error: --reason must not be empty", err=True)
        raise typer.Exit(code=2)

    mode: Literal["approve", "reject", "dry-run"] = modes[0]  # type: ignore[assignment]
    chosen_actor = (actor or "").strip() or _default_actor()
    url = (
        database_url.strip()
        if database_url and database_url.strip()
        else load_runtime_settings().database_url
    )

    try:
        promotion = load_promotion_config(config)
        training = load_training_config(training_config)
        uri = resolve_tracking_uri(training.tracking.uri, tracking_uri)
        candidate = MlflowCandidateInspector(uri).inspect(candidate_run_id)

        dataset_dir = dataset
        if dataset_dir is None:
            version = candidate.tag("dataset_version")
            if version is None:
                raise PromotionError(
                    f"run {candidate_run_id} has no dataset_version tag; pass --dataset explicitly"
                )
            dataset_dir = DEFAULT_DATASET_ROOT / version
        loaded = load_prepared_dataset(dataset_dir)

        store = PointerStore(url)
        registry = registry_for(tracking_uri=uri, promotion_config=promotion)
        result = perform_promotion(
            PromotionRequest(
                candidate_run_id=candidate_run_id,
                mode=mode,
                reason=reason,
                actor=chosen_actor,
            ),
            promotion_config=promotion,
            dataset=loaded,
            candidate=candidate,
            registry=registry,
            store=store,
            decision_dir=DEFAULT_DECISION_DIR,
        )
    except PromotionConflict as exc:
        _fail("promote", str(exc))
    except (
        ConfigError,
        DataError,
        DatasetValidationError,
        PromotionError,
        PointerError,
        RegistryError,
        TrackingError,
        SQLAlchemyError,
        OSError,
        ValueError,
    ) as exc:
        _fail("promote", str(exc))

    _print_promotion_result(result, candidate_run_id)
    if result.decision is not None:
        try:
            tag_candidate_run(uri, candidate_run_id, result.decision)
        except (MlflowException, OSError) as exc:
            typer.echo(
                "modelops promote: warning: the decision was recorded, but the candidate "
                f"run tags could not be written: {exc}",
                err=True,
            )
    if result.verdict is PromotionVerdict.REJECTED and not result.dry_run:
        raise typer.Exit(code=1)


def _default_actor() -> str:
    try:
        return getpass.getuser() or "unknown"
    except (KeyError, OSError):
        return "unknown"


def _print_promotion_result(result: PromotionResult, candidate_run_id: str) -> None:
    if result.decision is None:
        decision_id = "none (no-op)" if result.no_op else "none (dry run)"
    else:
        decision_id = result.decision.decision_id
    typer.echo(f"decision_id: {decision_id}")
    typer.echo(f"candidate_run_id: {candidate_run_id}")
    typer.echo(f"candidate_version: {result.candidate_version or '<unregistered>'}")
    before = result.active_before or "<none>"
    if result.dry_run:
        active = f"{before} (unchanged, dry run)"
    elif result.no_op:
        active = f"{before} (unchanged, no-op)"
    elif result.active_after != result.active_before:
        active = f"{before} -> {result.active_after}"
    else:
        active = f"{before} (unchanged)"
    typer.echo(f"active: {active}")
    if result.dry_run:
        verdict = "would approve" if result.verdict is PromotionVerdict.APPROVED else "would reject"
    else:
        verdict = result.verdict.value
    typer.echo(f"verdict: {verdict}")
    typer.echo("checks:")
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines = check.detail.splitlines() or [""]
        typer.echo(f"  {check.name}: {status}  {lines[0]}")
        for line in lines[1:]:
            typer.echo(f"    {line}")
    if result.decision_path is not None:
        typer.echo(f"decision_record: {result.decision_path}")
    if result.no_op:
        typer.echo(
            f"no-op: candidate version {result.candidate_version} is already active; "
            "no decision or transition written"
        )


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
