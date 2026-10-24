"""Closed retraining loop: evidence, candidate training, gate evaluation, audit.

Retraining is an explicit operator action. It can be justified by a drift report,
but a passing gate is never permission to promote (ADR-002): the candidate is
registered and evaluated in the same evaluation-only mode as ``promote
--dry-run``, then the operator receives the exact promotion command to run by
hand. Nothing here writes a decision row, moves the active pointer, or touches
the registry alias.
"""

from __future__ import annotations

import json
import shlex
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from modelops_lab.config import PromotionConfig, TrainingConfig, load_runtime_settings
from modelops_lab.contracts import DriftReport, DriftVerdict, PromotionCheck, PromotionVerdict
from modelops_lab.data import REGIMES, DataError, LoadedDataset, load_prepared_dataset
from modelops_lab.pointer import PointerStore
from modelops_lab.promotion import (
    CandidateRun,
    MetricEvaluator,
    MlflowCandidateInspector,
    ProbabilityPredictor,
    PromotionError,
    PromotionRequest,
    perform_promotion,
)
from modelops_lab.registry import ModelRegistry
from modelops_lab.tracking import TrackingError, TrainingOutcome, log_training_run
from modelops_lab.train import TrainingError, train_on_dataset

DEFAULT_RETRAIN_DIR = Path("artifacts/retraining")
RetrainTrigger = Literal["manual", "drift"]
RetrainVerdict = Literal["eligible", "rejected"]


class RetrainError(RuntimeError):
    """Raised when retraining cannot be justified, trained, or evaluated."""


@dataclass(frozen=True)
class RetrainRequest:
    """One operator-requested retraining run."""

    dataset_dir: Path
    trigger: RetrainTrigger = "manual"
    drift_report_path: Path | None = None
    actor: str = "unknown"
    reason: str = ""
    run_name: str | None = None


@dataclass(frozen=True)
class RetrainResult:
    """Everything the CLI and audit record need to describe the outcome."""

    retrain_id: str
    created_at: datetime
    trigger: RetrainTrigger
    drift_report_id: str | None
    drift_verdicts: dict[str, str] | None
    dataset_version: str
    dataset_fingerprint: str
    dataset_regime: str
    dataset_source_kind: str
    dataset_seed: int
    run_id: str
    candidate_version: str
    active_before: str | None
    checks: list[PromotionCheck]
    verdict: RetrainVerdict
    actor: str
    reason: str
    next_command: str
    audit_path: Path
    warnings: list[str]


def load_drift_report(path: Path) -> DriftReport:
    """Read and validate a drift report contract from disk."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RetrainError(f"cannot read drift report {path}: {exc.strerror or exc}") from exc
    try:
        return DriftReport.model_validate_json(text)
    except ValidationError as exc:
        raise RetrainError(
            f"{path} is not a valid DriftReport: {exc.error_count()} validation error(s); "
            "point --drift-report at a JSON written by `modelops monitor`"
        ) from exc
    except ValueError as exc:
        raise RetrainError(f"{path} is not valid JSON: {exc}") from exc


def drift_verdicts(report: DriftReport) -> dict[str, str]:
    """Return the three independent verdicts as plain strings."""
    return {
        "data_drift": report.data_drift.verdict.value,
        "concept_drift": report.concept_drift.verdict.value,
        "performance_degradation": report.performance_degradation.verdict.value,
    }


def require_drift_evidence(report: DriftReport, path: Path) -> dict[str, str]:
    """Refuse a drift-triggered run that the report does not actually justify."""
    verdicts = drift_verdicts(report)
    if DriftVerdict.DRIFTED.value not in verdicts.values():
        raise RetrainError(
            f"drift report {report.report_id} ({path}) shows no drifted signal "
            f"(data_drift={verdicts['data_drift']}, "
            f"concept_drift={verdicts['concept_drift']}, "
            f"performance_degradation={verdicts['performance_degradation']}). "
            "A drift-triggered retrain requires at least one `drifted` verdict; "
            "use --trigger manual for routine retraining."
        )
    return verdicts


def effective_training_config(config: TrainingConfig, loaded: LoadedDataset) -> TrainingConfig:
    """Bind training settings to the verified dataset artifact, not stale YAML.

    Regime, seed, source kind, row count, and split seed come from the manifest
    so the MLflow run describes the artifact that was actually trained. Model
    hyperparameters and the tracking destination still come from the config.
    """
    manifest = loaded.manifest
    parameters = manifest.generator_parameters
    regime = parameters.get("regime")
    if manifest.source_kind != "synthetic" or regime not in REGIMES:
        regime = "baseline"
    n_samples = parameters.get("n_samples")
    if not isinstance(n_samples, int) or n_samples < 100:
        n_samples = manifest.row_count
    dataset = config.dataset.model_copy(
        update={
            "kind": manifest.source_kind,
            "seed": manifest.seed,
            "n_samples": n_samples,
            "regime": regime,
        }
    )
    split = config.split.model_copy(update={"seed": manifest.split_seed})
    return config.model_copy(update={"dataset": dataset, "split": split})


def default_trainer(
    config: TrainingConfig,
    loaded: LoadedDataset,
    tracking_uri: str,
    run_name: str | None,
) -> TrainingOutcome:
    """Train with the existing deterministic pipeline and log one MLflow run."""
    result = train_on_dataset(config, loaded)
    return log_training_run(config, result, tracking_uri=tracking_uri, run_name=run_name)


def default_inspector(run_id: str, tracking_uri: str) -> CandidateRun:
    """Inspect a freshly logged run exactly as the promotion gate does."""
    return MlflowCandidateInspector(tracking_uri).inspect(run_id)


def _next_command(run_id: str, verdict: RetrainVerdict, reason: str) -> str:
    if verdict == "eligible":
        return (
            f"modelops promote --candidate-run-id {run_id} --approve --reason {shlex.quote(reason)}"
        )
    return (
        f"modelops promote --candidate-run-id {run_id} --dry-run "
        f"--reason {shlex.quote('inspect the rejected retrained candidate')}"
    )


def _write_audit(payload: dict[str, Any], directory: Path, retrain_id: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{retrain_id}.json"
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RetrainError(f"cannot write the retraining audit record {path}: {exc}") from exc
    return path


def perform_retraining(
    request: RetrainRequest,
    *,
    training_config: TrainingConfig,
    promotion_config: PromotionConfig,
    tracking_uri: str,
    store: PointerStore,
    registry: ModelRegistry,
    decision_dir: Path | None = None,
    retrain_dir: Path | None = None,
    dataset_loader: Callable[[Path], LoadedDataset] | None = None,
    trainer: Callable[[TrainingConfig, LoadedDataset, str, str | None], TrainingOutcome]
    | None = None,
    inspector: Callable[[str, str], CandidateRun] | None = None,
    evaluator: MetricEvaluator | None = None,
    predictor: ProbabilityPredictor | None = None,
    now: datetime | None = None,
) -> RetrainResult:
    """Train a candidate, evaluate the gate, and write an audit record."""
    settings = load_runtime_settings()
    decision_dir = decision_dir or settings.artifact_root / "decisions"
    retrain_dir = retrain_dir or settings.artifact_root / "retraining"
    actor = request.actor.strip()
    if not actor:
        raise RetrainError("a retraining actor is required")
    reason = request.reason.strip() or f"retraining triggered by {request.trigger}"

    drift_report: DriftReport | None = None
    verdicts: dict[str, str] | None = None
    if request.trigger == "drift" and request.drift_report_path is None:
        raise RetrainError(
            "--trigger drift requires --drift-report PATH pointing at a DriftReport "
            "JSON written by `modelops monitor`"
        )
    if request.drift_report_path is not None:
        drift_report = load_drift_report(request.drift_report_path)
        if request.trigger == "drift":
            verdicts = require_drift_evidence(drift_report, request.drift_report_path)
        else:
            # Manual retraining does not need evidence, but a supplied report is
            # recorded rather than silently ignored.
            verdicts = drift_verdicts(drift_report)

    loader = dataset_loader or load_prepared_dataset
    try:
        loaded = loader(request.dataset_dir)
    except (DataError, OSError, ValueError) as exc:
        raise RetrainError(f"cannot load dataset artifact {request.dataset_dir}: {exc}") from exc

    config = effective_training_config(training_config, loaded)
    train = trainer or default_trainer
    try:
        outcome = train(config, loaded, tracking_uri, request.run_name)
    except (TrainingError, TrackingError, OSError, ValueError) as exc:
        raise RetrainError(f"training failed: {exc}") from exc

    inspect = inspector or default_inspector
    try:
        candidate = inspect(outcome.run_id, tracking_uri)
    except (PromotionError, OSError, ValueError) as exc:
        raise RetrainError(f"cannot inspect the trained run {outcome.run_id}: {exc}") from exc
    if candidate.model_id is None:
        raise RetrainError(
            f"run {outcome.run_id} has no logged model; the candidate cannot be registered"
        )

    try:
        candidate_version = registry.register_run(candidate.run_id, candidate.model_id)
    except Exception as exc:
        raise RetrainError(f"cannot register the retrained candidate: {exc}") from exc

    promotion_kwargs: dict[str, Any] = {}
    if evaluator is not None:
        promotion_kwargs["evaluator"] = evaluator
    if predictor is not None:
        promotion_kwargs["predictor"] = predictor
    gate = perform_promotion(
        PromotionRequest(
            candidate_run_id=candidate.run_id,
            mode="dry-run",
            reason=reason,
            actor=actor,
        ),
        promotion_config=promotion_config,
        dataset=loaded,
        candidate=candidate,
        registry=registry,
        store=store,
        decision_dir=decision_dir,
        **promotion_kwargs,
    )
    if gate.decision is not None or not gate.dry_run:
        raise RetrainError(
            "internal error: the evaluation-only gate wrote a decision; "
            "retraining must never promote automatically"
        )

    after = store.active()
    after_version = after.model_version if after is not None else None
    if after_version != gate.active_before:
        raise RetrainError(
            "internal error: the active pointer moved during evaluation-only retraining "
            f"(before {gate.active_before or '<none>'}, after {after_version or '<none>'})"
        )

    eligible = gate.verdict is PromotionVerdict.APPROVED and not any(
        not check.passed for check in gate.checks
    )
    verdict: RetrainVerdict = "eligible" if eligible else "rejected"
    warnings: list[str] = []
    if drift_report is not None and drift_report.model_version != (gate.active_before or "<none>"):
        warnings.append(
            f"drift report {drift_report.report_id} was produced for model version "
            f"{drift_report.model_version}, but the active version at retraining time is "
            f"{gate.active_before or '<none>'}; the evidence may be stale"
        )
    created_at = now or datetime.now(UTC)
    retrain_id = f"retrain-{uuid.uuid4().hex}"
    next_command = _next_command(candidate.run_id, verdict, reason)

    payload: dict[str, Any] = {
        "retrain_id": retrain_id,
        "created_at": created_at.isoformat(),
        "trigger": request.trigger,
        "drift_report_id": drift_report.report_id if drift_report is not None else None,
        "drift_report_model_version": (
            drift_report.model_version if drift_report is not None else None
        ),
        "drift_verdicts": verdicts,
        "warnings": warnings,
        "dataset_version": loaded.manifest.dataset_version,
        "dataset_fingerprint": loaded.manifest.fingerprint,
        "dataset_regime": config.dataset.regime,
        "dataset_source_kind": loaded.manifest.source_kind,
        "dataset_seed": loaded.manifest.seed,
        "run_id": outcome.run_id,
        "candidate_version": str(candidate_version),
        "active_version_before": gate.active_before,
        "checks": [check.model_dump(mode="json") for check in gate.checks],
        "verdict": verdict,
        "actor": actor,
        "reason": reason,
        "next_command": next_command,
    }
    audit_path = _write_audit(payload, retrain_dir, retrain_id)

    return RetrainResult(
        retrain_id=retrain_id,
        created_at=created_at,
        trigger=request.trigger,
        drift_report_id=payload["drift_report_id"],
        drift_verdicts=verdicts,
        dataset_version=loaded.manifest.dataset_version,
        dataset_fingerprint=loaded.manifest.fingerprint,
        dataset_regime=config.dataset.regime,
        dataset_source_kind=loaded.manifest.source_kind,
        dataset_seed=loaded.manifest.seed,
        run_id=outcome.run_id,
        candidate_version=str(candidate_version),
        active_before=gate.active_before,
        checks=gate.checks,
        verdict=verdict,
        actor=actor,
        reason=reason,
        next_command=next_command,
        audit_path=audit_path,
        warnings=warnings,
    )


__all__ = [
    "DEFAULT_RETRAIN_DIR",
    "RetrainError",
    "RetrainRequest",
    "RetrainResult",
    "default_inspector",
    "default_trainer",
    "drift_verdicts",
    "effective_training_config",
    "load_drift_report",
    "perform_retraining",
    "require_drift_evidence",
]
