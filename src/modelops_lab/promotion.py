"""Promotion gate, decision records, and the approval orchestration.

The gate is a pure function of its inputs: an already-inspected candidate run, a
verified dataset, the current pointer, and injected registry/evaluator/loading
seams. That keeps every check unit-testable without Docker, and keeps the real
MLflow and Postgres implementations in :mod:`modelops_lab.registry` and
:mod:`modelops_lab.pointer`.

Approval never moves the pointer directly. It calls
:meth:`modelops_lab.pointer.PointerStore.apply_promotion`, which re-reads the
pointer under an advisory lock and refuses a stale expected version.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mlflow.sklearn
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from modelops_lab.config import PromotionConfig
from modelops_lab.contracts import (
    ApprovalMetadata,
    ConfusionMatrix,
    EvaluationMetrics,
    ModelDescriptor,
    PromotionCheck,
    PromotionDecision,
    PromotionVerdict,
)
from modelops_lab.data import LoadedDataset, split_frame
from modelops_lab.pointer import (
    ActiveModel,
    DecisionRecord,
    PointerStore,
    PromotionConflict,
)
from modelops_lab.registry import MlflowModelRegistry, ModelRegistry, RegistryError
from modelops_lab.tracking import redact_uri
from modelops_lab.train import (
    Metrics,
    evaluate_loaded_model,
    predict_probabilities,
)

DEFAULT_DECISION_DIR = Path("artifacts/decisions")
QUALITY_METRICS = ("roc_auc", "precision", "recall", "f1")

CHECK_RUN_FINISHED = "run_finished"
CHECK_MODEL_LOGGED = "model_logged"
CHECK_EVALUATION_ARTIFACT = "evaluation_artifact"
CHECK_DATASET_IDENTITY = "dataset_identity"
CHECK_REQUIRED_ARTIFACTS = "required_artifacts"
CHECK_QUALITY_FLOOR = "quality_floor"
CHECK_REGRESSION = "regression_vs_active"
CHECK_SMOKE_TEST = "smoke_test"
CHECK_REGISTRY_CONSISTENCY = "registry_consistency"

PromotionMode = Literal["approve", "reject", "dry-run"]

MetricEvaluator = Callable[[Any, LoadedDataset, str], Metrics]
ProbabilityPredictor = Callable[[Any, Any, list[str]], np.ndarray]


class PromotionError(RuntimeError):
    """Raised when promotion cannot be evaluated or recorded."""


@dataclass(frozen=True)
class CandidateRun:
    """Everything the gate needs to know about one candidate run."""

    run_id: str
    status: str
    tags: Mapping[str, str]
    model_id: str | None
    model_uri: str | None
    model: Any | None
    model_error: str | None
    evaluation: dict[str, Any] | None
    evaluation_error: str | None

    def tag(self, name: str) -> str | None:
        value = self.tags.get(name)
        return value.strip() if isinstance(value, str) and value.strip() else None


class MlflowCandidateInspector:
    """Read a candidate run, its logged model, and ``evaluation.json``."""

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri=tracking_uri)
        # ``mlflow.sklearn.load_model`` resolves ``models:/<id>`` through the
        # process-global tracking URI, not the client above.
        mlflow.set_tracking_uri(tracking_uri)

    def inspect(self, run_id: str) -> CandidateRun:
        try:
            run = self._client.get_run(run_id)
        except (MlflowException, OSError, ValueError) as exc:
            raise PromotionError(
                f"cannot read candidate run {run_id!r} from {redact_uri(self.tracking_uri)}: {exc}"
            ) from exc

        outputs = run.outputs.model_outputs if run.outputs is not None else []
        model_id = outputs[0].model_id if outputs and outputs[0].model_id else None
        model_uri = f"models:/{model_id}" if model_id else None
        model: Any | None = None
        model_error: str | None = None
        if model_uri is None:
            model_error = "the run has no logged model output"
        else:
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except (MlflowException, OSError, ValueError) as exc:
                model_error = f"cannot load logged model {model_uri}: {exc}"

        evaluation: dict[str, Any] | None = None
        evaluation_error: str | None = None
        try:
            downloaded = self._client.download_artifacts(run_id, "evaluation.json")
            evaluation = json.loads(Path(downloaded).read_text(encoding="utf-8"))
            if not isinstance(evaluation, dict):
                raise ValueError("evaluation.json is not a JSON object")
        except (MlflowException, OSError, ValueError, json.JSONDecodeError) as exc:
            evaluation = None
            evaluation_error = f"evaluation.json unavailable: {exc}"

        return CandidateRun(
            run_id=run_id,
            status=run.info.status,
            tags={key: str(value) for key, value in run.data.tags.items()},
            model_id=model_id,
            model_uri=model_uri,
            model=model,
            model_error=model_error,
            evaluation=evaluation,
            evaluation_error=evaluation_error,
        )


def _default_evaluator(model: Any, dataset: LoadedDataset, split: str) -> Metrics:
    return evaluate_loaded_model(model, dataset, split_name=split)


@dataclass(frozen=True)
class GateContext:
    """Inputs to the individual checks, kept explicit for testability."""

    config: PromotionConfig
    candidate: CandidateRun
    dataset: LoadedDataset
    pointer: ActiveModel | None
    alias_version: str | None
    alias_error: str | None
    active_model: Any | None
    active_model_error: str | None
    evaluator: MetricEvaluator = _default_evaluator
    predictor: ProbabilityPredictor = predict_probabilities
    registry: ModelRegistry | None = None
    candidate_version: str | None = None


@dataclass(frozen=True)
class GateEvaluation:
    """Result of evaluating every check, including live candidate metrics."""

    checks: list[PromotionCheck]
    candidate_metrics: Metrics | None
    active_metrics: Metrics | None
    failed: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    observed: float | None = None,
    threshold: float | None = None,
) -> PromotionCheck:
    return PromotionCheck(
        name=name,
        passed=passed,
        detail=detail.strip() or name,
        observed_value=observed,
        threshold_value=threshold,
    )


def _run_finished(candidate: CandidateRun) -> PromotionCheck:
    passed = candidate.status == "FINISHED"
    return _check(
        CHECK_RUN_FINISHED,
        passed,
        f"run {candidate.run_id} status {candidate.status}"
        + ("" if passed else " (requires FINISHED)"),
    )


def _model_logged(candidate: CandidateRun) -> PromotionCheck:
    if candidate.model_id is None:
        return _check(
            CHECK_MODEL_LOGGED,
            False,
            f"run {candidate.run_id} has no logged model output",
        )
    detail = f"logged model {candidate.model_id}"
    if candidate.model_error is not None:
        detail += f"; load failed: {candidate.model_error}"
    return _check(CHECK_MODEL_LOGGED, True, detail)


def _evaluation_artifact(candidate: CandidateRun) -> PromotionCheck:
    if candidate.evaluation is None:
        return _check(
            CHECK_EVALUATION_ARTIFACT,
            False,
            candidate.evaluation_error or "evaluation.json is missing",
        )
    split = candidate.evaluation.get("evaluation_split", "<missing>")
    metrics = candidate.evaluation.get("metrics", {})
    roc_auc = metrics.get("roc_auc") if isinstance(metrics, dict) else None
    return _check(
        CHECK_EVALUATION_ARTIFACT,
        True,
        f"evaluation.json parsed: split {split}, roc_auc {roc_auc}",
    )


def _dataset_identity(
    candidate: CandidateRun, dataset: LoadedDataset, split: str
) -> PromotionCheck:
    tag_fingerprint = candidate.tag("dataset_fingerprint")
    artifact_fingerprint = dataset.manifest.fingerprint
    problems: list[str] = []
    if tag_fingerprint is None:
        problems.append("run has no dataset_fingerprint tag")
    elif tag_fingerprint != artifact_fingerprint:
        problems.append(
            f"run dataset_fingerprint {tag_fingerprint} does not match artifact "
            f"{artifact_fingerprint}"
        )

    evaluation_split = None
    if candidate.evaluation is None:
        problems.append("no evaluation.json to confirm the evaluation split")
    else:
        evaluation_split = candidate.evaluation.get("evaluation_split")
        if evaluation_split != split:
            problems.append(f"evaluation split is {evaluation_split!r}, requires {split!r}")
        evaluation_fingerprint = candidate.evaluation.get("dataset_fingerprint")
        if evaluation_fingerprint != artifact_fingerprint:
            problems.append(
                f"evaluation.json dataset_fingerprint {evaluation_fingerprint} does not "
                f"match artifact {artifact_fingerprint}"
            )
    if problems:
        return _check(CHECK_DATASET_IDENTITY, False, "; ".join(problems))
    return _check(
        CHECK_DATASET_IDENTITY,
        True,
        f"run and evaluation.json match dataset fingerprint {artifact_fingerprint}; "
        f"evaluation split {split}",
    )


def _logical_artifact_status(name: str, candidate: CandidateRun) -> tuple[bool, str]:
    if name == "model":
        present = candidate.model_id is not None
        return present, "logged model" if present else "no logged model output"
    if name == "evaluation.json":
        present = candidate.evaluation is not None
        return present, "evaluation.json" if present else "evaluation.json is missing"
    return False, f"unknown logical artifact {name!r}"


def _required_artifacts(config: PromotionConfig, candidate: CandidateRun) -> PromotionCheck:
    missing: list[str] = []
    present: list[str] = []
    for name in config.required_artifacts:
        exists, detail = _logical_artifact_status(name, candidate)
        (present if exists else missing).append(detail if exists else name)
    if missing:
        return _check(
            CHECK_REQUIRED_ARTIFACTS,
            False,
            "missing required artifact(s): "
            + ", ".join(missing)
            + (f"; present: {', '.join(present)}" if present else ""),
        )
    return _check(
        CHECK_REQUIRED_ARTIFACTS,
        True,
        "present: " + ", ".join(config.required_artifacts),
    )


def _evaluate_candidate(
    context: GateContext,
) -> tuple[Metrics | None, str | None]:
    if context.candidate.model is None:
        return None, context.candidate.model_error or "candidate model is unavailable"
    split = context.config.evaluation_split
    try:
        metrics = context.evaluator(context.candidate.model, context.dataset, split)
    except Exception as exc:
        return None, f"cannot evaluate candidate on the {split} split: {exc}"
    return metrics, None


def _evaluate_active(context: GateContext) -> tuple[Metrics | None, str | None]:
    if context.pointer is None:
        return None, None
    if context.active_model is None:
        return (
            None,
            context.active_model_error or "active model artifact is unavailable",
        )
    split = context.config.evaluation_split
    try:
        metrics = context.evaluator(context.active_model, context.dataset, split)
    except Exception as exc:
        return (
            None,
            f"cannot evaluate active version {context.pointer.model_version} on the "
            f"candidate's {split} split: {exc}",
        )
    return metrics, None


def _quality_floor(
    config: PromotionConfig, metrics: Metrics | None, error: str | None
) -> PromotionCheck:
    if metrics is None:
        return _check(CHECK_QUALITY_FLOOR, False, error or "candidate metrics unavailable")
    lines: list[str] = []
    worst_margin: float | None = None
    worst_value = 0.0
    worst_floor = 0.0
    failed = False
    for name in QUALITY_METRICS:
        value = float(getattr(metrics, name))
        floor = float(getattr(config.quality_floor, name))
        passed = value >= floor
        failed = failed or not passed
        lines.append(
            f"{name} {value:.4f} {'>=' if passed else '<'} floor {floor:.4f}"
            f"{'' if passed else ' (below)'}"
        )
        margin = value - floor
        if worst_margin is None or margin < worst_margin:
            worst_margin = margin
            worst_value = value
            worst_floor = floor
    return _check(
        CHECK_QUALITY_FLOOR,
        not failed,
        "\n".join(lines),
        observed=worst_value,
        threshold=worst_floor,
    )


def _regression(
    config: PromotionConfig,
    context: GateContext,
    candidate_metrics: Metrics | None,
    candidate_error: str | None,
    active_metrics: Metrics | None,
    active_error: str | None,
) -> PromotionCheck:
    if context.pointer is None:
        return _check(
            CHECK_REGRESSION,
            True,
            "no active model (baseline promotion)",
        )
    if candidate_metrics is None:
        return _check(
            CHECK_REGRESSION,
            False,
            candidate_error or "candidate metrics unavailable",
        )
    if active_metrics is None:
        return _check(CHECK_REGRESSION, False, active_error or "active metrics unavailable")

    lines: list[str] = []
    failed = False
    worst_margin: float | None = None
    worst_delta = 0.0
    worst_tolerance = 0.0
    for name in QUALITY_METRICS:
        candidate_value = float(getattr(candidate_metrics, name))
        active_value = float(getattr(active_metrics, name))
        tolerance = float(getattr(config.regression_tolerance, name))
        delta = candidate_value - active_value
        passed = delta >= -tolerance
        failed = failed or not passed
        lines.append(
            f"{name}: candidate {candidate_value:.4f} vs active {active_value:.4f} "
            f"(delta {delta:+.4f}, tolerance {tolerance:.4f}) "
            f"{'PASS' if passed else 'FAIL'}"
        )
        margin = delta + tolerance
        if worst_margin is None or margin < worst_margin:
            worst_margin = margin
            worst_delta = delta
            worst_tolerance = tolerance
    return _check(
        CHECK_REGRESSION,
        not failed,
        "\n".join(lines),
        observed=worst_delta,
        threshold=-worst_tolerance,
    )


def _smoke_test(config: PromotionConfig, context: GateContext) -> PromotionCheck:
    smoke = config.smoke_test
    if not smoke.enabled:
        return _check(CHECK_SMOKE_TEST, True, "smoke test disabled by configuration")
    model = context.candidate.model
    model_error = context.candidate.model_error
    if context.candidate_version is not None and context.registry is not None:
        # Prefer the registered version: the smoke test should prove that the
        # version the alias/pointer would reference can actually be loaded.
        try:
            model = context.registry.load_version(context.candidate_version)
            model_error = None
        except RegistryError as exc:
            model = None
            model_error = (
                f"cannot load registered candidate version {context.candidate_version}: {exc}"
            )
    if model is None:
        return _check(
            CHECK_SMOKE_TEST,
            False,
            model_error or "candidate model is unavailable",
        )
    split = config.evaluation_split
    frame = split_frame(context.dataset, split).head(smoke.sample_count)
    if frame.empty:
        return _check(
            CHECK_SMOKE_TEST,
            False,
            f"the {split} split is empty; cannot run the smoke test",
        )
    features = list(context.dataset.manifest.feature_names)
    started = time.perf_counter()
    try:
        probabilities = np.asarray(context.predictor(model, frame, features), dtype=float)
    except Exception as exc:
        return _check(CHECK_SMOKE_TEST, False, f"prediction failed: {exc}")
    latency_ms = (time.perf_counter() - started) * 1000.0

    if probabilities.shape != (len(frame),):
        return _check(
            CHECK_SMOKE_TEST,
            False,
            f"expected {len(frame)} probabilities, received shape {probabilities.shape}",
            observed=latency_ms,
            threshold=smoke.max_latency_ms,
        )
    if not np.isfinite(probabilities).all():
        return _check(
            CHECK_SMOKE_TEST,
            False,
            "probabilities are not all finite",
            observed=latency_ms,
            threshold=smoke.max_latency_ms,
        )
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        return _check(
            CHECK_SMOKE_TEST,
            False,
            "probabilities fall outside [0, 1]",
            observed=latency_ms,
            threshold=smoke.max_latency_ms,
        )
    if latency_ms > smoke.max_latency_ms:
        return _check(
            CHECK_SMOKE_TEST,
            False,
            f"latency {latency_ms:.1f} ms exceeds {smoke.max_latency_ms:.1f} ms",
            observed=latency_ms,
            threshold=smoke.max_latency_ms,
        )
    return _check(
        CHECK_SMOKE_TEST,
        True,
        f"{len(frame)} rows scored; probabilities finite in [0, 1]; "
        f"latency {latency_ms:.1f} ms <= {smoke.max_latency_ms:.1f} ms",
        observed=latency_ms,
        threshold=smoke.max_latency_ms,
    )


def _registry_consistency(context: GateContext) -> PromotionCheck:
    if context.pointer is None:
        return _check(
            CHECK_REGISTRY_CONSISTENCY,
            True,
            "no active model; registry consistency is checked from the first promotion",
        )
    if context.alias_error is not None:
        return _check(
            CHECK_REGISTRY_CONSISTENCY,
            False,
            "cannot read the registry alias: "
            + context.alias_error
            + "; reconcile the registry before promoting",
        )
    if context.alias_version is None:
        return _check(
            CHECK_REGISTRY_CONSISTENCY,
            False,
            f"active pointer is version {context.pointer.model_version} but the registry "
            "alias is unset; reconcile by setting the alias to the active version",
        )
    if context.alias_version != context.pointer.model_version:
        return _check(
            CHECK_REGISTRY_CONSISTENCY,
            False,
            f"active pointer is version {context.pointer.model_version} but the alias "
            f"points at version {context.alias_version}; reconcile before promoting",
        )
    return _check(
        CHECK_REGISTRY_CONSISTENCY,
        True,
        f"active pointer and alias both reference version {context.pointer.model_version}",
    )


def evaluate_gate(context: GateContext) -> GateEvaluation:
    """Evaluate all checks in their documented order."""
    split = context.config.evaluation_split
    candidate_metrics, candidate_error = _evaluate_candidate(context)
    active_metrics, active_error = _evaluate_active(context)
    checks = [
        _run_finished(context.candidate),
        _model_logged(context.candidate),
        _evaluation_artifact(context.candidate),
        _dataset_identity(context.candidate, context.dataset, split),
        _required_artifacts(context.config, context.candidate),
        _quality_floor(context.config, candidate_metrics, candidate_error),
        _regression(
            context.config,
            context,
            candidate_metrics,
            candidate_error,
            active_metrics,
            active_error,
        ),
        _smoke_test(context.config, context),
        _registry_consistency(context),
    ]
    return GateEvaluation(
        checks=checks,
        candidate_metrics=candidate_metrics,
        active_metrics=active_metrics,
        failed=[check.name for check in checks if not check.passed],
    )


def _collapse(text: str) -> str:
    return " ".join(text.split())


def build_decision(
    *,
    decision_id: str,
    candidate_version: str,
    active_version: str | None,
    checks: list[PromotionCheck],
    verdict: PromotionVerdict,
    rejection_reasons: list[str],
    actor: str,
    reason: str,
    decided_at: datetime,
) -> PromotionDecision:
    """Build the frozen contract, refusing an approved verdict with failed checks."""
    failed = [check.name for check in checks if not check.passed]
    if verdict is PromotionVerdict.APPROVED and failed:
        raise PromotionError(
            "internal error: refusing to approve a decision with failed checks: "
            + ", ".join(failed)
        )
    return PromotionDecision(
        decision_id=decision_id,
        candidate_version=candidate_version,
        active_version=active_version,
        checks=checks,
        verdict=verdict,
        rejection_reasons=rejection_reasons,
        actor=actor,
        reason=reason,
        decided_at=decided_at,
    )


def _decision_record(
    decision: PromotionDecision,
    *,
    candidate_run_id: str,
    dataset_version: str,
    dataset_fingerprint: str,
    evaluation_split: str,
) -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision.decision_id,
        candidate_run_id=candidate_run_id,
        candidate_model_version=decision.candidate_version,
        active_model_version_before=decision.active_version,
        verdict=decision.verdict.value,
        checks=[check.model_dump(mode="json") for check in decision.checks],
        rejection_reasons=list(decision.rejection_reasons),
        actor=decision.actor,
        reason=decision.reason,
        decided_at=decision.decided_at,
        dataset_version=dataset_version,
        dataset_fingerprint=dataset_fingerprint,
        evaluation_split=evaluation_split,
    )


def decision_json_writer(decision: PromotionDecision, directory: Path) -> Callable[[], Path]:
    """Return a callback that writes the decision contract as canonical JSON."""

    def write() -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{decision.decision_id}.json"
        payload = decision.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    return write


@dataclass(frozen=True)
class PromotionRequest:
    """One explicit promotion operation requested by a human."""

    candidate_run_id: str
    mode: PromotionMode
    reason: str
    actor: str


@dataclass(frozen=True)
class PromotionResult:
    """Everything the CLI needs to report a promotion outcome."""

    checks: list[PromotionCheck]
    candidate_version: str | None
    active_before: str | None
    active_after: str | None
    verdict: PromotionVerdict
    dry_run: bool
    no_op: bool
    decision: PromotionDecision | None = None
    decision_path: Path | None = None
    metrics: Metrics | None = None


def perform_promotion(
    request: PromotionRequest,
    *,
    promotion_config: PromotionConfig,
    dataset: LoadedDataset,
    candidate: CandidateRun,
    registry: ModelRegistry,
    store: PointerStore,
    decision_dir: Path = DEFAULT_DECISION_DIR,
    evaluator: MetricEvaluator = _default_evaluator,
    predictor: ProbabilityPredictor = predict_probabilities,
    now: datetime | None = None,
) -> PromotionResult:
    """Evaluate the gate and, when allowed, commit a decision/pointer change."""
    if not request.reason.strip():
        raise PromotionError("a promotion reason is required")
    if not request.actor.strip():
        raise PromotionError("a promotion actor is required")

    decided_at = now or datetime.now(UTC)
    pointer = store.active()
    expected_active_version = pointer.model_version if pointer is not None else None

    candidate_version: str | None
    if request.mode == "dry-run":
        candidate_version = registry.find_version_for_run(candidate.run_id)
    elif candidate.model_id is not None and candidate.status == "FINISHED":
        try:
            candidate_version = registry.register_run(candidate.run_id, candidate.model_id)
        except RegistryError as exc:
            raise PromotionError(str(exc)) from exc
    else:
        candidate_version = None

    alias_version: str | None = None
    alias_error: str | None = None
    try:
        alias_version = registry.get_alias_version()
    except RegistryError as exc:
        alias_error = str(exc)

    active_model: Any | None = None
    active_model_error: str | None = None
    if pointer is not None:
        try:
            active_model = registry.load_version(pointer.model_version)
        except RegistryError as exc:
            active_model_error = str(exc)

    evaluation = evaluate_gate(
        GateContext(
            config=promotion_config,
            candidate=candidate,
            dataset=dataset,
            pointer=pointer,
            alias_version=alias_version,
            alias_error=alias_error,
            active_model=active_model,
            active_model_error=active_model_error,
            evaluator=evaluator,
            predictor=predictor,
            registry=registry,
            candidate_version=candidate_version,
        )
    )
    checks = evaluation.checks
    failed = evaluation.failed
    registered_version = candidate_version or "unregistered"

    if request.mode == "dry-run":
        return PromotionResult(
            checks=checks,
            candidate_version=candidate_version,
            active_before=expected_active_version,
            active_after=expected_active_version,
            verdict=PromotionVerdict.REJECTED if failed else PromotionVerdict.APPROVED,
            dry_run=True,
            no_op=False,
            metrics=evaluation.candidate_metrics,
        )

    verdict = PromotionVerdict.REJECTED if request.mode == "reject" else None
    if verdict is None:
        verdict = PromotionVerdict.REJECTED if failed else PromotionVerdict.APPROVED

    if verdict is PromotionVerdict.REJECTED:
        failed_reasons = [
            _collapse(f"{check.name}: {check.detail}") for check in checks if not check.passed
        ]
        if request.mode == "reject":
            reasons = [request.reason.strip(), *failed_reasons]
        else:
            reasons = failed_reasons
            if not reasons:
                raise PromotionError(
                    "internal error: rejected decision without failed checks or a reason"
                )
        reasons = list(dict.fromkeys(reasons))
        decision = build_decision(
            decision_id=f"dec-{uuid.uuid4().hex}",
            candidate_version=registered_version,
            active_version=expected_active_version,
            checks=checks,
            verdict=PromotionVerdict.REJECTED,
            rejection_reasons=reasons,
            actor=request.actor,
            reason=request.reason.strip(),
            decided_at=decided_at,
        )
        record = _decision_record(
            decision,
            candidate_run_id=candidate.run_id,
            dataset_version=dataset.manifest.dataset_version,
            dataset_fingerprint=dataset.manifest.fingerprint,
            evaluation_split=promotion_config.evaluation_split,
        )
        path = store.record_decision(
            record,
            write_decision_json=decision_json_writer(decision, decision_dir),
        )
        return PromotionResult(
            checks=checks,
            candidate_version=candidate_version,
            active_before=expected_active_version,
            active_after=expected_active_version,
            verdict=PromotionVerdict.REJECTED,
            dry_run=False,
            no_op=False,
            decision=decision,
            decision_path=path,
            metrics=evaluation.candidate_metrics,
        )

    if failed:  # defensive; an approved verdict is impossible with a failed check
        raise PromotionError(
            "internal error: approved verdict with failed checks: " + ", ".join(failed)
        )
    if candidate_version is None:
        raise PromotionError("the candidate has no registered model version; registration failed")

    if pointer is not None and candidate_version == pointer.model_version:
        return PromotionResult(
            checks=checks,
            candidate_version=candidate_version,
            active_before=expected_active_version,
            active_after=expected_active_version,
            verdict=PromotionVerdict.APPROVED,
            dry_run=False,
            no_op=True,
            metrics=evaluation.candidate_metrics,
        )

    decision = build_decision(
        decision_id=f"dec-{uuid.uuid4().hex}",
        candidate_version=candidate_version,
        active_version=expected_active_version,
        checks=checks,
        verdict=PromotionVerdict.APPROVED,
        rejection_reasons=[],
        actor=request.actor,
        reason=request.reason.strip(),
        decided_at=decided_at,
    )
    record = _decision_record(
        decision,
        candidate_run_id=candidate.run_id,
        dataset_version=dataset.manifest.dataset_version,
        dataset_fingerprint=dataset.manifest.fingerprint,
        evaluation_split=promotion_config.evaluation_split,
    )
    target = ActiveModel(
        model_name=promotion_config.registry.registered_model_name,
        model_version=candidate_version,
        mlflow_run_id=candidate.run_id,
        mlflow_model_id=candidate.model_id or "",
        dataset_version=dataset.manifest.dataset_version,
        dataset_fingerprint=dataset.manifest.fingerprint,
        decision_id=decision.decision_id,
        promoted_at=decided_at,
        updated_at=decided_at,
    )

    def set_alias() -> None:
        registry.set_alias(candidate_version)

    def restore_alias() -> None:
        if alias_version is None:
            registry.delete_alias()
        else:
            registry.set_alias(alias_version)

    path = store.apply_promotion(
        decision=record,
        expected_active_version=expected_active_version,
        operation="bootstrap" if pointer is None else "promotion",
        target=target,
        set_alias=set_alias,
        restore_alias=restore_alias,
        write_decision_json=decision_json_writer(decision, decision_dir),
    )
    return PromotionResult(
        checks=checks,
        candidate_version=candidate_version,
        active_before=expected_active_version,
        active_after=candidate_version,
        verdict=PromotionVerdict.APPROVED,
        dry_run=False,
        no_op=False,
        decision=decision,
        decision_path=path,
        metrics=evaluation.candidate_metrics,
    )


def build_model_descriptor(
    *,
    candidate: CandidateRun,
    metrics: Metrics,
    model_version: str,
    dataset_fingerprint: str,
    decision: PromotionDecision,
    registry_alias: str,
    artifact_uri: str,
    created_at: datetime | None = None,
) -> ModelDescriptor:
    """Build the frozen descriptor for an approved version, with approval metadata."""
    if decision.verdict is not PromotionVerdict.APPROVED:
        raise PromotionError("a ModelDescriptor can only be built from an approved decision")
    evaluation_metrics = EvaluationMetrics(
        roc_auc=metrics.roc_auc,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        accuracy=metrics.accuracy,
        sample_count=metrics.sample_count,
        class_counts={
            "0": metrics.true_negative + metrics.false_positive,
            "1": metrics.false_negative + metrics.true_positive,
        },
        confusion_matrix=ConfusionMatrix(
            true_negative=metrics.true_negative,
            false_positive=metrics.false_positive,
            false_negative=metrics.false_negative,
            true_positive=metrics.true_positive,
        ),
    )
    return ModelDescriptor(
        model_version=model_version,
        mlflow_run_id=candidate.run_id,
        dataset_fingerprint=dataset_fingerprint,
        artifact_uri=artifact_uri,
        evaluation_split="validation",
        metrics=evaluation_metrics,
        approval=ApprovalMetadata(
            approved_by=decision.actor,
            approved_at=decision.decided_at,
            decision_id=decision.decision_id,
            reason=decision.reason,
            registry_alias=registry_alias,
        ),
        created_at=created_at or decision.decided_at,
    )


def decision_tags(decision: PromotionDecision) -> dict[str, str]:
    """Additive MLflow run tags that link a candidate run to its decision."""
    return {
        "promotion_decision_id": decision.decision_id,
        "promotion_verdict": decision.verdict.value,
        "promotion_actor": decision.actor,
    }


def tag_candidate_run(tracking_uri: str, run_id: str, decision: PromotionDecision) -> None:
    """Write the decision tags onto the candidate run without touching training tags."""
    client = MlflowClient(tracking_uri=tracking_uri)
    for key, value in decision_tags(decision).items():
        client.set_tag(run_id, key, value)


def registry_for(
    *,
    tracking_uri: str,
    promotion_config: PromotionConfig,
) -> MlflowModelRegistry:
    """Build the real MLflow registry from the promotion config."""
    return MlflowModelRegistry(
        tracking_uri=tracking_uri,
        registered_model_name=promotion_config.registry.registered_model_name,
        alias=promotion_config.registry.alias,
    )


__all__ = [
    "CandidateRun",
    "GateContext",
    "GateEvaluation",
    "MlflowCandidateInspector",
    "PromotionError",
    "PromotionRequest",
    "PromotionResult",
    "build_decision",
    "build_model_descriptor",
    "evaluate_gate",
    "perform_promotion",
    "registry_for",
    "PromotionConflict",
    "DEFAULT_DECISION_DIR",
]
