"""Verified rollback to a previously active model version.

Rollback is a lifecycle operation, not a re-evaluation: the target must have
been active before, so its approval is already recorded. It still refuses an
unloadable target by loading and smoke-scoring it *before* the critical
section. The pointer move then reuses the W3 transaction shape: advisory lock,
re-read, expected-version conflict, alias move, transition append, pointer last
(ADR-012).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from modelops_lab.pointer import (
    ActiveModel,
    PointerStore,
    PromotionConflict,
    TransitionRecord,
)
from modelops_lab.registry import ModelRegistry, Reconciliation, RegistryError, reconcile
from modelops_lab.serving import (
    FeatureSpec,
    feature_spec_from_model,
    positive_probability,
    redact_secrets,
)
from modelops_lab.tracking import redact_uri

DEFAULT_ROLLBACK_DIR = Path("artifacts/rollbacks")


class RollbackError(RuntimeError):
    """Raised when a rollback cannot be verified or recorded."""


@dataclass(frozen=True)
class RollbackArtifact:
    """Registry metadata needed to reconstruct the pointer for a target version."""

    run_id: str
    model_id: str
    dataset_version: str
    dataset_fingerprint: str


class RollbackResolver(Protocol):
    """Metadata and model loading for a registered version."""

    def resolve(self, model_name: str, version: str) -> RollbackArtifact: ...

    def load(self, model_name: str, version: str) -> Any: ...


class MlflowRollbackResolver:
    """Resolve a registered version and load it from MLflow."""

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def resolve(self, model_name: str, version: str) -> RollbackArtifact:
        try:
            model_version = self._client.get_model_version(model_name, version)
            run = self._client.get_run(model_version.run_id)
        except (MlflowException, OSError, ValueError) as exc:
            raise RegistryError(
                f"cannot read version {version} of {model_name!r} from "
                f"{redact_uri(self.tracking_uri)}: {exc}"
            ) from exc
        tags = {key: str(value) for key, value in run.data.tags.items()}
        outputs = run.outputs.model_outputs if run.outputs is not None else []
        model_id = (
            outputs[0].model_id if outputs and outputs[0].model_id else tags.get("model_id", "")
        )
        dataset_version = tags.get("dataset_version")
        dataset_fingerprint = tags.get("dataset_fingerprint")
        if not dataset_version or not dataset_fingerprint:
            raise RollbackError(
                f"run {model_version.run_id} for version {version} lacks dataset_version or "
                "dataset_fingerprint tags; the pointer cannot be reconstructed safely"
            )
        return RollbackArtifact(
            run_id=str(model_version.run_id),
            model_id=str(model_id),
            dataset_version=dataset_version,
            dataset_fingerprint=dataset_fingerprint,
        )

    def load(self, model_name: str, version: str) -> Any:
        model_uri = f"models:/{model_name}/{version}"
        try:
            return mlflow.sklearn.load_model(model_uri)
        except (MlflowException, OSError, ValueError) as exc:
            raise RegistryError(f"cannot load {model_uri}: {exc}") from exc


@dataclass(frozen=True)
class SmokeVerification:
    """Pre-commit proof that the rollback target can actually score a request."""

    sample_count: int
    latency_ms: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": True,
            "sample_count": self.sample_count,
            "latency_ms": round(self.latency_ms, 3),
            "detail": self.detail,
        }


def smoke_predict(model: Any, spec: FeatureSpec | None = None) -> SmokeVerification:
    """Score one schema-valid row derived from the model's own fitted metadata."""
    resolved = spec or feature_spec_from_model(model)
    row: dict[str, float | str] = {name: 0.0 for name in resolved.numeric}
    for name, levels in resolved.categorical.items():
        if not levels:
            raise RollbackError(f"categorical feature {name!r} has no known levels")
        row[name] = levels[0]
    frame = pd.DataFrame([row], columns=list(resolved.names))
    started = time.perf_counter()
    try:
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        probability = positive_probability(model, probabilities)
    except Exception as exc:
        raise RollbackError(
            "the rollback target failed its pre-commit smoke prediction: "
            + redact_secrets(f"{type(exc).__name__}: {exc}")
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0
    return SmokeVerification(
        sample_count=1,
        latency_ms=latency_ms,
        detail=(
            f"loaded and scored 1 synthetic schema-valid row; "
            f"probability {probability:.6f} in [0, 1]; latency {latency_ms:.3f} ms"
        ),
    )


@dataclass(frozen=True)
class RollbackResult:
    rollback_id: str | None
    from_version: str
    to_version: str
    no_op: bool
    verification: SmokeVerification | None
    audit_path: Path | None = None


def _authorizing_transition(
    transitions: list[TransitionRecord], model_name: str, target_version: str
) -> TransitionRecord | None:
    matching = [
        transition
        for transition in transitions
        if transition.model_name == model_name and transition.to_model_version == target_version
    ]
    return matching[-1] if matching else None


def _rollback_payload(
    *,
    rollback_id: str,
    from_version: str,
    to_version: str,
    model_name: str,
    run_id: str,
    actor: str,
    reason: str,
    timestamp: datetime,
    verification: SmokeVerification,
    authorizing_decision_id: str,
) -> dict[str, Any]:
    return {
        "rollback_id": rollback_id,
        "model_name": model_name,
        "from_model_version": from_version,
        "to_model_version": to_version,
        "mlflow_run_id": run_id,
        "actor": actor,
        "reason": reason,
        "timestamp": timestamp.isoformat(),
        "authorizing_decision_id": authorizing_decision_id,
        "verification": verification.as_dict(),
    }


def _audit_writer(payload: dict[str, Any], directory: Path) -> Callable[[], Path]:
    def write() -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{payload['rollback_id']}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    return write


def perform_rollback(
    *,
    target_version: str,
    reason: str,
    actor: str,
    store: PointerStore,
    resolver: RollbackResolver,
    registry: ModelRegistry,
    rollback_dir: Path | None = None,
    now: datetime | None = None,
    smoke: Callable[[Any], SmokeVerification] = smoke_predict,
) -> RollbackResult:
    """Restore ``target_version`` if it was previously active and can serve."""
    if not reason.strip():
        raise RollbackError("a rollback reason is required")
    if not actor.strip():
        raise RollbackError("a rollback actor is required")

    pointer = store.active()
    if pointer is None:
        raise RollbackError(
            "no model is active; rollback needs an active version to move away from"
        )
    if target_version == pointer.model_version:
        return RollbackResult(
            rollback_id=None,
            from_version=pointer.model_version,
            to_version=target_version,
            no_op=True,
            verification=None,
        )

    transitions = store.transitions()
    authorizing = _authorizing_transition(transitions, pointer.model_name, target_version)
    if authorizing is None:
        raise RollbackError(
            f"version {target_version} was never active for {pointer.model_name!r}; "
            "rollback can only target a previously active (approved) version. "
            "Use `modelops promote` for a registered candidate."
        )

    # Verify before moving anything: load the registered target and score it.
    try:
        artifact = resolver.resolve(pointer.model_name, target_version)
        model = resolver.load(pointer.model_name, target_version)
        verification = smoke(model)
    except (RollbackError, RegistryError):
        raise
    except Exception as exc:
        raise RollbackError(
            f"cannot verify rollback target version {target_version}: "
            + redact_secrets(f"{type(exc).__name__}: {exc}")
        ) from exc

    timestamp = now or datetime.now(UTC)
    rollback_id = f"rb-{uuid.uuid4().hex}"
    target = ActiveModel(
        model_name=pointer.model_name,
        model_version=target_version,
        mlflow_run_id=artifact.run_id,
        mlflow_model_id=artifact.model_id,
        dataset_version=artifact.dataset_version,
        dataset_fingerprint=artifact.dataset_fingerprint,
        decision_id=authorizing.decision_id,
        promoted_at=timestamp,
        updated_at=timestamp,
    )
    payload = _rollback_payload(
        rollback_id=rollback_id,
        from_version=pointer.model_version,
        to_version=target_version,
        model_name=pointer.model_name,
        run_id=artifact.run_id,
        actor=actor,
        reason=reason.strip(),
        timestamp=timestamp,
        verification=verification,
        authorizing_decision_id=authorizing.decision_id,
    )

    previous_alias = registry.get_alias_version()
    audit_path = store.apply_rollback(
        expected_active_version=pointer.model_version,
        target=target,
        transition_decision_id=authorizing.decision_id,
        actor=actor,
        reason=reason.strip(),
        occurred_at=timestamp,
        set_alias=lambda: registry.set_alias(target_version),
        restore_alias=(
            (lambda: registry.set_alias(previous_alias))
            if previous_alias is not None
            else registry.delete_alias
        ),
        write_rollback_json=_audit_writer(payload, rollback_dir or DEFAULT_ROLLBACK_DIR),
    )
    return RollbackResult(
        rollback_id=rollback_id,
        from_version=pointer.model_version,
        to_version=target_version,
        no_op=False,
        verification=verification,
        audit_path=audit_path,
    )


@dataclass(frozen=True)
class ReconcileResult:
    pointer_version: str | None
    alias_version: str | None
    consistent: bool
    fixed: bool
    detail: str


def perform_reconcile(
    *, store: PointerStore, registry: ModelRegistry, fix: bool = False
) -> ReconcileResult:
    """Compare the pointer with the alias, optionally repairing it under the lock."""
    pointer = store.active()
    pointer_version = pointer.model_version if pointer is not None else None
    comparison: Reconciliation = reconcile(registry, pointer_version)
    if comparison.consistent:
        return ReconcileResult(
            pointer_version=pointer_version,
            alias_version=comparison.alias_version,
            consistent=True,
            fixed=False,
            detail=comparison.detail,
        )
    if not fix:
        return ReconcileResult(
            pointer_version=pointer_version,
            alias_version=comparison.alias_version,
            consistent=False,
            fixed=False,
            detail=comparison.detail,
        )

    if pointer_version is None:
        store.apply_alias_reconciliation(
            expected_active_version=None,
            set_alias=lambda: None,
            delete_alias=registry.delete_alias,
        )
    else:
        store.apply_alias_reconciliation(
            expected_active_version=pointer_version,
            set_alias=lambda: registry.set_alias(pointer_version),
            delete_alias=registry.delete_alias,
        )
    repaired = reconcile(registry, pointer_version)
    if not repaired.consistent:  # defensive: the write did not take effect
        raise RollbackError(
            "reconciliation write reported success but the alias still disagrees: "
            + repaired.detail
        )
    return ReconcileResult(
        pointer_version=pointer_version,
        alias_version=repaired.alias_version,
        consistent=True,
        fixed=True,
        detail=f"alias repaired to pointer version {pointer_version}"
        if pointer_version is not None
        else "alias removed because no model is active",
    )


__all__ = [
    "DEFAULT_ROLLBACK_DIR",
    "MlflowRollbackResolver",
    "ReconcileResult",
    "RollbackArtifact",
    "RollbackError",
    "RollbackResolver",
    "RollbackResult",
    "SmokeVerification",
    "perform_reconcile",
    "perform_rollback",
    "smoke_predict",
    "PromotionConflict",
]
