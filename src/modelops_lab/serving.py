"""FastAPI inference service, prediction-event logging, and serving metrics.

The served model is never inferred from the newest artifact. On every request
the service reads the PostgreSQL pointer (authoritative, ADR-006), loads the
referenced MLflow model version on first use, and caches it by
``(model_name, version, run_id)``. A rollback therefore takes effect on the
next request without a restart.

Feature expectations come from the fitted pipeline itself:
``feature_names_in_`` supplies the ordered names and the fitted
``OneHotEncoder`` supplies the known categorical levels. Nothing about the
synthetic schema is hardcoded here.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from modelops_lab.config import RuntimeSettings, load_promotion_config, load_runtime_settings
from modelops_lab.contracts import PredictionEvent
from modelops_lab.db import PredictionEventRow, create_engine_for
from modelops_lab.pointer import ActiveModel, DecisionRecord, PointerStore
from modelops_lab.registry import MlflowModelRegistry

LOGGER = logging.getLogger("modelops_lab.serving")
DEFAULT_DECISION_THRESHOLD = 0.5
MAX_REQUEST_ID_LENGTH = 128
_SECRET_IN_URI = re.compile(r"(://[^:/@\s]+:)([^@\s]+)(@)")

NO_ACTIVE_MODEL_MESSAGE = (
    "no model is active; promote a candidate with `modelops promote --approve` "
    "and confirm the pointer and alias with `modelops registry reconcile`"
)


class ModelUnavailable(RuntimeError):
    """Raised when no healthy model can serve the request (HTTP 503)."""


class FeatureValidationError(ValueError):
    """Raised internally when a request payload does not match the model schema."""

    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        super().__init__("feature validation failed")


def redact_secrets(text: str) -> str:
    """Remove passwords from URI-shaped text before it reaches a log or client."""
    return _SECRET_IN_URI.sub(r"\1***\3", str(text))


def _json_log(logger: logging.Logger, payload: dict[str, Any], level: int = logging.INFO) -> None:
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Feature expectations and validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Ordered feature names and the categorical levels learned during fit."""

    names: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: Mapping[str, tuple[str, ...]]

    def describe(self) -> dict[str, Any]:
        return {
            "features": list(self.names),
            "numeric_features": list(self.numeric),
            "categorical_features": {
                name: list(levels) for name, levels in self.categorical.items()
            },
        }


def _encoder_levels(transformer: Any, columns: Any) -> list[tuple[str, tuple[str, ...]]]:
    categories = getattr(transformer, "categories_", None)
    if categories is None:
        return []
    fitted_names = getattr(transformer, "feature_names_in_", None)
    if fitted_names is not None:
        names = [str(name) for name in fitted_names]
    else:
        names = [str(name) for name in columns]
    if len(names) != len(categories):
        raise ModelUnavailable(
            "the fitted encoder exposes inconsistent category metadata; refusing to serve it"
        )
    return [
        (name, tuple(str(level) for level in levels))
        for name, levels in zip(names, categories, strict=True)
    ]


def feature_spec_from_model(model: Any) -> FeatureSpec:
    """Derive the accepted input schema from a fitted pipeline.

    A pipeline without ``feature_names_in_`` cannot be served safely because
    positional feature order would be a guess. That is a hard error, not a
    fallback.
    """
    raw_names = getattr(model, "feature_names_in_", None)
    if raw_names is None:
        raise ModelUnavailable(
            "the loaded pipeline does not expose feature_names_in_; it was not fitted on a "
            "named DataFrame and cannot be served safely"
        )
    names = tuple(str(name) for name in raw_names)
    if not names:
        raise ModelUnavailable("the loaded pipeline declares no input features")

    named_steps = getattr(model, "named_steps", None) or {}
    candidates = list(named_steps.values()) if named_steps else [model]
    categorical: dict[str, tuple[str, ...]] = {}
    for step in candidates:
        transformers = getattr(step, "transformers_", None)
        if transformers is None:
            for name, levels in _encoder_levels(step, []):
                categorical[name] = levels
            continue
        for _name, transformer, columns in transformers:
            for feature, levels in _encoder_levels(transformer, columns):
                categorical[feature] = levels

    unknown = sorted(set(categorical) - set(names))
    if unknown:
        raise ModelUnavailable(
            "the fitted encoder references features the pipeline does not declare: "
            + ", ".join(unknown)
        )
    numeric = tuple(name for name in names if name not in categorical)
    return FeatureSpec(names=names, numeric=numeric, categorical=categorical)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def validate_features(payload: Mapping[str, Any], spec: FeatureSpec) -> dict[str, float | str]:
    """Validate a flat JSON object against ``spec`` without coercing values.

    Missing, extra, non-finite, wrongly-typed, and unknown-category values are
    all reported with the offending field name. Numeric strings are rejected
    rather than parsed; categorical numbers are rejected rather than stringified.
    """
    errors: list[dict[str, str]] = []
    expected = set(spec.names)
    supplied = set(payload)
    for name in spec.names:
        if name not in supplied:
            errors.append({"field": name, "message": "missing required feature"})
    for name in sorted(supplied - expected):
        errors.append({"field": name, "message": "unexpected feature (not used by the model)"})

    cleaned: dict[str, float | str] = {}
    for name in spec.numeric:
        if name not in payload:
            continue
        value = payload[name]
        if not _is_finite_number(value):
            errors.append(
                {
                    "field": name,
                    "message": ("must be a finite JSON number (booleans and strings are rejected)"),
                }
            )
        else:
            cleaned[name] = float(value)

    for name, levels in spec.categorical.items():
        if name not in payload:
            continue
        value = payload[name]
        if not isinstance(value, str):
            errors.append(
                {
                    "field": name,
                    "message": "must be one of the known string levels: " + ", ".join(levels),
                }
            )
        elif value not in levels:
            errors.append(
                {
                    "field": name,
                    "message": f"unknown category {value!r}; expected one of: " + ", ".join(levels),
                }
            )
        else:
            cleaned[name] = value

    if errors:
        raise FeatureValidationError(errors)
    return cleaned


# ---------------------------------------------------------------------------
# Prediction event storage
# ---------------------------------------------------------------------------


class PredictionEventRepository(Protocol):
    """Minimal sink for served predictions; failures must never fail inference."""

    def record(self, event: PredictionEvent) -> None: ...

    def ping(self) -> bool: ...


class SqlPredictionEventStore:
    """Best-effort Postgres/SQLite writer for ``prediction_events``."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._engine = None
        self._session_factory: sessionmaker[Session] | None = None
        self._lock = Lock()

    def _ensure_engine(self) -> sessionmaker[Session]:
        with self._lock:
            if self._session_factory is None:
                self._engine = create_engine_for(self.database_url)
                self._session_factory = sessionmaker(
                    bind=self._engine, expire_on_commit=False, future=True
                )
            return self._session_factory

    def record(self, event: PredictionEvent) -> None:
        session_factory = self._ensure_engine()
        with session_factory() as session, session.begin():
            session.add(
                PredictionEventRow(
                    request_id=event.request_id,
                    timestamp=event.timestamp,
                    features=event.features,
                    prediction=event.prediction,
                    probability=event.probability,
                    model_version=event.model_version,
                    latency_ms=event.latency_ms,
                )
            )

    def ping(self) -> bool:
        try:
            factory = self._ensure_engine()
            with factory() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class ServingMetrics:
    """Prometheus collectors scoped to one app instance (safe in tests)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.predictions = Counter(
            "modelops_predictions_total",
            "Inference requests by served model version and outcome.",
            ["model_version", "outcome"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "modelops_prediction_latency_seconds",
            "Server-side inference latency.",
            ["model_version"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.model_load_failures = Counter(
            "modelops_model_load_failures_total",
            "Active-model loads that failed.",
            registry=self.registry,
        )
        self.active_model_info = Gauge(
            "modelops_active_model_info",
            "Active model metadata; value is always 1 while a model is active.",
            ["model_version", "model_name", "run_id"],
            registry=self.registry,
        )
        self.alias_mismatches = Counter(
            "modelops_registry_alias_mismatch_total",
            "Times the served pointer disagreed with the MLflow champion alias.",
            registry=self.registry,
        )
        self.event_failures = Counter(
            "modelops_event_log_failures_total",
            "Prediction events that could not be persisted.",
            registry=self.registry,
        )
        self.events = Counter(
            "modelops_event_writes_total",
            "Prediction-event persistence attempts by outcome.",
            ["outcome"],
            registry=self.registry,
        )

    def observe_prediction(self, version: str, outcome: str, latency_seconds: float) -> None:
        self.predictions.labels(model_version=version, outcome=outcome).inc()
        self.latency.labels(model_version=version).observe(latency_seconds)

    def set_active(self, pointer: ActiveModel | None) -> None:
        self.active_model_info.clear()
        if pointer is not None:
            self.active_model_info.labels(
                model_version=pointer.model_version,
                model_name=pointer.model_name,
                run_id=pointer.mlflow_run_id,
            ).set(1)


# ---------------------------------------------------------------------------
# Active-model provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalInfo:
    decision_id: str
    actor: str | None
    reason: str | None
    approved_at: datetime | None


@dataclass(frozen=True)
class ServedModel:
    """A loaded model plus the pointer metadata that selected it."""

    model_name: str
    version: str
    run_id: str
    dataset_version: str
    dataset_fingerprint: str
    decision_id: str
    promoted_at: datetime
    updated_at: datetime
    model: Any
    spec: FeatureSpec
    approval: ApprovalInfo

    @property
    def artifact_uri(self) -> str:
        return f"models:/{self.model_name}/{self.version}"


@dataclass(frozen=True)
class ProviderStatus:
    database_reachable: bool
    pointer: ActiveModel | None
    model_loaded: bool
    loaded_version: str | None
    error: str | None = None


class PointerModelProvider:
    """Resolve the served model from the pointer on every call.

    ``pointer_reader``, ``loader``, ``alias_reader``, and ``decision_reader``
    are injectable so the unit tier can exercise the full cache/reload logic
    without Docker.
    """

    def __init__(
        self,
        *,
        pointer_reader: Callable[[], ActiveModel | None],
        loader: Callable[[str, str], Any],
        metrics: ServingMetrics,
        alias_reader: Callable[[], str | None] | None = None,
        decision_reader: Callable[[str], DecisionRecord | None] | None = None,
        database_ping: Callable[[], bool] | None = None,
        alias_check_interval_seconds: float = 10.0,
        max_cached_models: int = 8,
        logger: logging.Logger | None = None,
    ) -> None:
        self._pointer_reader = pointer_reader
        self._loader = loader
        self._metrics = metrics
        self._alias_reader = alias_reader
        self._decision_reader = decision_reader
        self._database_ping = database_ping
        self._alias_check_interval_seconds = alias_check_interval_seconds
        self._max_cached_models = max(1, max_cached_models)
        self._logger = logger or LOGGER
        self._cache: OrderedDict[tuple[str, str, str], ServedModel] = OrderedDict()
        self._lock = Lock()
        self._last_alias_check: float | None = None

    @staticmethod
    def _cache_key(pointer: ActiveModel) -> tuple[str, str, str]:
        return (pointer.model_name, pointer.model_version, pointer.mlflow_run_id)

    def _approval(self, decision_id: str) -> ApprovalInfo:
        decision: DecisionRecord | None = None
        if self._decision_reader is not None:
            try:
                decision = self._decision_reader(decision_id)
            except Exception as exc:  # approval metadata is explanatory, not load-critical
                _json_log(
                    self._logger,
                    {
                        "event": "approval_read_failed",
                        "decision_id": decision_id,
                        "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
                    },
                    level=logging.WARNING,
                )
        if decision is None:
            return ApprovalInfo(decision_id=decision_id, actor=None, reason=None, approved_at=None)
        return ApprovalInfo(
            decision_id=decision.decision_id,
            actor=decision.actor,
            reason=decision.reason,
            approved_at=decision.decided_at,
        )

    def _check_alias(self, pointer: ActiveModel) -> None:
        if self._alias_reader is None:
            return
        now = time.monotonic()
        if (
            self._last_alias_check is not None
            and now - self._last_alias_check < self._alias_check_interval_seconds
        ):
            return
        self._last_alias_check = now
        try:
            alias_version = self._alias_reader()
        except Exception as exc:
            _json_log(
                self._logger,
                {
                    "event": "registry_alias_check_failed",
                    "model_name": pointer.model_name,
                    "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
                },
                level=logging.WARNING,
            )
            return
        if alias_version != pointer.model_version:
            self._metrics.alias_mismatches.inc()
            _json_log(
                self._logger,
                {
                    "event": "registry_alias_mismatch",
                    "model_name": pointer.model_name,
                    "pointer_version": pointer.model_version,
                    "alias_version": alias_version,
                    "action": "serving the pointer; run `modelops registry reconcile --fix`",
                },
                level=logging.WARNING,
            )

    def get_active(self) -> ServedModel | None:
        try:
            pointer = self._pointer_reader()
        except Exception as exc:
            raise ModelUnavailable(
                "cannot read the active model pointer; the database is unavailable. "
                "Check MODELOPS_DATABASE_URL and run `modelops db upgrade`."
            ) from exc

        if pointer is None:
            self._metrics.set_active(None)
            return None

        self._check_alias(pointer)
        key = self._cache_key(pointer)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            try:
                model = self._loader(pointer.model_name, pointer.model_version)
                spec = feature_spec_from_model(model)
            except ModelUnavailable:
                self._metrics.model_load_failures.inc()
                raise
            except Exception as exc:
                self._metrics.model_load_failures.inc()
                _json_log(
                    self._logger,
                    {
                        "event": "model_load_failed",
                        "model_name": pointer.model_name,
                        "model_version": pointer.model_version,
                        "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
                    },
                    level=logging.ERROR,
                )
                raise ModelUnavailable(
                    f"cannot load model {pointer.model_name!r} version "
                    f"{pointer.model_version!r} from the MLflow registry; see the serving log"
                ) from exc

            served = ServedModel(
                model_name=pointer.model_name,
                version=pointer.model_version,
                run_id=pointer.mlflow_run_id,
                dataset_version=pointer.dataset_version,
                dataset_fingerprint=pointer.dataset_fingerprint,
                decision_id=pointer.decision_id,
                promoted_at=pointer.promoted_at,
                updated_at=pointer.updated_at,
                model=model,
                spec=spec,
                approval=self._approval(pointer.decision_id),
            )
            self._cache[key] = served
            while len(self._cache) > self._max_cached_models:
                self._cache.popitem(last=False)
            self._metrics.set_active(pointer)
            return served

    def status(self) -> ProviderStatus:
        database_reachable = True
        error: str | None = None
        if self._database_ping is not None:
            try:
                database_reachable = bool(self._database_ping())
            except Exception:
                database_reachable = False
        pointer: ActiveModel | None = None
        try:
            pointer = self._pointer_reader()
        except Exception as exc:
            database_reachable = False
            error = "cannot read the active model pointer: " + redact_secrets(
                f"{type(exc).__name__}: {exc}"
            )
        model_loaded = False
        loaded_version: str | None = None
        if pointer is not None:
            with self._lock:
                cached = self._cache.get(self._cache_key(pointer))
            if cached is not None:
                model_loaded = True
                loaded_version = cached.version
        return ProviderStatus(
            database_reachable=database_reachable,
            pointer=pointer,
            model_loaded=model_loaded,
            loaded_version=loaded_version,
            error=error,
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    raw = request.headers.get("x-request-id")
    if raw is None:
        return uuid.uuid4().hex
    value = raw.strip()
    if not value:
        return uuid.uuid4().hex
    if len(value) > MAX_REQUEST_ID_LENGTH or not raw.isprintable():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_request_id",
                "message": (f"X-Request-ID must be 1-{MAX_REQUEST_ID_LENGTH} printable characters"),
                "errors": [
                    {
                        "field": "X-Request-ID",
                        "message": (
                            f"must be 1-{MAX_REQUEST_ID_LENGTH} printable characters; "
                            f"received {len(value)}"
                        ),
                    }
                ],
            },
        )
    return value


def positive_probability(model: Any, probabilities: Any) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ModelUnavailable("the model returned an unexpected probability shape")
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        index = classes.index(1)
    elif len(classes) == 1:
        return 1.0 if int(classes[0]) == 1 else 0.0
    else:
        raise ModelUnavailable("the model does not expose a positive class")
    if index >= values.shape[1]:
        raise ModelUnavailable("the model did not return a probability for the positive class")
    probability = float(values[0, index])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ModelUnavailable("the model returned a non-finite or out-of-range probability")
    return probability


def _http_503(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": {"error": "model_unavailable", "message": message}},
    )


def _resolve_default_provider(
    settings: RuntimeSettings, metrics: ServingMetrics, event_store: PredictionEventRepository
) -> PointerModelProvider:
    promotion = load_promotion_config()
    pointer_store = PointerStore(settings.database_url)
    registry = MlflowModelRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=promotion.registry.registered_model_name,
        alias=promotion.registry.alias,
    )

    def load(name: str, version: str) -> Any:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        return mlflow.sklearn.load_model(f"models:/{name}/{version}")

    return PointerModelProvider(
        pointer_reader=pointer_store.active,
        loader=load,
        metrics=metrics,
        alias_reader=registry.get_alias_version,
        decision_reader=pointer_store.decision,
        database_ping=event_store.ping,
    )


def create_app(
    *,
    settings: RuntimeSettings | None = None,
    provider: PointerModelProvider | None = None,
    event_store: PredictionEventRepository | None = None,
    metrics: ServingMetrics | None = None,
    logger: logging.Logger | None = None,
) -> FastAPI:
    """Build the inference app; every seam is injectable for the unit tier."""
    runtime = settings or load_runtime_settings()
    log = logger or LOGGER
    serving_metrics = metrics or ServingMetrics()
    store = event_store or SqlPredictionEventStore(runtime.database_url)
    active_provider = provider or _resolve_default_provider(runtime, serving_metrics, store)

    app = FastAPI(
        title="ModelOps Lab Inference",
        version="0.1.0",
        description="Serves only the model referenced by the PostgreSQL active pointer.",
    )
    app.state.settings = runtime
    app.state.provider = active_provider
    app.state.event_store = store
    app.state.metrics = serving_metrics

    @app.get("/health")
    def health() -> JSONResponse:
        status = active_provider.status()
        payload: dict[str, Any] = {
            "status": "ok" if status.database_reachable else "degraded",
            "database_reachable": status.database_reachable,
            "model_loaded": status.model_loaded,
            "active_model_version": (
                status.pointer.model_version if status.pointer is not None else None
            ),
        }
        if status.error is not None:
            payload["detail"] = status.error
        return JSONResponse(status_code=200, content=payload)

    @app.get("/model")
    def model_info() -> JSONResponse:
        served = active_provider.get_active()
        if served is None:
            raise ModelUnavailable(NO_ACTIVE_MODEL_MESSAGE)
        approval = served.approval
        payload = {
            "model_name": served.model_name,
            "model_version": served.version,
            "mlflow_run_id": served.run_id,
            "dataset_version": served.dataset_version,
            "dataset_fingerprint": served.dataset_fingerprint,
            "artifact_uri": served.artifact_uri,
            "promoted_at": served.promoted_at.isoformat(),
            "updated_at": served.updated_at.isoformat(),
            "created_at": (
                approval.approved_at.isoformat()
                if approval.approved_at is not None
                else served.promoted_at.isoformat()
            ),
            "approval": {
                "decision_id": approval.decision_id,
                "approved_by": approval.actor,
                "approved_at": (
                    approval.approved_at.isoformat() if approval.approved_at is not None else None
                ),
                "reason": approval.reason,
            },
            "features": served.spec.describe(),
        }
        return JSONResponse(status_code=200, content=payload)

    @app.post("/predict")
    async def predict(request: Request) -> JSONResponse:
        served = active_provider.get_active()
        if served is None:
            raise ModelUnavailable(NO_ACTIVE_MODEL_MESSAGE)

        request_id = _request_id(request)
        version = served.version
        try:
            raw = await request.json()
        except Exception:
            serving_metrics.observe_prediction(version, "invalid", 0.0)
            _json_log(
                log,
                {
                    "event": "prediction",
                    "request_id": request_id,
                    "model_version": version,
                    "latency_ms": 0.0,
                    "status": "invalid_request",
                },
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_features",
                    "message": "request body must be a JSON object of feature values",
                    "errors": [{"field": "<body>", "message": "body is not a JSON object"}],
                },
            ) from None
        if not isinstance(raw, dict):
            serving_metrics.observe_prediction(version, "invalid", 0.0)
            _json_log(
                log,
                {
                    "event": "prediction",
                    "request_id": request_id,
                    "model_version": version,
                    "latency_ms": 0.0,
                    "status": "invalid_request",
                },
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_features",
                    "message": "request body must be a JSON object of feature values",
                    "errors": [{"field": "<body>", "message": "body is not a JSON object"}],
                },
            )

        try:
            features = validate_features(raw, served.spec)
        except FeatureValidationError as exc:
            serving_metrics.observe_prediction(version, "invalid", 0.0)
            _json_log(
                log,
                {
                    "event": "prediction",
                    "request_id": request_id,
                    "model_version": version,
                    "latency_ms": 0.0,
                    "status": "invalid_features",
                    "invalid_fields": [error["field"] for error in exc.errors],
                },
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_features",
                    "message": "feature validation failed",
                    "errors": exc.errors,
                },
            ) from None

        frame = pd.DataFrame([features], columns=list(served.spec.names))
        started = time.perf_counter()
        try:
            probabilities = served.model.predict_proba(frame)
            probability = positive_probability(served.model, probabilities)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            serving_metrics.observe_prediction(version, "error", latency_ms / 1000.0)
            _json_log(
                log,
                {
                    "event": "prediction",
                    "request_id": request_id,
                    "model_version": version,
                    "latency_ms": round(latency_ms, 3),
                    "status": "error",
                    "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
                },
                level=logging.ERROR,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "inference_failed",
                    "message": "the active model failed to score the request; see the serving log",
                },
            ) from None

        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = int(probability >= DEFAULT_DECISION_THRESHOLD)

        # Event logging is explicitly best-effort: a database failure must not
        # change the inference response the caller receives.
        try:
            event = PredictionEvent(
                request_id=request_id,
                timestamp=datetime.now(UTC),
                features=features,
                prediction=prediction,
                probability=probability,
                model_version=version,
                latency_ms=latency_ms,
            )
            store.record(event)
            serving_metrics.events.labels(outcome="logged").inc()
        except Exception as exc:
            serving_metrics.event_failures.inc()
            serving_metrics.events.labels(outcome="failed").inc()
            _json_log(
                log,
                {
                    "event": "prediction_event_write_failed",
                    "request_id": request_id,
                    "model_version": version,
                    "error": redact_secrets(f"{type(exc).__name__}: {exc}"),
                },
                level=logging.ERROR,
            )

        serving_metrics.observe_prediction(version, "success", latency_ms / 1000.0)
        _json_log(
            log,
            {
                "event": "prediction",
                "request_id": request_id,
                "model_version": version,
                "latency_ms": round(latency_ms, 3),
                "status": "success",
                "prediction": prediction,
            },
        )
        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "model_version": version,
                "prediction": prediction,
                "probability": probability,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        serving_metrics.set_active(active_provider.status().pointer)
        return Response(
            content=generate_latest(serving_metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.exception_handler(ModelUnavailable)
    async def _model_unavailable(_request: Request, exc: ModelUnavailable) -> JSONResponse:
        return _http_503(str(exc))

    return app


app = create_app()
