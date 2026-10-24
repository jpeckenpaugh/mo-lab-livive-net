"""In-repo drift statistics, three independent verdicts, and report generation.

The statistical core is deliberately ours (ADR-005); no Evidently dependency
is involved. Two windows are compared:

* the **reference** is a split of the active model's own dataset artifact
  (``validation_split`` by default), capped deterministically at
  ``reference.window_size`` rows;
* the **current** is the most recent ``current.window_size`` prediction events
  for that model version, in chronological order.

Three verdicts are computed independently:

``data_drift``
    Per-feature KS (numeric), PSI (numeric deciles and categorical
    frequencies), and missing-rate delta, aggregated by the share of drifted
    features. A current window below ``min_sample_size`` is
    ``insufficient_data``, never ``not_drifted``.
``performance_degradation``
    Requires delayed labels. Compares the same model's ROC-AUC on the current
    labeled window against its ROC-AUC on the reference validation split.
``concept_drift``
    Explicit rule (see :func:`evaluate_concept_drift`). Performance must be
    degraded **and** data drift must not be flagged; otherwise the outcome is
    ``not_evaluated``, ``insufficient_data``, or ``not_drifted``. This is what
    keeps covariate shift from being mislabelled as concept drift.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from modelops_lab.config import MonitoringConfig
from modelops_lab.contracts import (
    DriftReport,
    DriftThresholds,
    DriftVerdict,
    FeatureDriftStatistic,
    PerformanceSnapshot,
    SignalVerdict,
    WindowSummary,
)
from modelops_lab.data import (
    DEFAULT_DATASET_ROOT,
    LoadedDataset,
    split_frame,
)
from modelops_lab.db import PredictionEventRow, PredictionLabelRow, create_engine_for, utc
from modelops_lab.train import predict_probabilities

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
NUMERIC = "numeric"
CATEGORICAL = "categorical"
DEFAULT_REFERENCE_SOURCE = "validation_split"
# Smoothing keeps a zero-proportion bin from turning the PSI logarithm into
# +/-infinity. It is applied to both windows so an identical empty bin is 0.
ZERO_BIN_EPSILON = 1e-6
PERFORMANCE_METRIC = "roc_auc"
_SECRET_IN_URI = re.compile(r"(://[^:/@\s]+:)([^@\s]+)(@)")
_OPERATOR_ERROR_LIMIT = 300


class MonitoringError(RuntimeError):
    """Raised when monitoring cannot read its inputs or write its outputs."""


def operator_error(exc: BaseException) -> str:
    """Describe a failure in one operator-facing line, without secrets or a traceback."""
    message = " ".join(str(exc).split())
    message = _SECRET_IN_URI.sub(r"\1***\3", message)
    if len(message) > _OPERATOR_ERROR_LIMIT:
        message = message[: _OPERATOR_ERROR_LIMIT - 3] + "..."
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def _numeric(values: Any) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    return series.to_numpy(dtype=float)


def missing_rate(values: Any) -> float:
    """Fraction of missing (null/NaN/non-numeric) values, in ``[0, 1]``."""
    series = pd.Series(values)
    if len(series) == 0:
        return 0.0
    return float(series.isna().sum()) / len(series)


def population_stability_index(reference: Any, current: Any, *, bins: int = 10) -> float | None:
    """PSI with reference-decile bins and zero-bin smoothing.

    Bins are the reference deciles (plus infinite tails), so the reference is
    approximately uniform across bins. Both proportions are floored at
    ``ZERO_BIN_EPSILON`` before the logarithm; an empty bin in both windows
    contributes zero. Returns ``None`` when either window has no numeric
    values.
    """
    ref = _numeric(reference)
    cur = _numeric(current)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, max(2, bins) + 1)
    internal = np.unique(np.quantile(ref, quantiles))
    internal = internal[(internal > ref.min()) & (internal < ref.max())]
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_share = ref_counts / ref_counts.sum()
    cur_share = cur_counts / cur_counts.sum()
    ref_share = np.clip(ref_share, ZERO_BIN_EPSILON, None)
    cur_share = np.clip(cur_share, ZERO_BIN_EPSILON, None)
    value = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return max(0.0, value)


def categorical_population_stability_index(reference: Any, current: Any) -> float | None:
    """PSI over category frequencies; unseen categories are included.

    A category observed only in the current window gets a smoothed reference
    share, so it contributes a positive divergence instead of being silently
    dropped. Missing values are excluded here and reported separately.
    """
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if len(ref) == 0 or len(cur) == 0:
        return None
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()
    ref_share = np.array([ref_counts.get(category, 0) / len(ref) for category in categories])
    cur_share = np.array([cur_counts.get(category, 0) / len(cur) for category in categories])
    ref_share = np.clip(ref_share, ZERO_BIN_EPSILON, None)
    cur_share = np.clip(cur_share, ZERO_BIN_EPSILON, None)
    value = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return max(0.0, value)


def ks_two_sample(reference: Any, current: Any) -> tuple[float, float] | None:
    """Two-sample Kolmogorov-Smirnov statistic and p-value for numeric data."""
    ref = _numeric(reference)
    cur = _numeric(current)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return None
    result = ks_2samp(ref, cur, method="auto", nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Feature and window evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureComparison:
    """A per-feature statistic plus the concrete rules that fired."""

    statistic: FeatureDriftStatistic
    rules: tuple[str, ...]

    @property
    def drifted(self) -> bool:
        return self.statistic.drifted


@dataclass(frozen=True)
class DataDriftResult:
    comparisons: tuple[FeatureComparison, ...]
    drifted_count: int
    feature_count: int
    share: float
    verdict: SignalVerdict


def compare_feature(
    name: str,
    kind: Literal["numeric", "categorical"],
    reference: Any,
    current: Any,
    *,
    psi_threshold: float,
    ks_pvalue_threshold: float,
    missing_rate_delta_threshold: float,
) -> FeatureComparison:
    """Compare one feature and record exactly which rules fired."""
    ref_values = pd.Series(reference)
    cur_values = pd.Series(current)
    ref_missing = missing_rate(ref_values)
    cur_missing = missing_rate(cur_values)
    missing_delta = abs(cur_missing - ref_missing)
    rules: list[str] = []

    if kind == NUMERIC:
        psi = population_stability_index(ref_values, cur_values)
        ks = ks_two_sample(ref_values, cur_values)
        ks_statistic = ks[0] if ks is not None else None
        ks_pvalue = ks[1] if ks is not None else None
        if psi is not None and psi >= psi_threshold:
            rules.append(f"psi {psi:.4f} >= {psi_threshold:g}")
        if ks_pvalue is not None and ks_pvalue < ks_pvalue_threshold:
            rules.append(f"ks_pvalue {ks_pvalue:.6g} < {ks_pvalue_threshold:g}")
    else:
        psi = categorical_population_stability_index(ref_values, cur_values)
        ks_statistic = None
        ks_pvalue = None
        if psi is not None and psi >= psi_threshold:
            rules.append(f"psi {psi:.4f} >= {psi_threshold:g}")

    if missing_delta >= missing_rate_delta_threshold:
        rules.append(f"missing_rate_delta {missing_delta:.4f} >= {missing_rate_delta_threshold:g}")

    return FeatureComparison(
        statistic=FeatureDriftStatistic(
            feature=name,
            reference_missing_rate=ref_missing,
            current_missing_rate=cur_missing,
            psi=psi,
            ks_statistic=ks_statistic,
            ks_pvalue=ks_pvalue,
            drifted=bool(rules),
        ),
        rules=tuple(rules),
    )


def evaluate_data_drift(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    config: MonitoringConfig,
) -> DataDriftResult:
    """Compute per-feature statistics and the aggregate data-drift verdict."""
    thresholds = config.feature_drift
    comparisons = tuple(
        compare_feature(
            name,
            kind,
            reference_frame[name].to_numpy() if name in reference_frame else [],
            current_frame[name].to_numpy() if name in current_frame else [],
            psi_threshold=thresholds.psi_threshold,
            ks_pvalue_threshold=thresholds.ks_pvalue_threshold,
            missing_rate_delta_threshold=thresholds.missing_rate_delta_threshold,
        )
        for kind, names in ((NUMERIC, numeric_features), (CATEGORICAL, categorical_features))
        for name in names
    )
    feature_count = len(comparisons)
    drifted = [comparison for comparison in comparisons if comparison.drifted]
    share = len(drifted) / feature_count if feature_count else 0.0

    if len(current_frame) < config.min_sample_size:
        verdict = SignalVerdict(
            verdict=DriftVerdict.INSUFFICIENT_DATA,
            method="psi+ks+missing_rate",
            score=share,
            detail=(
                f"current window has {len(current_frame)} rows, below "
                f"min_sample_size={config.min_sample_size}; data drift is not asserted"
            ),
        )
    elif len(reference_frame) < config.min_sample_size:
        verdict = SignalVerdict(
            verdict=DriftVerdict.INSUFFICIENT_DATA,
            method="psi+ks+missing_rate",
            score=share,
            detail=(
                f"reference window has {len(reference_frame)} rows, below "
                f"min_sample_size={config.min_sample_size}; data drift is not asserted"
            ),
        )
    elif feature_count == 0:
        verdict = SignalVerdict(
            verdict=DriftVerdict.INSUFFICIENT_DATA,
            method="psi+ks+missing_rate",
            score=0.0,
            detail="no declared features were available to compare",
        )
    else:
        fired = "; ".join(
            f"{comparison.statistic.feature} [" + ", ".join(comparison.rules) + "]"
            for comparison in drifted
        )
        detail = (
            f"{len(drifted)}/{feature_count} features drifted "
            f"(share {share:.3f}, threshold {config.aggregate_feature_share_threshold:g})"
        )
        if fired:
            detail += f": {fired}"
        else:
            detail += "; no feature exceeded a per-feature threshold"
        verdict = SignalVerdict(
            verdict=(
                DriftVerdict.DRIFTED
                if share >= config.aggregate_feature_share_threshold
                else DriftVerdict.NOT_DRIFTED
            ),
            method="psi+ks+missing_rate",
            score=share,
            detail=detail,
        )
    return DataDriftResult(
        comparisons=comparisons,
        drifted_count=len(drifted),
        feature_count=feature_count,
        share=share,
        verdict=verdict,
    )


@dataclass(frozen=True)
class PerformanceEvaluation:
    verdict: SignalVerdict
    snapshot: PerformanceSnapshot | None
    current_auc: float | None


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    values = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=float)
    if values.size == 0 or values.size != probabilities.size:
        return None
    if np.unique(values).size < 2:
        return None
    return float(roc_auc_score(values, probabilities))


def evaluate_performance(
    *,
    reference_value: float | None,
    current_labels: Sequence[int] | None,
    current_scores: Sequence[float] | None,
    min_labeled_samples: int,
    max_drop: float,
    reference_error: str | None = None,
) -> PerformanceEvaluation:
    """Compare labeled current-window ROC-AUC with the reference value.

    ``reference_error`` explains why the reference could not be computed. It is
    reported as ``not_evaluated`` rather than raised: data drift remains a
    separate, valid signal even when the model cannot be loaded or scored.
    """
    labels = list(current_labels or [])
    scores = list(current_scores or [])
    if not labels:
        return PerformanceEvaluation(
            verdict=SignalVerdict(
                verdict=DriftVerdict.NOT_EVALUATED,
                method="delayed_labels",
                detail=(
                    "no delayed labels exist for the current window; performance was not measured"
                ),
            ),
            snapshot=None,
            current_auc=None,
        )
    if reference_value is None:
        reason = reference_error or (
            "no reference performance value is available for this model version"
        )
        return PerformanceEvaluation(
            verdict=SignalVerdict(
                verdict=DriftVerdict.NOT_EVALUATED,
                method="delayed_labels",
                detail=(
                    "performance not evaluated because the reference ROC-AUC is unavailable: "
                    f"{reason}"
                ),
            ),
            snapshot=None,
            current_auc=None,
        )
    current_auc = _binary_auc(labels, scores)
    if current_auc is None:
        return PerformanceEvaluation(
            verdict=SignalVerdict(
                verdict=DriftVerdict.INSUFFICIENT_DATA,
                method="delayed_labels",
                score=float(len(labels)),
                detail=(
                    f"{len(labels)} labeled rows but only one observed class; ROC-AUC is undefined"
                ),
            ),
            snapshot=None,
            current_auc=None,
        )

    snapshot = PerformanceSnapshot(
        metric=PERFORMANCE_METRIC,
        reference_value=reference_value,
        current_value=current_auc,
        delta=current_auc - reference_value,
        labeled_sample_count=len(labels),
    )
    drop = reference_value - current_auc
    if len(labels) < min_labeled_samples:
        verdict = SignalVerdict(
            verdict=DriftVerdict.INSUFFICIENT_DATA,
            method="delayed_labels",
            score=drop,
            detail=(
                f"{len(labels)} labeled rows, below min_labeled_samples={min_labeled_samples}; "
                f"observed ROC-AUC {current_auc:.4f} vs reference {reference_value:.4f} "
                "is not asserted as degradation"
            ),
        )
    elif drop > max_drop:
        verdict = SignalVerdict(
            verdict=DriftVerdict.DRIFTED,
            method="delayed_labels",
            score=drop,
            detail=(
                f"ROC-AUC dropped {drop:.4f} from {reference_value:.4f} to "
                f"{current_auc:.4f}, above max_drop={max_drop:g} "
                f"on {len(labels)} labeled rows"
            ),
        )
    else:
        verdict = SignalVerdict(
            verdict=DriftVerdict.NOT_DRIFTED,
            method="delayed_labels",
            score=drop,
            detail=(
                f"ROC-AUC changed by {drop:+.4f} (reference {reference_value:.4f}, "
                f"current {current_auc:.4f}), within max_drop={max_drop:g} "
                f"on {len(labels)} labeled rows"
            ),
        )
    return PerformanceEvaluation(verdict=verdict, snapshot=snapshot, current_auc=current_auc)


def evaluate_concept_drift(
    *,
    data_verdict: DriftVerdict,
    performance_verdict: DriftVerdict,
    labeled_sample_count: int,
    min_labeled_samples: int,
    performance_detail: str | None = None,
) -> SignalVerdict:
    """Apply the explicit concept-drift rule.

    Concept drift is asserted only when performance is measurably degraded
    while the input distribution is stable. If data drift is also flagged the
    cause is confounded, so the honest verdict is ``not_evaluated``.
    """
    if labeled_sample_count == 0:
        return SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED,
            method="performance_plus_stable_inputs",
            detail="no delayed labels; concept drift cannot be evaluated",
        )
    if labeled_sample_count < min_labeled_samples:
        return SignalVerdict(
            verdict=DriftVerdict.INSUFFICIENT_DATA,
            method="performance_plus_stable_inputs",
            score=float(labeled_sample_count),
            detail=(
                f"{labeled_sample_count} labeled rows, below "
                f"min_labeled_samples={min_labeled_samples}"
            ),
        )
    if performance_verdict is DriftVerdict.DRIFTED:
        if data_verdict is DriftVerdict.DRIFTED:
            return SignalVerdict(
                verdict=DriftVerdict.NOT_EVALUATED,
                method="performance_plus_stable_inputs",
                detail=(
                    "performance is degraded but data drift is also flagged; the cause is "
                    "confounded by data drift and concept drift is not asserted"
                ),
            )
        if data_verdict is DriftVerdict.NOT_DRIFTED:
            return SignalVerdict(
                verdict=DriftVerdict.DRIFTED,
                method="performance_plus_stable_inputs",
                detail=(
                    "performance is degraded while the input distribution is stable; "
                    "this is evidence of a changed input-output relationship"
                ),
            )
        return SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED,
            method="performance_plus_stable_inputs",
            detail=(
                "performance is degraded but data drift could not be evaluated; "
                "concept drift is not asserted"
            ),
        )
    if performance_verdict is DriftVerdict.NOT_DRIFTED:
        return SignalVerdict(
            verdict=DriftVerdict.NOT_DRIFTED,
            method="performance_plus_stable_inputs",
            detail="performance is not degraded; no concept drift is asserted",
        )
    detail = "performance could not be evaluated; concept drift is not asserted"
    if performance_detail:
        detail = (
            "performance could not be evaluated; concept drift is not asserted: "
            f"{performance_detail}"
        )
    return SignalVerdict(
        verdict=DriftVerdict.NOT_EVALUATED,
        method="performance_plus_stable_inputs",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Report composition and rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringResult:
    """Contract report plus the audit detail a reader needs from the HTML."""

    report: DriftReport
    feature_rules: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reference_auc: float | None = None
    current_auc: float | None = None


def _report_id(now: datetime, model_version: str) -> str:
    safe_version = "".join(character if character.isalnum() else "-" for character in model_version)
    return (
        f"{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-v{safe_version}-{uuid.uuid4().hex[:8]}"
    )


def analyze_windows(
    *,
    model_version: str,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    reference_window: WindowSummary,
    current_window: WindowSummary,
    config: MonitoringConfig,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    reference_auc: float | None = None,
    reference_error: str | None = None,
    current_labels: Sequence[int] | None = None,
    current_scores: Sequence[float] | None = None,
    now: datetime | None = None,
    report_id: str | None = None,
) -> MonitoringResult:
    """Build a :class:`DriftReport` from two frames and, optionally, labels."""
    if len(current_labels or []) != len(current_scores or []):
        raise MonitoringError("current labels and scores must have the same length")
    if reference_window.row_count != len(reference_frame):
        raise MonitoringError(
            "reference window row_count does not match the supplied reference frame"
        )
    if current_window.row_count != len(current_frame):
        raise MonitoringError("current window row_count does not match the supplied current frame")
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    data = evaluate_data_drift(
        reference_frame=reference_frame,
        current_frame=current_frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )
    performance = evaluate_performance(
        reference_value=reference_auc,
        current_labels=current_labels,
        current_scores=current_scores,
        min_labeled_samples=config.performance_degradation.min_labeled_samples,
        max_drop=config.performance_degradation.max_drop,
        reference_error=reference_error,
    )
    labeled_count = len(current_labels or [])
    concept = evaluate_concept_drift(
        data_verdict=data.verdict.verdict,
        performance_verdict=performance.verdict.verdict,
        labeled_sample_count=labeled_count,
        min_labeled_samples=config.performance_degradation.min_labeled_samples,
        performance_detail=(
            performance.verdict.detail
            if performance.verdict.verdict is DriftVerdict.NOT_EVALUATED
            else None
        ),
    )
    thresholds = DriftThresholds(
        psi_threshold=config.feature_drift.psi_threshold,
        ks_pvalue_threshold=config.feature_drift.ks_pvalue_threshold,
        missing_rate_delta_threshold=config.feature_drift.missing_rate_delta_threshold,
        aggregate_feature_share_threshold=config.aggregate_feature_share_threshold,
        min_sample_size=config.min_sample_size,
        performance_degradation_threshold=(
            None
            if performance.verdict.verdict is DriftVerdict.NOT_EVALUATED
            else config.performance_degradation.max_drop
        ),
    )
    report = DriftReport(
        schema_version="1.1.0",
        report_id=report_id or _report_id(generated_at, model_version),
        generated_at=generated_at,
        model_version=model_version,
        reference_window=reference_window,
        current_window=current_window,
        feature_statistics=[comparison.statistic for comparison in data.comparisons],
        thresholds=thresholds,
        data_drift=data.verdict,
        concept_drift=concept,
        performance_degradation=performance.verdict,
        performance=performance.snapshot,
    )
    return MonitoringResult(
        report=report,
        feature_rules={
            comparison.statistic.feature: comparison.rules for comparison in data.comparisons
        },
        reference_auc=reference_auc,
        current_auc=performance.current_auc,
    )


def _render(template_name: str, context: dict[str, Any]) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(("html", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(template_name)
    return template.render(**context)


def render_drift_html(result: MonitoringResult) -> str:
    """Render the self-contained, JavaScript-free drift report."""
    report = result.report
    payload = report.model_dump(mode="json")
    return _render(
        "drift_report.html.j2",
        {
            "report": payload,
            "feature_rules": result.feature_rules,
            "reference_auc": result.reference_auc,
            "current_auc": result.current_auc,
        },
    )


def write_drift_report(result: MonitoringResult, output_dir: Path) -> tuple[Path, Path]:
    """Write ``<report_id>.json`` and ``<report_id>.html``; return both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = result.report
    json_path = output_dir / f"drift-{report.report_id}.json"
    html_path = output_dir / f"drift-{report.report_id}.html"
    try:
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        html_path.write_text(render_drift_html(result), encoding="utf-8")
    except OSError as exc:
        raise MonitoringError(f"cannot write drift report to {output_dir}: {exc}") from exc
    return json_path, html_path


# ---------------------------------------------------------------------------
# Infrastructure-backed collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredEvent:
    event_id: int
    timestamp: datetime
    features: dict[str, Any]
    prediction: int
    probability: float
    model_version: str


class SqlMonitoringRepository:
    """Read prediction events and their delayed labels for one window."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._engine = create_engine_for(database_url)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)

    def _session(self) -> Session:
        return self._session_factory()

    def recent_events(self, model_version: str, limit: int) -> list[StoredEvent]:
        with self._session() as session:
            rows = (
                session.execute(
                    select(PredictionEventRow)
                    .where(PredictionEventRow.model_version == model_version)
                    .order_by(
                        PredictionEventRow.timestamp.desc(),
                        PredictionEventRow.event_id.desc(),
                    )
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            events = [
                StoredEvent(
                    event_id=int(row.event_id),
                    timestamp=utc(row.timestamp),
                    features=dict(row.features),
                    prediction=int(row.prediction),
                    probability=float(row.probability),
                    model_version=row.model_version,
                )
                for row in rows
            ]
            events.reverse()
            return events

    def labels_for(self, event_ids: Sequence[int]) -> dict[int, int]:
        if not event_ids:
            return {}
        with self._session() as session:
            rows = session.execute(
                select(PredictionLabelRow).where(PredictionLabelRow.event_id.in_(list(event_ids)))
            ).scalars()
            return {int(row.event_id): int(row.label) for row in rows}

    def event_counts(self) -> tuple[int, dict[str, int]]:
        with self._session() as session:
            total = int(
                session.execute(select(func.count()).select_from(PredictionEventRow)).scalar_one()
            )
            rows = session.execute(
                select(PredictionEventRow.model_version, func.count())
                .group_by(PredictionEventRow.model_version)
                .order_by(PredictionEventRow.model_version)
            ).all()
            return total, {str(version): int(count) for version, count in rows}


@dataclass(frozen=True)
class MonitoringRun:
    """Outcome of collecting windows and writing a report."""

    result: MonitoringResult
    model_version: str
    reference_source: str
    current_window_size: int
    json_path: Path
    html_path: Path


def _current_frame(events: Sequence[StoredEvent], feature_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        row: dict[str, Any] = {}
        for name in feature_names:
            value = event.features.get(name)
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=list(feature_names))


def _labeled_scores(
    events: Sequence[StoredEvent], labels: dict[int, int]
) -> tuple[list[int], list[float]]:
    labeled: list[int] = []
    scores: list[float] = []
    for event in events:
        label = labels.get(event.event_id)
        if label is None:
            continue
        labeled.append(int(label))
        scores.append(float(event.probability))
    return labeled, scores


def run_monitoring(
    *,
    config: MonitoringConfig,
    database_url: str,
    dataset: LoadedDataset,
    model_version: str,
    model_loader: Any,
    window_size: int | None = None,
    output_dir: Path | None = None,
    now: datetime | None = None,
) -> MonitoringRun:
    """Collect the windows, score the reference, and write both report files.

    ``model_loader`` is a callable ``(version) -> fitted pipeline``. It is
    injected so the CLI can use MLflow and unit tests can use a plain pipeline.
    """
    chosen_window = window_size or config.current.window_size
    if chosen_window < config.min_sample_size:
        raise MonitoringError(
            f"--window-size {chosen_window} is below min_sample_size={config.min_sample_size}"
        )
    numeric_features = list(dataset.schema.numeric_features)
    categorical_features = list(dataset.schema.categorical_features)
    feature_names = list(dataset.schema.feature_names)

    split_name = "validation" if config.reference.source == "validation_split" else "train"
    full_reference = split_frame(dataset, split_name)
    reference_frame = full_reference.head(config.reference.window_size).reset_index(drop=True)

    reference_auc: float | None = None
    reference_error: str | None = None
    try:
        model = model_loader(model_version)
        reference_scores = predict_probabilities(model, full_reference, feature_names)
        reference_auc = _binary_auc(
            full_reference["target"].to_numpy(dtype=np.int64), reference_scores
        )
        if reference_auc is None:
            reference_error = (
                "the reference validation split contains a single observed class, "
                "so ROC-AUC is undefined"
            )
    except Exception as exc:
        reference_error = operator_error(exc)

    repository = SqlMonitoringRepository(database_url)
    events = repository.recent_events(model_version, chosen_window)
    current_frame = _current_frame(events, feature_names)
    labels = repository.labels_for([event.event_id for event in events])
    current_labels, current_scores = _labeled_scores(events, labels)

    current_start = events[0].timestamp if events else None
    current_end = events[-1].timestamp if events else None
    reference_window = WindowSummary(
        label="reference",
        source=f"dataset:{dataset.manifest.dataset_version}:{config.reference.source}",
        row_count=len(reference_frame),
    )
    current_window = WindowSummary(
        label="current",
        source=f"prediction_events:model_version={model_version}",
        row_count=len(current_frame),
        start=current_start,
        end=current_end,
    )
    result = analyze_windows(
        model_version=model_version,
        reference_frame=reference_frame,
        current_frame=current_frame,
        reference_window=reference_window,
        current_window=current_window,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        reference_auc=reference_auc,
        reference_error=reference_error,
        current_labels=current_labels,
        current_scores=current_scores,
        now=now,
    )
    target_dir = output_dir or Path(config.report_output_dir)
    json_path, html_path = write_drift_report(result, target_dir)
    return MonitoringRun(
        result=result,
        model_version=model_version,
        reference_source=config.reference.source,
        current_window_size=chosen_window,
        json_path=json_path,
        html_path=html_path,
    )


def default_dataset_dir(manifest_version: str) -> Path:
    """Location of a prepared dataset artifact for a manifest version."""
    return DEFAULT_DATASET_ROOT / manifest_version


def describe_result(run: MonitoringRun) -> list[str]:
    """Human-readable verdict summary for the CLI."""
    report = run.result.report
    rows = [
        f"model_version: {run.model_version}",
        f"reference: {report.reference_window.source} ({report.reference_window.row_count} rows)",
        f"current: {report.current_window.source} ({report.current_window.row_count} rows, "
        f"window {run.current_window_size})",
    ]
    if report.performance is not None:
        snapshot = report.performance
        rows.append(
            f"labeled_rows: {snapshot.labeled_sample_count}  "
            f"{snapshot.metric}: reference {snapshot.reference_value:.4f} -> "
            f"current {snapshot.current_value:.4f} (delta {snapshot.delta:+.4f})"
        )
    else:
        rows.append("labeled_rows: 0")
    rows.extend(
        [
            f"data_drift: {report.data_drift.verdict.value}  {report.data_drift.detail}",
            f"performance_degradation: {report.performance_degradation.verdict.value}  "
            f"{report.performance_degradation.detail}",
            f"concept_drift: {report.concept_drift.verdict.value}  {report.concept_drift.detail}",
            f"report_json: {run.json_path}",
            f"report_html: {run.html_path}",
        ]
    )
    return rows


def load_report(path: Path) -> DriftReport:
    """Read and validate a drift report JSON artifact."""
    try:
        return DriftReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MonitoringError(f"{path}: not a valid DriftReport: {exc}") from exc
