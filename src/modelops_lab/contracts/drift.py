"""Drift report contract with three independent, non-inferred verdicts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from modelops_lab.contracts.base import ContractModel, VersionedContract


class DriftVerdict(StrEnum):
    """Verdict for a single monitored signal.

    ``NOT_EVALUATED`` means the signal was out of scope for this report;
    ``INSUFFICIENT_DATA`` means it was in scope but sample guards failed. They
    are deliberately distinct from ``NOT_DRIFTED``.
    """

    DRIFTED = "drifted"
    NOT_DRIFTED = "not_drifted"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_EVALUATED = "not_evaluated"


class SignalVerdict(ContractModel):
    """Outcome and supporting context for one signal, kept independent of the others."""

    verdict: DriftVerdict
    detail: str = Field(min_length=1, description="Human-readable evidence for the verdict.")
    method: str | None = Field(
        default=None,
        description="Test or comparison used, for example 'ks' or 'delayed_labels'.",
    )
    score: float | None = Field(
        default=None,
        description="Signal-specific score. Its meaning is defined by method and must not be compared across signals.",
    )


class WindowSummary(ContractModel):
    """Description and sample count for a reference or current window."""

    label: Literal["reference", "current"] = Field(
        description="Whether this window is the baseline or the window under test."
    )
    source: str = Field(min_length=1, description="Origin such as 'training_split' or 'prediction_events'.")
    row_count: int = Field(ge=0, description="Rows observed in this window.")
    start: AwareDatetime | None = Field(
        default=None, description="Window start; null when the window is not time-based."
    )
    end: AwareDatetime | None = Field(
        default=None, description="Window end; null when the window is not time-based."
    )

    @model_validator(mode="after")
    def _time_range_is_ordered(self) -> Self:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("window end must not be earlier than window start")
        return self


class FeatureDriftStatistic(ContractModel):
    """Per-feature distribution comparison against the reference window."""

    feature: str = Field(min_length=1)
    reference_missing_rate: float = Field(ge=0.0, le=1.0)
    current_missing_rate: float = Field(ge=0.0, le=1.0)
    psi: float | None = Field(
        default=None, ge=0.0, description="Population Stability Index, when computable."
    )
    ks_statistic: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Kolmogorov-Smirnov statistic, when computable."
    )
    ks_pvalue: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Kolmogorov-Smirnov p-value, when computable."
    )
    drifted: bool = Field(description="This feature's data-drift flag under the stated thresholds.")


class DriftThresholds(ContractModel):
    """Thresholds used to produce this report, recorded for audit."""

    psi_threshold: float = Field(gt=0.0)
    ks_pvalue_threshold: float = Field(gt=0.0, le=1.0)
    missing_rate_delta_threshold: float = Field(ge=0.0, le=1.0)
    aggregate_feature_share_threshold: float = Field(gt=0.0, le=1.0)
    min_sample_size: int = Field(ge=1)
    performance_degradation_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Absolute metric drop counted as degradation; null when performance was not evaluated.",
    )


class PerformanceSnapshot(ContractModel):
    """Delayed-label performance comparison for the current window."""

    metric: str = Field(min_length=1, description="Metric name, for example 'roc_auc'.")
    reference_value: float
    current_value: float
    delta: float = Field(description="current_value minus reference_value.")
    labeled_sample_count: int = Field(ge=0, description="Current-window rows with ground-truth labels.")


class DriftReport(VersionedContract):
    """Reference-versus-current monitoring report.

    Data drift, concept drift, and observed performance degradation are three
    separate fields with independent verdicts. A drifted feature distribution
    never forces a performance verdict, and missing labels never imply that
    concept drift was absent.
    """

    report_id: str = Field(min_length=1)
    generated_at: AwareDatetime
    model_version: str = Field(min_length=1, description="Model version monitored in the current window.")
    reference_window: WindowSummary
    current_window: WindowSummary
    feature_statistics: list[FeatureDriftStatistic] = Field(default_factory=list)
    thresholds: DriftThresholds
    data_drift: SignalVerdict
    concept_drift: SignalVerdict
    performance_degradation: SignalVerdict
    performance: PerformanceSnapshot | None = Field(
        default=None,
        description="Delayed-label comparison; present only when performance was evaluated.",
    )

    @field_validator("feature_statistics")
    @classmethod
    def _features_are_unique(
        cls, value: list[FeatureDriftStatistic]
    ) -> list[FeatureDriftStatistic]:
        names = [statistic.feature for statistic in value]
        if len(set(names)) != len(names):
            raise ValueError("feature_statistics must contain at most one entry per feature")
        return value

    @model_validator(mode="after")
    def _verdicts_have_supporting_evidence(self) -> Self:
        evaluated = {DriftVerdict.DRIFTED, DriftVerdict.NOT_DRIFTED}

        if self.current_window.row_count < self.thresholds.min_sample_size:
            if self.data_drift.verdict in evaluated:
                raise ValueError(
                    "data_drift must be insufficient_data when the current window is smaller "
                    f"than min_sample_size={self.thresholds.min_sample_size}"
                )
        if not self.feature_statistics and self.data_drift.verdict in evaluated:
            raise ValueError("data_drift cannot be evaluated without feature_statistics")

        if self.performance_degradation.verdict in evaluated and self.performance is None:
            raise ValueError(
                "an evaluated performance_degradation verdict requires a performance snapshot"
            )
        return self
