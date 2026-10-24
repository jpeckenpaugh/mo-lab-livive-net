"""Verdict matrix for data, performance, and concept drift (ADR-010/011)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from modelops_lab.config import MonitoringConfig, load_monitoring_config, load_training_config
from modelops_lab.contracts import DriftVerdict, WindowSummary
from modelops_lab.data import assign_splits, generate_synthetic
from modelops_lab.dataset_schema import SYNTHETIC_SCHEMA
from modelops_lab.monitoring import MonitoringResult, analyze_windows
from modelops_lab.train import build_pipeline, predict_probabilities

SPLIT = load_training_config("configs/training.yaml").split
MODEL = load_training_config("configs/training.yaml").model
FEATURES = list(SYNTHETIC_SCHEMA.feature_names)
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Scenario:
    frame: pd.DataFrame
    validation: pd.DataFrame
    scores: np.ndarray


@pytest.fixture(scope="module")
def config() -> MonitoringConfig:
    return load_monitoring_config("configs/monitoring.yaml")


@pytest.fixture(scope="module")
def baseline_model():
    frame = generate_synthetic(2000, 42, "baseline")
    splits = assign_splits(frame["row_id"].to_numpy(), SPLIT)
    train = frame.loc[frame["row_id"].isin(splits["train"])]
    pipeline = build_pipeline(MODEL, SYNTHETIC_SCHEMA)
    with threadpool_limits(limits=1):
        pipeline.fit(train[FEATURES], train["target"])
    return pipeline


def _scenario(regime: str) -> Scenario:
    frame = generate_synthetic(2000, 42, regime)
    splits = assign_splits(frame["row_id"].to_numpy(), SPLIT)
    validation = (
        frame.loc[frame["row_id"].isin(splits["validation"])]
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    return Scenario(frame=frame, validation=validation, scores=np.array([]))


@pytest.fixture(scope="module")
def scenarios(baseline_model) -> dict[str, Scenario]:
    result: dict[str, Scenario] = {}
    for regime in ("baseline", "covariate_shift", "concept_shift"):
        scenario = _scenario(regime)
        scores = predict_probabilities(baseline_model, scenario.validation, FEATURES)
        result[regime] = Scenario(
            frame=scenario.frame, validation=scenario.validation, scores=scores
        )
    return result


def _analyze(
    config: MonitoringConfig,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    reference_auc: float | None,
    labels: list[int] | None,
    scores: list[float] | None,
) -> MonitoringResult:
    return analyze_windows(
        model_version="1",
        reference_frame=reference[FEATURES],
        current_frame=current[FEATURES],
        reference_window=WindowSummary(
            label="reference", source="dataset:test:validation_split", row_count=len(reference)
        ),
        current_window=WindowSummary(
            label="current", source="prediction_events", row_count=len(current)
        ),
        config=config,
        numeric_features=list(SYNTHETIC_SCHEMA.numeric_features),
        categorical_features=list(SYNTHETIC_SCHEMA.categorical_features),
        reference_auc=reference_auc,
        current_labels=labels,
        current_scores=scores,
        now=NOW,
    )


def test_clean_traffic_is_not_drifted_on_every_verdict(
    config: MonitoringConfig, scenarios: dict[str, Scenario]
) -> None:
    baseline = scenarios["baseline"]
    reference_auc = float(roc_auc_score(baseline.validation["target"], baseline.scores))

    result = _analyze(
        config,
        baseline.validation,
        baseline.validation,
        reference_auc=reference_auc,
        labels=baseline.validation["target"].tolist(),
        scores=baseline.scores.tolist(),
    )
    report = result.report

    assert report.data_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.NOT_DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance is not None
    assert report.performance.delta == pytest.approx(0.0, abs=1e-9)


def test_covariate_shift_is_data_drift_not_concept_drift(
    config: MonitoringConfig, scenarios: dict[str, Scenario]
) -> None:
    baseline = scenarios["baseline"]
    shifted = scenarios["covariate_shift"]
    reference_auc = float(roc_auc_score(baseline.validation["target"], baseline.scores))

    result = _analyze(
        config,
        baseline.validation,
        shifted.validation,
        reference_auc=reference_auc,
        labels=shifted.validation["target"].tolist(),
        scores=shifted.scores.tolist(),
    )
    report = result.report

    assert report.data_drift.verdict is DriftVerdict.DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.NOT_EVALUATED
    assert "confounded" in report.concept_drift.detail
    assert report.performance is not None
    assert report.performance.delta < -0.05


def test_concept_shift_is_concept_drift_with_stable_inputs(
    config: MonitoringConfig, scenarios: dict[str, Scenario]
) -> None:
    baseline = scenarios["baseline"]
    concept = scenarios["concept_shift"]
    reference_auc = float(roc_auc_score(baseline.validation["target"], baseline.scores))

    result = _analyze(
        config,
        baseline.validation,
        concept.validation,
        reference_auc=reference_auc,
        labels=concept.validation["target"].tolist(),
        scores=concept.scores.tolist(),
    )
    report = result.report

    assert report.data_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.DRIFTED
    assert report.performance is not None
    assert report.performance.delta < -0.05


def test_no_labels_leaves_performance_and_concept_not_evaluated(
    config: MonitoringConfig, scenarios: dict[str, Scenario]
) -> None:
    baseline = scenarios["baseline"]
    reference_auc = float(roc_auc_score(baseline.validation["target"], baseline.scores))

    result = _analyze(
        config,
        baseline.validation,
        baseline.validation,
        reference_auc=reference_auc,
        labels=None,
        scores=None,
    )
    report = result.report

    assert report.data_drift.verdict is DriftVerdict.NOT_DRIFTED
    assert report.performance_degradation.verdict is DriftVerdict.NOT_EVALUATED
    assert report.concept_drift.verdict is DriftVerdict.NOT_EVALUATED
    assert report.performance is None


def test_small_labeled_window_is_insufficient_data(
    config: MonitoringConfig, scenarios: dict[str, Scenario]
) -> None:
    baseline = scenarios["baseline"]
    reference_auc = float(roc_auc_score(baseline.validation["target"], baseline.scores))
    limit = config.performance_degradation.min_labeled_samples - 1

    result = _analyze(
        config,
        baseline.validation,
        baseline.validation,
        reference_auc=reference_auc,
        labels=baseline.validation["target"].tolist()[:limit],
        scores=baseline.scores.tolist()[:limit],
    )
    report = result.report

    assert report.performance_degradation.verdict is DriftVerdict.INSUFFICIENT_DATA
    assert report.concept_drift.verdict is DriftVerdict.INSUFFICIENT_DATA
    assert report.performance is not None
    assert report.performance.labeled_sample_count == limit
