"""Docker-free tests for the in-repo drift statistics (ADR-005)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelops_lab.config import MonitoringConfig, load_monitoring_config
from modelops_lab.contracts import DriftVerdict
from modelops_lab.monitoring import (
    categorical_population_stability_index,
    compare_feature,
    evaluate_data_drift,
    missing_rate,
    population_stability_index,
)

ROOT_CONFIG = "configs/monitoring.yaml"


@pytest.fixture(scope="module")
def config() -> MonitoringConfig:
    return load_monitoring_config(ROOT_CONFIG)


def test_identical_distributions_are_not_drifted(config: MonitoringConfig) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(50.0, 5.0, size=500)

    psi = population_stability_index(values, values)
    comparison = compare_feature(
        "age",
        "numeric",
        values,
        values,
        psi_threshold=config.feature_drift.psi_threshold,
        ks_pvalue_threshold=config.feature_drift.ks_pvalue_threshold,
        missing_rate_delta_threshold=config.feature_drift.missing_rate_delta_threshold,
    )

    assert psi == pytest.approx(0.0, abs=1e-12)
    assert comparison.drifted is False
    assert comparison.rules == ()
    assert comparison.statistic.ks_statistic == pytest.approx(0.0)
    assert comparison.statistic.ks_pvalue == pytest.approx(1.0)


def test_numeric_mean_shift_is_flagged(config: MonitoringConfig) -> None:
    rng = np.random.default_rng(11)
    reference = rng.normal(50.0, 4.0, size=500)
    current = rng.normal(60.0, 4.0, size=500)

    comparison = compare_feature(
        "age",
        "numeric",
        reference,
        current,
        psi_threshold=config.feature_drift.psi_threshold,
        ks_pvalue_threshold=config.feature_drift.ks_pvalue_threshold,
        missing_rate_delta_threshold=config.feature_drift.missing_rate_delta_threshold,
    )

    assert comparison.drifted is True
    assert comparison.statistic.psi is not None and comparison.statistic.psi > 0.2
    assert comparison.statistic.ks_pvalue is not None
    assert comparison.statistic.ks_pvalue < 0.05
    assert any(rule.startswith("psi ") for rule in comparison.rules)
    assert any(rule.startswith("ks_pvalue ") for rule in comparison.rules)


def test_categorical_proportion_shift_is_flagged(config: MonitoringConfig) -> None:
    reference = ["never"] * 500 + ["former"] * 300 + ["current"] * 200
    current = ["never"] * 100 + ["former"] * 300 + ["current"] * 600

    comparison = compare_feature(
        "smoker",
        "categorical",
        reference,
        current,
        psi_threshold=config.feature_drift.psi_threshold,
        ks_pvalue_threshold=config.feature_drift.ks_pvalue_threshold,
        missing_rate_delta_threshold=config.feature_drift.missing_rate_delta_threshold,
    )

    assert comparison.drifted is True
    assert comparison.statistic.psi is not None and comparison.statistic.psi > 0.2
    assert comparison.statistic.ks_statistic is None
    assert comparison.statistic.ks_pvalue is None


def test_unseen_category_is_included_in_the_psi(config: MonitoringConfig) -> None:
    reference = ["a"] * 500 + ["b"] * 500
    current = ["a"] * 500 + ["brand_new"] * 500

    psi = categorical_population_stability_index(reference, current)
    comparison = compare_feature(
        "sex",
        "categorical",
        reference,
        current,
        psi_threshold=config.feature_drift.psi_threshold,
        ks_pvalue_threshold=config.feature_drift.ks_pvalue_threshold,
        missing_rate_delta_threshold=config.feature_drift.missing_rate_delta_threshold,
    )

    assert psi is not None and psi > 0.2
    assert comparison.drifted is True


def test_missing_rate_drift_is_flagged_without_value_drift(
    config: MonitoringConfig,
) -> None:
    reference = np.linspace(0.0, 100.0, 200)
    current = np.concatenate([np.linspace(0.0, 100.0, 160), [np.nan] * 40])

    assert missing_rate(current) == pytest.approx(0.2)
    comparison = compare_feature(
        "bmi",
        "numeric",
        reference,
        current,
        psi_threshold=config.feature_drift.psi_threshold,
        ks_pvalue_threshold=config.feature_drift.ks_pvalue_threshold,
        missing_rate_delta_threshold=config.feature_drift.missing_rate_delta_threshold,
    )

    assert comparison.statistic.psi == pytest.approx(0.0, abs=1e-9)
    assert comparison.drifted is True
    assert comparison.rules == ("missing_rate_delta 0.2000 >= 0.05",)


def test_aggregate_share_controls_the_data_verdict(config: MonitoringConfig) -> None:
    rng = np.random.default_rng(3)
    reference = pd.DataFrame(
        {"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300), "c": rng.normal(0, 1, 300)}
    )
    current = pd.DataFrame(
        {"a": rng.normal(3, 1, 300), "b": rng.normal(0, 1, 300), "c": rng.normal(0, 1, 300)}
    )

    result = evaluate_data_drift(
        reference_frame=reference,
        current_frame=current,
        numeric_features=["a", "b", "c"],
        categorical_features=[],
        config=config,
    )

    assert result.drifted_count == 1
    assert result.share == pytest.approx(1 / 3)
    assert result.verdict.verdict is DriftVerdict.DRIFTED
    assert "a [psi" in result.verdict.detail


def test_below_min_sample_size_is_insufficient_never_not_drifted(
    config: MonitoringConfig,
) -> None:
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 300)})
    current = pd.DataFrame({"a": rng.normal(9, 1, config.min_sample_size - 1)})

    result = evaluate_data_drift(
        reference_frame=reference,
        current_frame=current,
        numeric_features=["a"],
        categorical_features=[],
        config=config,
    )

    assert result.verdict.verdict is DriftVerdict.INSUFFICIENT_DATA
    assert "below min_sample_size" in result.verdict.detail
