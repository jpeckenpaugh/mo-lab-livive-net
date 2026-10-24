"""Pure training, leakage, metric, and determinism tests (no MLflow, no Docker)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from modelops_lab.config import TrainingConfig
from modelops_lab.data import LoadedDataset, split_frame
from modelops_lab.dataset_schema import SYNTHETIC_NUMERIC_FEATURES
from modelops_lab.train import (
    Metrics,
    TrainingError,
    classifier_coefficients,
    evaluate_predictions,
    fit_and_evaluate,
    train_on_dataset,
)


def test_preprocessing_is_fit_on_train_only(
    training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    train = split_frame(loaded_dataset, "train")
    validation = split_frame(loaded_dataset, "validation").copy()
    # Outliers far outside the declared ranges would move a scaler that leaked.
    for name in SYNTHETIC_NUMERIC_FEATURES:
        validation[name] = 1.0e6

    pipeline, _ = fit_and_evaluate(training_config.model, loaded_dataset.schema, train, validation)

    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    train_means = train.loc[:, list(SYNTHETIC_NUMERIC_FEATURES)].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_means, rtol=0.0, atol=1e-12)

    combined = pd.concat([train, validation], ignore_index=True)
    combined_means = combined.loc[:, list(SYNTHETIC_NUMERIC_FEATURES)].mean().to_numpy()
    assert not np.allclose(scaler.mean_, combined_means)


def test_metric_correctness_on_hand_checked_case() -> None:
    metrics = evaluate_predictions(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5
    )

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.log_loss == pytest.approx(
        -(math.log(0.9) + math.log(0.8) + math.log(0.8) + math.log(0.9)) / 4
    )
    assert metrics.confusion_dict() == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }
    assert (metrics.support, metrics.predicted_positive, metrics.sample_count) == (2, 2, 4)
    assert metrics.positive_rate == pytest.approx(0.5)


def test_zero_division_metrics_and_confusion_matrix() -> None:
    metrics = evaluate_predictions(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.5
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.accuracy == 0.5
    assert metrics.confusion_dict() == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 2,
        "true_positive": 0,
    }
    assert metrics.predicted_positive == 0
    assert metrics.positive_rate == 0.0
    assert metrics.support == 2


def test_single_class_slice_reports_no_skill_auc() -> None:
    metrics = evaluate_predictions(np.array([1, 1]), np.array([0.9, 0.1]), threshold=0.5)

    assert metrics.roc_auc == 0.5
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.support == 2


def test_invalid_threshold_and_labels_are_rejected() -> None:
    with pytest.raises(TrainingError):
        evaluate_predictions(np.array([0, 1]), np.array([0.2, 0.8]), threshold=1.5)
    with pytest.raises(TrainingError):
        evaluate_predictions(np.array([0, 2]), np.array([0.2, 0.8]))
    with pytest.raises(TrainingError):
        evaluate_predictions(np.array([0, 1]), np.array([0.2, np.nan]))


def test_two_fits_produce_identical_coefficients_and_metrics(
    training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    first_result = train_on_dataset(training_config, loaded_dataset)
    second_result = train_on_dataset(training_config, loaded_dataset)

    np.testing.assert_array_equal(
        classifier_coefficients(first_result.pipeline),
        classifier_coefficients(second_result.pipeline),
    )
    assert first_result.metrics == second_result.metrics
    assert (
        first_result.pipeline.predict_proba(
            split_frame(loaded_dataset, "validation")[list(loaded_dataset.schema.feature_names)]
        ).tolist()
        == second_result.pipeline.predict_proba(
            split_frame(loaded_dataset, "validation")[list(loaded_dataset.schema.feature_names)]
        ).tolist()
    )


def test_unknown_split_is_rejected(
    training_config: TrainingConfig, loaded_dataset: LoadedDataset
) -> None:
    with pytest.raises(TrainingError):
        train_on_dataset(training_config, loaded_dataset, split_name="train")


def test_metrics_metric_values_are_finite(loaded_dataset: LoadedDataset) -> None:
    metrics = evaluate_predictions(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert all(math.isfinite(value) for value in metrics.metric_values().values())
    assert isinstance(metrics, Metrics)
