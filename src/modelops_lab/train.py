"""Deterministic training, evaluation metrics, and logged-model evaluation.

The model stays deliberately simple (ADR-001): a scikit-learn pipeline with a
standard scaler for numeric features, a one-hot encoder for categorical
features, and logistic regression. The engineering guarantees are the product:

* preprocessing and the classifier are fit on the training split only;
* feature order is the declared schema order, never a pandas-inferred order;
* BLAS/OpenMP pools are limited to one thread around fit and predict, and the
  classifier seed is fixed, so identical inputs produce identical outputs;
* metrics are pure functions of ``(y_true, probabilities, threshold)`` and are
  reusable without MLflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from modelops_lab.config import ModelConfig, TrainingConfig
from modelops_lab.data import LoadedDataset, split_frame
from modelops_lab.dataset_schema import DatasetSchema

DEFAULT_DECISION_THRESHOLD = 0.5
NO_SKILL_ROC_AUC = 0.5
EVALUATION_SPLITS = ("validation", "test")


class TrainingError(RuntimeError):
    """Raised when training or evaluation cannot produce a valid result."""


@dataclass(frozen=True)
class Metrics:
    """Binary-classification metrics at one decision threshold.

    ``support`` is the number of rows whose observed label is positive;
    ``predicted_positive`` and ``positive_rate`` describe the predicted side.
    ROC-AUC is undefined for a single-class slice; it is recorded as the
    no-skill value ``0.5`` and the class counts remain part of the record so
    the condition is visible rather than silently implied.
    """

    roc_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    log_loss: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    support: int
    predicted_positive: int
    sample_count: int

    @property
    def positive_rate(self) -> float:
        return self.predicted_positive / self.sample_count

    def metric_values(self) -> dict[str, float]:
        """Return the flat metric dictionary logged to MLflow."""
        return {
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "confusion_matrix_tn": float(self.true_negative),
            "confusion_matrix_fp": float(self.false_positive),
            "confusion_matrix_fn": float(self.false_negative),
            "confusion_matrix_tp": float(self.true_positive),
            "support": float(self.support),
            "predicted_positive": float(self.predicted_positive),
            "positive_rate": self.positive_rate,
            "sample_count": float(self.sample_count),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe metric block used by ``evaluation.json``."""
        return {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "log_loss": self.log_loss,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc,
        }

    def confusion_dict(self) -> dict[str, int]:
        return {
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
        }


def build_pipeline(model_config: ModelConfig, schema: DatasetSchema) -> Pipeline:
    """Build the reference pipeline with a schema-stable feature order.

    The encoder ignores unseen categorical levels at inference time instead of
    raising, which is what lets a saved model score data outside the training
    sample. It never invents a level for an unknown value; such rows simply
    contribute no one-hot signal.
    """
    numeric_features = list(schema.numeric_features)
    categorical_features = list(schema.categorical_features)
    if not numeric_features and not categorical_features:
        raise TrainingError(f"schema {schema.name!r} declares no features")

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )
    preprocessing = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    classifier = LogisticRegression(
        C=model_config.hyperparameters.C,
        max_iter=model_config.hyperparameters.max_iter,
        solver=model_config.hyperparameters.solver,
        random_state=model_config.random_state,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def feature_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Select features in declared order, failing loudly if one is missing."""
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise TrainingError("dataset is missing declared feature column(s): " + ", ".join(missing))
    return frame.loc[:, feature_names]


def predict_probabilities(
    pipeline: Pipeline, frame: pd.DataFrame, feature_names: list[str]
) -> np.ndarray:
    """Return the positive-class probability for every row, single-threaded."""
    with threadpool_limits(limits=1):
        probabilities = pipeline.predict_proba(feature_frame(frame, feature_names))
    return np.asarray(probabilities[:, 1], dtype=float)


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> Metrics:
    """Compute auditable binary metrics from labels and probabilities.

    Zero-division cases follow scikit-learn's ``zero_division=0`` convention:
    precision is 0 when nothing is predicted positive, recall and F1 are 0 when
    there are no observed positives, and confusion counts are always reported
    so the reason is explicit.
    """
    if not 0.0 < threshold < 1.0:
        raise TrainingError(f"decision threshold must be in (0, 1), got {threshold!r}")
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise TrainingError("y_true and probabilities must be one-dimensional and equal length")
    if labels.size == 0:
        raise TrainingError("cannot evaluate an empty slice")
    if not np.isin(labels, (0, 1)).all():
        raise TrainingError("y_true contains a label outside {0, 1}")
    if not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
        raise TrainingError("probabilities must be finite and inside [0, 1]")

    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    present_classes = np.unique(labels)
    if present_classes.size == 2:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = NO_SKILL_ROC_AUC

    return Metrics(
        roc_auc=roc_auc,
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        accuracy=float(accuracy_score(labels, predictions)),
        log_loss=float(log_loss(labels, scores, labels=[0, 1])),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
        support=int(np.count_nonzero(labels == 1)),
        predicted_positive=int(np.count_nonzero(predictions == 1)),
        sample_count=int(labels.size),
    )


def fit_and_evaluate(
    model_config: ModelConfig,
    schema: DatasetSchema,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> tuple[Pipeline, Metrics]:
    """Fit on ``train_frame`` only and evaluate on a separate slice.

    Preprocessing is part of the pipeline, so ``fit`` sees the training rows
    alone. Validation and test rows are never passed to ``fit`` or
    ``fit_transform``; they are only scored after the model is complete.
    """
    feature_names = list(schema.feature_names)
    if "target" not in train_frame.columns or "target" not in evaluation_frame.columns:
        raise TrainingError("train and evaluation frames must both contain a 'target' column")
    train_labels = train_frame["target"].to_numpy(dtype=np.int64)
    if np.unique(train_labels).size < 2:
        raise TrainingError("the training split must contain both classes")

    pipeline = build_pipeline(model_config, schema)
    with threadpool_limits(limits=1):
        pipeline.fit(feature_frame(train_frame, feature_names), train_labels)
    probabilities = predict_probabilities(pipeline, evaluation_frame, feature_names)
    metrics = evaluate_predictions(
        evaluation_frame["target"].to_numpy(dtype=np.int64), probabilities, threshold
    )
    return pipeline, metrics


def classifier_coefficients(pipeline: Pipeline) -> np.ndarray:
    """Return a flat copy of the classifier coefficients for comparisons."""
    coefficients = np.asarray(pipeline.named_steps["classifier"].coef_, dtype=float)
    return coefficients.reshape(-1).copy()


def evaluation_payload(
    loaded: LoadedDataset,
    split_name: str,
    metrics: Metrics,
    *,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Build the deterministic ``evaluation.json`` document (no timestamps)."""
    if split_name not in loaded.splits:
        raise TrainingError(f"unknown evaluation split {split_name!r}")
    split_manifests = {split.name: split for split in loaded.manifest.splits}
    return {
        "confusion_matrix": metrics.confusion_dict(),
        "counts": {
            "predicted_positive": metrics.predicted_positive,
            "sample_count": metrics.sample_count,
            "support": metrics.support,
        },
        "dataset_fingerprint": loaded.manifest.fingerprint,
        "dataset_version": loaded.manifest.dataset_version,
        "evaluation_split": split_name,
        "features": list(loaded.manifest.feature_names),
        "metrics": metrics.as_dict(),
        "positive_rate": metrics.positive_rate,
        "split_fingerprint": split_manifests[split_name].fingerprint,
        "split_fingerprints": {
            name: split_manifests[name].fingerprint for name in sorted(split_manifests)
        },
        "threshold": threshold,
    }


def evaluation_json_bytes(payload: dict[str, Any]) -> bytes:
    """Render a payload as stable, sorted, LF-terminated JSON bytes."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


@dataclass(frozen=True)
class TrainingResult:
    """A fitted pipeline plus the metrics it produced on one evaluation slice."""

    pipeline: Pipeline
    metrics: Metrics
    loaded: LoadedDataset
    split_name: str
    threshold: float

    def evaluation_payload(self) -> dict[str, Any]:
        return evaluation_payload(
            self.loaded, self.split_name, self.metrics, threshold=self.threshold
        )


def train_on_dataset(
    config: TrainingConfig,
    loaded: LoadedDataset,
    *,
    split_name: str = "validation",
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> TrainingResult:
    """Fit on the verified train split and evaluate on ``split_name``."""
    if split_name not in EVALUATION_SPLITS:
        raise TrainingError(
            f"evaluation split must be one of {', '.join(EVALUATION_SPLITS)}; got {split_name!r}"
        )
    train_frame = split_frame(loaded, "train")
    evaluation_frame = split_frame(loaded, split_name)
    pipeline, metrics = fit_and_evaluate(
        config.model, loaded.schema, train_frame, evaluation_frame, threshold=threshold
    )
    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        loaded=loaded,
        split_name=split_name,
        threshold=threshold,
    )


def evaluate_loaded_model(
    pipeline: Pipeline,
    loaded: LoadedDataset,
    *,
    split_name: str,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> Metrics:
    """Evaluate an already-loaded pipeline on one verified split."""
    if split_name not in EVALUATION_SPLITS:
        raise TrainingError(
            f"evaluation split must be one of {', '.join(EVALUATION_SPLITS)}; got {split_name!r}"
        )
    frame = split_frame(loaded, split_name)
    try:
        probabilities = predict_probabilities(pipeline, frame, list(loaded.manifest.feature_names))
    except (ValueError, TypeError) as exc:
        raise TrainingError(
            f"model cannot score the {split_name} split of this dataset: {exc}"
        ) from exc
    return evaluate_predictions(frame["target"].to_numpy(dtype=np.int64), probabilities, threshold)
