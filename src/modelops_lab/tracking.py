"""MLflow run logging and model read-back for ModelOps Lab.

One training execution produces exactly one MLflow run. The run is the audit
record: every configuration value, the exact dataset and split fingerprints,
the environment that produced the numbers, all evaluation metrics, the fitted
pipeline, and a deterministic ``evaluation.json``.

Nothing here registers a model. W3 owns the registry; this module only creates
a run-scoped logged model and can load it back by run id.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from modelops_lab.config import TrainingConfig
from modelops_lab.data import LoadedDataset, split_frame
from modelops_lab.train import (
    Metrics,
    TrainingResult,
    evaluation_json_bytes,
    predict_probabilities,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ARTIFACT_NAME = "model"
EVALUATION_ARTIFACT_NAME = "evaluation.json"
INPUT_EXAMPLE_ROWS = 5


class TrackingError(RuntimeError):
    """Raised when MLflow cannot record or load a run."""


@dataclass(frozen=True)
class TrainingOutcome:
    """Identifiers and metrics produced by one logged training run."""

    run_id: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str
    model_id: str
    model_uri: str
    metrics: Metrics
    dataset_fingerprint: str


@dataclass(frozen=True)
class RunModel:
    """A model loaded from a run, plus the audit tags needed by evaluation."""

    model: Any
    model_uri: str
    run_id: str
    tags: dict[str, str]

    @property
    def trained_dataset_fingerprint(self) -> str | None:
        value = self.tags.get("dataset_fingerprint")
        return value or None


def resolve_tracking_uri(config_uri: str, override: str | None = None) -> str:
    """Resolve the tracking URI: explicit flag, then env, then config."""
    if override is not None and override.strip():
        return override.strip()
    environment = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if environment:
        return environment
    if not config_uri.strip():
        raise TrackingError(
            "tracking.uri is empty and no --tracking-uri or MLFLOW_TRACKING_URI was set"
        )
    return config_uri.strip()


def redact_uri(uri: str) -> str:
    """Strip a password from a URI before it is written to run metadata."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "<unparseable tracking uri>"
    if parts.password is None:
        return uri
    netloc = f"{parts.username}:***@{parts.hostname}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def git_revision() -> str:
    """Return ``git rev-parse HEAD``, falling back to ``GIT_SHA`` or ``unknown``.

    The environment fallback matters inside containers, where the checkout may
    not carry ``.git``. The value is never silently empty: an unavailable
    revision is recorded as ``unknown`` so a reader sees that it is absent.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode == 0:
        revision = completed.stdout.strip()
        if revision:
            return revision
    fallback = os.environ.get("GIT_SHA", "").strip()
    return fallback or "unknown"


def lockfile_sha256(path: Path | None = None) -> str:
    """Return the SHA-256 of ``uv.lock`` or an explicit ``unavailable`` marker."""
    lockfile = path or (REPO_ROOT / "uv.lock")
    try:
        digest = hashlib.sha256()
        with lockfile.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "unavailable"
    return digest.hexdigest()


def _package_version(distribution: str, fallback: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def environment_tags() -> dict[str, str]:
    """Return the interpreter and library versions that shaped the numbers."""
    return {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "sklearn_version": _package_version("scikit-learn", "unknown"),
        "mlflow_version": _package_version("mlflow", "unknown"),
        "numpy_version": _package_version("numpy", "unknown"),
        "pandas_version": _package_version("pandas", "unknown"),
    }


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = ",".join(str(item) for item in value)
    elif isinstance(value, bool):
        output[prefix] = "true" if value else "false"
    elif value is None:
        output[prefix] = ""
    else:
        output[prefix] = value


def flatten_params(config: TrainingConfig, threshold: float) -> dict[str, Any]:
    """Flatten the complete training config into MLflow param names.

    Dotted names mirror the YAML structure. The tracking URI is included in a
    credential-redacted form so an operator can identify the destination
    without writing a password into run metadata.
    """
    payload = config.model_dump(mode="json")
    payload["tracking"]["uri"] = redact_uri(str(payload["tracking"]["uri"]))
    flattened: dict[str, Any] = {}
    _flatten("", payload, flattened)
    flattened["model_type"] = config.model.type
    flattened["threshold"] = threshold
    return flattened


def run_tags(
    loaded: LoadedDataset, config: TrainingConfig, *, evaluation_split: str = "validation"
) -> dict[str, str]:
    """Build the audit tags, including dataset and per-split fingerprints."""
    regime = loaded.manifest.generator_parameters.get("regime")
    tags = {
        "git_revision": git_revision(),
        "uv_lock_sha256": lockfile_sha256(),
        **environment_tags(),
        "dataset_version": loaded.manifest.dataset_version,
        "dataset_fingerprint": loaded.manifest.fingerprint,
        "regime": regime if isinstance(regime, str) and regime else "unknown",
        "experiment_name": config.tracking.experiment_name,
        "evaluation_split": evaluation_split,
    }
    for split in loaded.manifest.splits:
        tags[f"split_{split.name}_fingerprint"] = split.fingerprint
        tags[f"split_{split.name}_row_count"] = str(split.row_count)
    return tags


def _ensure_experiment(client: MlflowClient, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(name)
    except MlflowException:
        # A concurrent creator may have won; re-read before treating it as fatal.
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        raise


def _input_example_frame(loaded: LoadedDataset, frame_limit: int = INPUT_EXAMPLE_ROWS):
    """Return the first training rows in declared feature order."""
    train = split_frame(loaded, "train").head(frame_limit)
    return train.loc[:, list(loaded.manifest.feature_names)]


def log_training_run(
    config: TrainingConfig,
    result: TrainingResult,
    *,
    tracking_uri: str,
    run_name: str | None = None,
) -> TrainingOutcome:
    """Write the complete audit record for ``result`` and return its identifiers."""
    metadata = result.evaluation_payload()
    evaluation_text = evaluation_json_bytes(metadata).decode("utf-8")
    feature_names = list(result.loaded.manifest.feature_names)
    example_frame = _input_example_frame(result.loaded)
    example = {name: example_frame[name].tolist() for name in feature_names}
    signature = infer_signature(
        example_frame,
        predict_probabilities(result.pipeline, example_frame, feature_names),
    )

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_id = _ensure_experiment(client, config.tracking.experiment_name)
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            log_system_metrics=False,
        ) as run:
            mlflow.log_params(
                flatten_params(config, result.threshold),
                synchronous=True,
            )
            mlflow.set_tags(
                run_tags(result.loaded, config, evaluation_split=result.split_name),
                synchronous=True,
            )
            mlflow.log_metrics(result.metrics.metric_values(), synchronous=True)
            model_info = mlflow.sklearn.log_model(
                result.pipeline,
                name=MODEL_ARTIFACT_NAME,
                signature=signature,
                input_example=example,
            )
            if not model_info.model_id:
                raise TrackingError("MLflow logged the model without returning a model id")
            mlflow.log_text(evaluation_text, EVALUATION_ARTIFACT_NAME)
            mlflow.set_tags(
                {"model_id": model_info.model_id, "model_uri": model_info.model_uri},
                synchronous=True,
            )
            run_id = run.info.run_id
    except TrackingError:
        raise
    except (MlflowException, OSError, TimeoutError, ValueError) as exc:
        raise TrackingError(f"cannot record the run at {redact_uri(tracking_uri)}: {exc}") from exc

    return TrainingOutcome(
        run_id=run_id,
        experiment_id=experiment_id,
        experiment_name=config.tracking.experiment_name,
        tracking_uri=tracking_uri,
        model_id=model_info.model_id,
        model_uri=model_info.model_uri,
        metrics=result.metrics,
        dataset_fingerprint=result.loaded.manifest.fingerprint,
    )


def load_run_model(run_id: str, tracking_uri: str) -> RunModel:
    """Load the logged model for ``run_id`` and return its audit tags.

    Prefers the MLflow 3 run output (a logged-model id). Older run-scoped
    artifacts, or a run created by an earlier version of this command, fall
    back to the recorded ``model_uri`` tag and then ``runs:/<id>/model``.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
    except (MlflowException, OSError, TimeoutError) as exc:
        raise TrackingError(
            f"cannot read run {run_id!r} from {redact_uri(tracking_uri)}: {exc}"
        ) from exc

    outputs = run.outputs.model_outputs if run.outputs is not None else []
    if outputs and outputs[0].model_id:
        model_uri = f"models:/{outputs[0].model_id}"
    else:
        model_uri = run.data.tags.get("model_uri") or f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    tags = {key: str(value) for key, value in run.data.tags.items()}

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except (MlflowException, OSError, ValueError) as exc:
        raise TrackingError(
            f"cannot load the model for run {run_id!r} from {model_uri}: {exc}"
        ) from exc
    return RunModel(model=model, model_uri=model_uri, run_id=run_id, tags=tags)
