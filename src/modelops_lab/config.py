"""Validated configuration models and YAML loaders for ModelOps Lab.

YAML files under ``configs/`` are authoritative for experiment inputs.
Environment variables prefixed with ``MODELOPS_`` override YAML values, using
``__`` as the nesting separator and matching field names case-insensitively::

    MODELOPS_DATASET__SEED=7
    MODELOPS_MODEL__HYPERPARAMETERS__C=0.5

Runtime settings (database URL, MLflow URI, artifact root, app port) are loaded
from the environment or a local ``.env`` file through :class:`RuntimeSettings`.
"""

from __future__ import annotations

import json
import os
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from modelops_lab.contracts.base import DatasetKind

ENV_PREFIX = "MODELOPS_"
DEFAULT_CONFIG_DIR = Path("configs")


class ConfigError(ValueError):
    """Raised when a configuration file cannot be read or fails validation."""


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown YAML keys."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DatasetConfig(StrictConfigModel):
    """Source dataset generation or loading parameters."""

    kind: DatasetKind = Field(
        description="Synthetic generator or the optional breast-cancer loader."
    )
    seed: int = Field(
        ge=0,
        le=2**32 - 1,
        description="Seed for dataset generation or deterministic sampling.",
    )
    n_samples: int = Field(
        ge=100,
        description="Rows for the synthetic generator; fixed-size loaders ignore this value.",
    )
    regime: Literal["baseline", "covariate_shift", "improved_signal"] = Field(
        default="baseline",
        description=(
            "Synthetic distribution/label regime; the breast-cancer loader accepts only baseline."
        ),
    )


class SplitConfig(StrictConfigModel):
    """Deterministic train/validation/test partition fractions."""

    seed: int = Field(ge=0, le=2**32 - 1, description="Seed for partition assignment.")
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> Self:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")
        return self


class LogisticRegressionHyperparameters(StrictConfigModel):
    """Hyperparameters for the reference scikit-learn classifier."""

    C: float = Field(gt=0.0, description="Inverse regularization strength.")
    max_iter: int = Field(ge=1)
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]


class ModelConfig(StrictConfigModel):
    """Model family and deterministic training settings."""

    type: Literal["logistic_regression"] = "logistic_regression"
    random_state: int = Field(ge=0, le=2**32 - 1)
    hyperparameters: LogisticRegressionHyperparameters


class TrackingConfig(StrictConfigModel):
    """MLflow tracking destination for training runs."""

    uri: str = Field(min_length=1, description="MLflow tracking server URI.")
    experiment_name: str = Field(min_length=1)


class TrainingConfig(StrictConfigModel):
    """Complete input contract for the training pipeline."""

    dataset: DatasetConfig
    split: SplitConfig
    model: ModelConfig
    tracking: TrackingConfig


class QualityFloor(StrictConfigModel):
    """Minimum validation metrics a candidate must reach to be promotable."""

    roc_auc: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class RegressionTolerance(StrictConfigModel):
    """Maximum allowed drop versus the active model on the same evaluation slice."""

    roc_auc: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class SmokeTestConfig(StrictConfigModel):
    """Post-load inference smoke test required before approval."""

    enabled: bool = True
    sample_count: int = Field(ge=1, description="Synthetic requests sent through the loaded model.")
    max_latency_ms: float = Field(gt=0.0)


class PromotionConfig(StrictConfigModel):
    """Policy and guardrails for candidate promotion."""

    quality_floor: QualityFloor
    regression_tolerance: RegressionTolerance
    required_artifacts: list[str] = Field(min_length=1)
    auto_promote: bool = Field(
        default=False,
        description="Reserved flag; automated promotion is not implemented and must remain false.",
    )
    smoke_test: SmokeTestConfig

    @field_validator("required_artifacts")
    @classmethod
    def _artifact_names_are_unique_and_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [name.strip() for name in value]
        if any(not name for name in cleaned):
            raise ValueError("required_artifacts must not contain empty names")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("required_artifacts must be unique")
        return cleaned

    @field_validator("auto_promote")
    @classmethod
    def _auto_promote_is_reserved(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "auto_promote is reserved and must remain false until automated promotion "
                "is implemented"
            )
        return value


class ReferenceWindowConfig(StrictConfigModel):
    """Window used as the monitoring baseline."""

    source: Literal["training_split", "prediction_events"]
    window_size: int = Field(ge=1, description="Maximum reference rows to include.")


class CurrentWindowConfig(StrictConfigModel):
    """Most recent inference window compared against the reference."""

    window_size: int = Field(ge=1)


class FeatureDriftThresholds(StrictConfigModel):
    """Per-feature thresholds for the in-repo drift statistics."""

    psi_threshold: float = Field(gt=0.0)
    ks_pvalue_threshold: float = Field(gt=0.0, le=1.0)
    missing_rate_delta_threshold: float = Field(ge=0.0, le=1.0)


class MonitoringConfig(StrictConfigModel):
    """Drift monitoring windows, sample guards, and report destination."""

    reference: ReferenceWindowConfig
    current: CurrentWindowConfig
    min_sample_size: int = Field(ge=1)
    feature_drift: FeatureDriftThresholds
    aggregate_feature_share_threshold: float = Field(
        gt=0.0, le=1.0, description="Share of drifted features that triggers an aggregate alert."
    )
    report_output_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def _sample_guards_fit_windows(self) -> Self:
        if self.reference.window_size < self.min_sample_size:
            raise ValueError(
                f"reference.window_size must be at least min_sample_size={self.min_sample_size}"
            )
        if self.current.window_size < self.min_sample_size:
            raise ValueError(
                f"current.window_size must be at least min_sample_size={self.min_sample_size}"
            )
        return self


class RuntimeSettings(BaseSettings):
    """Process-level settings sourced from ``MODELOPS_*`` environment variables.

    Values in a local ``.env`` file are used as defaults; real environment
    variables take precedence. The defaults target a local development stack,
    but no secrets belong in this model.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(
        default="postgresql+psycopg://modelops:modelops@localhost:5432/modelops",
        min_length=1,
    )
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", min_length=1)
    artifact_root: Path = Path("artifacts")
    app_port: int = Field(default=8000, ge=1, le=65535)


def _format_validation_errors(path: Path, error: ValidationError) -> str:
    lines = [f"{path}: invalid configuration"]
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "<root>"
        lines.append(f"  {location}: {detail['msg']}")
    return "\n".join(lines)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise ConfigError(f"{path}: cannot read configuration file: {reason}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML document must be a mapping")
    return raw


def _nested_model_type(annotation: Any) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return _nested_model_type(members[0])
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _match_field(name: str, model_type: type[BaseModel]) -> str | None:
    for field_name in model_type.model_fields:
        if field_name.lower() == name.lower():
            return field_name
    return None


def _resolve_env_path(model_type: type[BaseModel], parts: list[str]) -> list[str]:
    resolved: list[str] = []
    current_type: type[BaseModel] | None = model_type
    for index, part in enumerate(parts):
        if current_type is None:
            resolved.extend(parts[index:])
            break
        field_name = _match_field(part, current_type)
        if field_name is None:
            resolved.extend(part.lower() for part in parts[index:])
            break
        resolved.append(field_name)
        current_type = _nested_model_type(current_type.model_fields[field_name].annotation)
    return resolved


def _parse_env_value(raw: str) -> Any:
    stripped = raw.strip()
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return raw


def _environment_overrides(
    model_type: type[BaseModel], environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    source = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for key, raw in source.items():
        if not key.startswith(ENV_PREFIX):
            continue
        remainder = key[len(ENV_PREFIX) :]
        if not remainder:
            continue
        raw_parts = remainder.split("__")
        if _match_field(raw_parts[0], model_type) is None:
            continue
        parts = _resolve_env_path(model_type, raw_parts)
        cursor = overrides
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                existing = {}
                cursor[part] = existing
            elif not isinstance(existing, dict):
                joined = "__".join(parts)
                raise ConfigError(
                    f"environment override for {joined} conflicts with a scalar value"
                )
            cursor = existing
        cursor[parts[-1]] = _parse_env_value(raw)
    return overrides


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _load[ConfigT: BaseModel](path: Path, model_type: type[ConfigT]) -> ConfigT:
    data = _read_yaml(path)
    overrides = _environment_overrides(model_type)
    try:
        return model_type.model_validate(_merge(data, overrides))
    except ValidationError as exc:
        raise ConfigError(_format_validation_errors(path, exc)) from exc


def _resolve_path(path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return DEFAULT_CONFIG_DIR / default_name
    return Path(path)


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Load and validate ``configs/training.yaml`` or an explicit path."""
    return _load(_resolve_path(path, "training.yaml"), TrainingConfig)


def load_promotion_config(path: str | Path | None = None) -> PromotionConfig:
    """Load and validate ``configs/promotion.yaml`` or an explicit path."""
    return _load(_resolve_path(path, "promotion.yaml"), PromotionConfig)


def load_monitoring_config(path: str | Path | None = None) -> MonitoringConfig:
    """Load and validate ``configs/monitoring.yaml`` or an explicit path."""
    return _load(_resolve_path(path, "monitoring.yaml"), MonitoringConfig)


def load_runtime_settings() -> RuntimeSettings:
    """Load runtime settings from ``MODELOPS_*`` variables and an optional ``.env`` file."""
    return RuntimeSettings()
