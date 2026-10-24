"""Deterministic dataset generation, canonical serialization, and artifacts.

The synthetic generator is the primary workload (ADR-001). It draws correlated
numeric features, categorical features, and a binary label from a logistic
function of a latent linear score. The label is *sampled*, so the target is
noisy rather than a deterministic threshold on the score.

Three regimes share one schema:

``baseline``
    Reference distribution and reference label mechanism.
``covariate_shift``
    Only ``P(X)`` changes: numeric means/scales and categorical proportions
    move, while the latent coefficients, standardization constants, and link
    stay identical.
``improved_signal``
    ``P(X)`` is byte-for-byte the baseline draws; the label mechanism changes
    (reweighted, larger coefficients and a sharper link), so a model trained on
    this regime can genuinely beat a baseline-trained model on the same rows.

Artifacts are written as::

    <out>/<dataset_version>/
        dataset.csv      canonical bytes (see :func:`canonical_csv_bytes`)
        manifest.json    DatasetManifest as JSON with sorted keys
        splits.json      {"train": [row_id...], "validation": [...], "test": [...]}
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from modelops_lab.config import DatasetConfig, SplitConfig
from modelops_lab.contracts import DatasetManifest, SplitManifest
from modelops_lab.dataset_schema import (
    BREAST_CANCER_SCHEMA,
    SCHEMAS_BY_SOURCE_KIND,
    SYNTHETIC_CATEGORICAL_LEVELS,
    SYNTHETIC_NUMERIC_FEATURES,
    SYNTHETIC_NUMERIC_RANGES,
    SYNTHETIC_SCHEMA,
    DatasetSchema,
)
from modelops_lab.validation import DatasetValidationError, read_dataset_csv, validate_dataset

Regime = Literal["baseline", "covariate_shift", "improved_signal"]
REGIMES: tuple[Regime, ...] = ("baseline", "covariate_shift", "improved_signal")
SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")
DEFAULT_DATASET_ROOT = Path("artifacts/datasets")

# Baseline marginal parameters for the numeric features. The same constants
# standardize features inside the latent score for every regime, which is what
# keeps P(y|X) identical under covariate shift.
_BASELINE_NUMERIC: dict[str, tuple[float, float]] = {
    "age": (52.0, 12.0),
    "bmi": (27.5, 4.5),
    "systolic_bp": (128.0, 16.0),
    "diastolic_bp": (82.0, 10.0),
    "cholesterol": (235.0, 38.0),
    "glucose": (98.0, 14.0),
    "resting_heart_rate": (72.0, 10.0),
    "max_heart_rate": (150.0, 20.0),
    "st_depression": (1.0, 0.9),
    "exercise_minutes": (95.0, 35.0),
}

# Rounding keeps values clinically plausible and shortens the CSV without
# weakening round-trip exactness (the rounded float is the stored value).
_DECIMALS: dict[str, int] = {
    **{name: 0 for name in SYNTHETIC_NUMERIC_FEATURES},
    "bmi": 1,
    "st_depression": 2,
}

# Modest, explicit correlations between numeric features (index pairs).
_CORRELATION_PAIRS: dict[tuple[int, int], float] = {
    (0, 2): 0.45,  # age / systolic_bp
    (0, 3): 0.30,  # age / diastolic_bp
    (2, 3): 0.55,  # systolic_bp / diastolic_bp
    (1, 5): 0.35,  # bmi / glucose
    (1, 9): -0.25,  # bmi / exercise_minutes
    (0, 7): -0.40,  # age / max_heart_rate
    (0, 4): 0.18,  # age / cholesterol
    (4, 5): 0.18,  # cholesterol / glucose
    (2, 1): 0.20,  # systolic_bp / bmi
    (6, 9): -0.15,  # resting_heart_rate / exercise_minutes
}

_BASELINE_COEFFICIENTS: dict[str, float] = {
    "age": 0.825,
    "bmi": 0.42,
    "systolic_bp": 0.48,
    "diastolic_bp": 0.12,
    "cholesterol": 0.45,
    "glucose": 0.63,
    "resting_heart_rate": 0.18,
    "max_heart_rate": -0.48,
    "st_depression": 0.72,
    "exercise_minutes": -0.33,
}

_CATEGORICAL_EFFECTS: dict[str, dict[str, float]] = {
    "sex": {"female": 0.0, "male": 0.32},
    "chest_pain_type": {
        "typical": 0.55,
        "atypical": 0.25,
        "non_anginal": 0.0,
        "asymptomatic": -0.40,
    },
    "smoker": {"never": 0.0, "former": 0.12, "current": 0.28},
}

_BASELINE_CATEGORICAL_PROBABILITIES: dict[str, dict[str, float]] = {
    "sex": {"female": 0.58, "male": 0.42},
    "chest_pain_type": {
        "typical": 0.16,
        "atypical": 0.24,
        "non_anginal": 0.22,
        "asymptomatic": 0.38,
    },
    "smoker": {"never": 0.52, "former": 0.30, "current": 0.18},
}

# Covariate shift: an older, narrower screening cohort with a different
# referral and smoking mix. Scales below 1.0 reduce spread, which also removes
# some of the ranking signal the baseline model relied on (the mechanism is
# unchanged). Mean shifts stay modest so every seed keeps the base rate well
# inside the declared [0.20, 0.60] gate.
_COVARIATE_NUMERIC: dict[str, tuple[float, float]] = {
    "age": (1.5, 0.60),
    "bmi": (1.0, 0.70),
    "systolic_bp": (3.0, 0.65),
    "diastolic_bp": (2.5, 0.75),
    "cholesterol": (5.0, 0.65),
    "glucose": (2.0, 0.60),
    "resting_heart_rate": (2.5, 0.75),
    "max_heart_rate": (-2.0, 0.70),
    "st_depression": (0.08, 0.65),
    "exercise_minutes": (-5.0, 0.70),
}

_COVARIATE_CATEGORICAL_PROBABILITIES: dict[str, dict[str, float]] = {
    "sex": {"female": 0.50, "male": 0.50},
    "chest_pain_type": {
        "typical": 0.10,
        "atypical": 0.18,
        "non_anginal": 0.20,
        "asymptomatic": 0.52,
    },
    "smoker": {"never": 0.38, "former": 0.28, "current": 0.34},
}

# Improved regime: metabolic and exercise features dominate, symptom and
# demographic effects recede, coefficients grow, and the link sharpens.
# Uniformly scaling coefficients alone would be a monotone transform and could
# not change ROC-AUC; the reweighting is what makes the comparison meaningful.
_IMPROVED_NUMERIC_MULTIPLIERS: dict[str, float] = {
    "age": 0.6,
    "bmi": 0.8,
    "systolic_bp": 0.8,
    "diastolic_bp": 0.6,
    "cholesterol": 0.6,
    "glucose": 1.6,
    "resting_heart_rate": 0.8,
    "max_heart_rate": 1.4,
    "st_depression": 1.5,
    "exercise_minutes": 1.6,
}
_IMPROVED_SIGNAL_SCALE = 1.3
_IMPROVED_CATEGORICAL_SCALE = 0.5
_IMPROVED_LINK_TEMPERATURE = 0.72

_LATENT_INTERCEPT = -0.75
_BASELINE_LINK_TEMPERATURE = 1.0


class DataError(RuntimeError):
    """Raised for dataset preparation, artifact, and verification failures."""


def _correlation_matrix() -> np.ndarray:
    size = len(SYNTHETIC_NUMERIC_FEATURES)
    matrix = np.eye(size)
    for (left, right), value in _CORRELATION_PAIRS.items():
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


_CHOLESKY = np.linalg.cholesky(_correlation_matrix())


def _improved_coefficients() -> dict[str, float]:
    return {
        name: round(
            _BASELINE_COEFFICIENTS[name]
            * _IMPROVED_NUMERIC_MULTIPLIERS[name]
            * _IMPROVED_SIGNAL_SCALE,
            6,
        )
        for name in SYNTHETIC_NUMERIC_FEATURES
    }


_IMPROVED_COEFFICIENTS = _improved_coefficients()


def label_probabilities(frame: pd.DataFrame, regime: Regime) -> np.ndarray:
    """Return ``P(y=1|X)`` for ``frame`` under ``regime``.

    The standardization constants and (for covariate shift) the true
    coefficients are identical across regimes, so the same feature row always
    maps to the same probability. This is the invariant ``covariate_shift``
    isolates and ``improved_signal`` deliberately breaks.
    """
    if regime == "improved_signal":
        coefficients = _IMPROVED_COEFFICIENTS
        categorical_scale = _IMPROVED_CATEGORICAL_SCALE
        temperature = _IMPROVED_LINK_TEMPERATURE
    elif regime in {"baseline", "covariate_shift"}:
        coefficients = _BASELINE_COEFFICIENTS
        categorical_scale = 1.0
        temperature = _BASELINE_LINK_TEMPERATURE
    else:  # pragma: no cover - guarded by DatasetConfig
        raise DataError(f"unknown regime: {regime!r}")

    score = np.full(len(frame), _LATENT_INTERCEPT, dtype=float)
    for name in SYNTHETIC_NUMERIC_FEATURES:
        mean, scale = _BASELINE_NUMERIC[name]
        values = frame[name].to_numpy(dtype=float)
        score += coefficients[name] * (values - mean) / scale
    for name in SYNTHETIC_CATEGORICAL_LEVELS:
        effects = _CATEGORICAL_EFFECTS[name]
        score += categorical_scale * frame[name].map(effects).to_numpy(dtype=float)
    return 1.0 / (1.0 + np.exp(-score / temperature))


def generate_synthetic(n_samples: int, seed: int, regime: Regime = "baseline") -> pd.DataFrame:
    """Generate the seeded synthetic dataset for ``regime``.

    One PCG64 stream is consumed in a fixed order: latent normals, categorical
    draws, then label uniforms. ``baseline`` and ``improved_signal`` therefore
    share identical feature values for a given seed, while ``covariate_shift``
    changes only the feature distributions.
    """
    if n_samples < 1:
        raise DataError(f"n_samples must be positive, got {n_samples}")
    rng = np.random.Generator(np.random.PCG64(seed))
    latent = rng.standard_normal((n_samples, len(SYNTHETIC_NUMERIC_FEATURES))) @ _CHOLESKY.T

    shift = _COVARIATE_NUMERIC if regime == "covariate_shift" else {}
    data: dict[str, np.ndarray] = {}
    for index, name in enumerate(SYNTHETIC_NUMERIC_FEATURES):
        mean, scale = _BASELINE_NUMERIC[name]
        delta_mean, delta_scale = shift.get(name, (0.0, 1.0))
        low, high = SYNTHETIC_NUMERIC_RANGES[name]
        raw = mean + delta_mean + (scale * delta_scale) * latent[:, index]
        data[name] = np.round(np.clip(raw, low, high), _DECIMALS[name]).astype(np.float64)

    probabilities = (
        _COVARIATE_CATEGORICAL_PROBABILITIES
        if regime == "covariate_shift"
        else _BASELINE_CATEGORICAL_PROBABILITIES
    )
    for name, levels in SYNTHETIC_CATEGORICAL_LEVELS.items():
        distribution = probabilities[name]
        data[name] = rng.choice(
            np.asarray(levels),
            size=n_samples,
            p=np.asarray([distribution[level] for level in levels]),
        )

    frame = pd.DataFrame(data, columns=list(SYNTHETIC_SCHEMA.feature_names))
    frame.insert(0, "row_id", np.arange(1, n_samples + 1, dtype=np.int64))
    label_draws = rng.random(n_samples)
    frame["target"] = (label_draws < label_probabilities(frame, regime)).astype(np.int64)
    return frame[list(SYNTHETIC_SCHEMA.columns)]


def load_breast_cancer_dataset() -> pd.DataFrame:
    """Load the bundled WDBC data with ``row_id`` and malignant-as-1 labels.

    scikit-learn encodes malignant as 0 and benign as 1 (base rate 0.627).
    Labels are inverted so ``target=1`` means malignant, which keeps the
    declared base-rate gate meaningful and matches the usual clinical reading.
    """
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    feature_names = tuple(str(name) for name in bunch.feature_names)
    if feature_names != BREAST_CANCER_SCHEMA.numeric_features:
        raise DataError(
            "the installed scikit-learn breast-cancer feature names do not match the "
            "declared schema; refusing to guess"
        )
    frame = pd.DataFrame(bunch.data, columns=list(feature_names), dtype=np.float64)
    frame.insert(0, "row_id", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame["target"] = (1 - np.asarray(bunch.target, dtype=np.int64)).astype(np.int64)
    return frame[list(BREAST_CANCER_SCHEMA.columns)]


@dataclass(frozen=True)
class DatasetSource:
    """A generated/loaded frame plus the declarations needed to reproduce it."""

    frame: pd.DataFrame
    schema: DatasetSchema
    generator_parameters: dict[str, Any]


def _synthetic_parameters(regime: Regime, n_samples: int, seed: int) -> dict[str, Any]:
    if regime == "improved_signal":
        numeric_multipliers: dict[str, float] = dict(_IMPROVED_NUMERIC_MULTIPLIERS)
        categorical_scale: float = _IMPROVED_CATEGORICAL_SCALE
        temperature: float = _IMPROVED_LINK_TEMPERATURE
        numeric_delta: dict[str, dict[str, float]] = {}
        categorical_probabilities = _BASELINE_CATEGORICAL_PROBABILITIES
    elif regime == "covariate_shift":
        numeric_multipliers = {name: 1.0 for name in SYNTHETIC_NUMERIC_FEATURES}
        categorical_scale = 1.0
        temperature = _BASELINE_LINK_TEMPERATURE
        numeric_delta = {
            name: {"mean_shift": mean_shift, "scale_multiplier": scale_multiplier}
            for name, (mean_shift, scale_multiplier) in _COVARIATE_NUMERIC.items()
        }
        categorical_probabilities = _COVARIATE_CATEGORICAL_PROBABILITIES
    else:
        numeric_multipliers = {name: 1.0 for name in SYNTHETIC_NUMERIC_FEATURES}
        categorical_scale = 1.0
        temperature = _BASELINE_LINK_TEMPERATURE
        numeric_delta = {}
        categorical_probabilities = _BASELINE_CATEGORICAL_PROBABILITIES

    numeric_distribution = {
        name: {
            "mean": _BASELINE_NUMERIC[name][0] + numeric_delta.get(name, {}).get("mean_shift", 0.0),
            "scale": _BASELINE_NUMERIC[name][1]
            * numeric_delta.get(name, {}).get("scale_multiplier", 1.0),
            "min": SYNTHETIC_NUMERIC_RANGES[name][0],
            "max": SYNTHETIC_NUMERIC_RANGES[name][1],
            "decimals": _DECIMALS[name],
        }
        for name in SYNTHETIC_NUMERIC_FEATURES
    }
    return {
        "generator": "modelops_lab.data.generate_synthetic",
        "generator_version": "1",
        "provenance": "generated in-repo from a seeded PCG64 stream; no external data",
        "regime": regime,
        "n_samples": n_samples,
        "seed": seed,
        "numeric_distribution": numeric_distribution,
        "categorical_probabilities": {
            name: {level: probability for level, probability in probabilities.items()}
            for name, probabilities in categorical_probabilities.items()
        },
        "label_mechanism": {
            "target_sampling": "bernoulli",
            "link": "logistic",
            "intercept": _LATENT_INTERCEPT,
            "temperature": temperature,
            "coefficients": dict(
                _IMPROVED_COEFFICIENTS if regime == "improved_signal" else _BASELINE_COEFFICIENTS
            ),
            "numeric_multipliers": numeric_multipliers,
            "categorical_effect_scale": categorical_scale,
            "categorical_effects": {
                name: dict(effects) for name, effects in _CATEGORICAL_EFFECTS.items()
            },
            "standardization": {
                name: {"mean": mean, "scale": scale}
                for name, (mean, scale) in _BASELINE_NUMERIC.items()
            },
        },
    }


def _breast_cancer_parameters(configured_n_samples: int) -> dict[str, Any]:
    return {
        "loader": "sklearn.datasets.load_breast_cancer",
        "loader_version": "1",
        "provenance": (
            "UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic); "
            "Street, Wolberg & Mangasarian (1993)"
        ),
        "license": (
            "UCI ML Repository dataset terms; loaded from the scikit-learn package "
            "and not redistributed by this repository"
        ),
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "n_samples": 569,
        "configured_n_samples_ignored": configured_n_samples,
        "feature_count": 30,
        "target_encoding": {"0": "benign", "1": "malignant"},
        "label_transform": "1 - sklearn.target",
        "regime": "not_applicable",
    }


def build_dataset_source(config: DatasetConfig) -> DatasetSource:
    """Build the in-memory dataset and provenance for one configuration."""
    regime: Regime = config.regime
    if config.kind == "synthetic":
        return DatasetSource(
            frame=generate_synthetic(config.n_samples, config.seed, regime),
            schema=SYNTHETIC_SCHEMA,
            generator_parameters=_synthetic_parameters(regime, config.n_samples, config.seed),
        )
    if config.kind == "breast_cancer":
        if regime != "baseline":
            raise DataError(
                f"regime {regime!r} is only supported for the synthetic generator; "
                "the breast_cancer loader has no drift regimes"
            )
        return DatasetSource(
            frame=load_breast_cancer_dataset(),
            schema=BREAST_CANCER_SCHEMA,
            generator_parameters=_breast_cancer_parameters(config.n_samples),
        )
    raise DataError(f"unknown dataset kind: {config.kind!r}")


def _format_value(kind: str, value: Any) -> str:
    if kind in {"row_id", "target"}:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise DataError(f"expected an integer for a {kind} column, got {value!r}")
        return str(int(value))
    if kind == "numeric":
        number = float(value)
        if not math.isfinite(number):
            raise DataError(f"refusing to serialize a non-finite numeric value: {value!r}")
        return format(number, ".17g")
    text = str(value)
    if not text:
        raise DataError("refusing to serialize an empty category value")
    return text


def canonical_csv_bytes(frame: pd.DataFrame, schema: DatasetSchema) -> bytes:
    """Serialize ``frame`` to canonical CSV bytes.

    The same function is used for writing and hashing, so a dataset artifact is
    self-verifying. Rules: declared column order, rows sorted by ``row_id``,
    UTF-8, LF line endings, one header row, no index, floats formatted with
    ``%.17g`` (round-trip exact), integers in decimal.
    """
    actual = list(frame.columns)
    if actual != list(schema.columns):
        raise DataError(
            f"cannot serialize: expected columns [{', '.join(schema.columns)}], "
            f"found [{', '.join(str(name) for name in actual)}]"
        )
    if "row_id" not in frame.columns:
        raise DataError("cannot serialize: row_id column is missing")
    ordered = frame.sort_values("row_id", kind="mergesort")
    kinds = [schema.kind_of(name) for name in schema.columns]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(schema.columns)
    for row in ordered.itertuples(index=False, name=None):
        writer.writerow(
            [_format_value(kind, value) for kind, value in zip(kinds, row, strict=True)]
        )
    return buffer.getvalue().encode("utf-8")


def fingerprint_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def dataset_version_for(source_kind: str, data_schema_version: str, fingerprint: str) -> str:
    """Build ``<source_kind>-v<data_schema_version>-<fingerprint[:12]>``."""
    return f"{source_kind}-v{data_schema_version}-{fingerprint[:12]}"


def split_row_counts(row_count: int, split: SplitConfig) -> dict[str, int]:
    """Assign exact split sizes with largest-remainder rounding.

    Ties go to the earlier declared split, so identical inputs always produce
    identical counts. A split that rounds to zero is an error rather than a
    silently empty partition: ``SplitManifest`` (and meaningful metrics) need
    at least one row.
    """
    fractions = (split.train_fraction, split.validation_fraction, split.test_fraction)
    quotas = [fraction * row_count for fraction in fractions]
    counts = [int(math.floor(quota)) for quota in quotas]
    remainder = row_count - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    if any(count < 1 for count in counts):
        raise DataError(
            "split fractions produce an empty partition for "
            f"{row_count} rows (counts: {dict(zip(SPLIT_NAMES, counts, strict=True))}); "
            "increase the dataset size or the fraction"
        )
    return dict(zip(SPLIT_NAMES, counts, strict=True))


def assign_splits(row_ids: np.ndarray, split: SplitConfig) -> dict[str, list[int]]:
    """Partition ``row_ids`` with a seeded permutation, preserving split order."""
    counts = split_row_counts(len(row_ids), split)
    permutation = np.random.Generator(np.random.PCG64(split.seed)).permutation(
        np.asarray(row_ids, dtype=np.int64)
    )
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    return {
        "train": sorted(int(value) for value in permutation[:train_end]),
        "validation": sorted(int(value) for value in permutation[train_end:validation_end]),
        "test": sorted(int(value) for value in permutation[validation_end:]),
    }


def _split_manifest(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    name: str,
    row_ids: list[int],
) -> SplitManifest:
    subset = frame.loc[frame["row_id"].isin(row_ids)]
    if len(subset) != len(row_ids):
        raise DataError(
            f"split {name!r} references {len(row_ids)} row ids but only "
            f"{len(subset)} exist in the dataset"
        )
    targets = subset["target"].to_numpy(dtype=np.int64)
    if not np.isin(targets, (0, 1)).all():
        raise DataError(f"split {name!r} contains a target outside {{0, 1}}")
    return SplitManifest(
        name=name,  # type: ignore[arg-type]
        row_count=len(subset),
        fingerprint=fingerprint_bytes(canonical_csv_bytes(subset, schema)),
        class_counts={
            "0": int(np.count_nonzero(targets == 0)),
            "1": int(np.count_nonzero(targets == 1)),
        },
    )


@dataclass(frozen=True)
class PreparedDataset:
    """Outcome of one preparation, including whether the artifact was reused."""

    manifest: DatasetManifest
    directory: Path
    already_prepared: bool


@dataclass(frozen=True)
class DatasetVerification:
    """Result of independent verification of an existing artifact directory."""

    manifest: DatasetManifest
    splits: dict[str, list[int]]
    schema: DatasetSchema
    directory: Path


def _manifest_json(manifest: DatasetManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _splits_json(splits: dict[str, list[int]]) -> bytes:
    ordered = {name: list(splits[name]) for name in SPLIT_NAMES}
    return (json.dumps(ordered, indent=2) + "\n").encode("utf-8")


def _write_artifact(
    directory: Path,
    payload: bytes,
    manifest: DatasetManifest,
    splits: dict[str, list[int]],
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        (temporary / "dataset.csv").write_bytes(payload)
        (temporary / "manifest.json").write_bytes(_manifest_json(manifest))
        (temporary / "splits.json").write_bytes(_splits_json(splits))
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)
    except OSError as exc:
        raise DataError(f"cannot write dataset artifact {directory}: {exc}") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def prepare_dataset(
    config: DatasetConfig,
    split: SplitConfig,
    out_root: Path = DEFAULT_DATASET_ROOT,
    force: bool = False,
) -> PreparedDataset:
    """Generate, validate, and persist one versioned dataset artifact."""
    source = build_dataset_source(config)
    validate_dataset(source.frame, source.schema)

    payload = canonical_csv_bytes(source.frame, source.schema)
    fingerprint = fingerprint_bytes(payload)
    version = dataset_version_for(config.kind, source.schema.data_schema_version, fingerprint)
    row_ids = source.frame["row_id"].to_numpy(dtype=np.int64)
    splits = assign_splits(row_ids, split)
    manifests = [
        _split_manifest(source.frame, source.schema, name, splits[name]) for name in SPLIT_NAMES
    ]
    manifest = DatasetManifest(
        dataset_version=version,
        source_kind=config.kind,
        seed=config.seed,
        split_seed=split.seed,
        data_schema_version=source.schema.data_schema_version,
        row_count=len(source.frame),
        fingerprint=fingerprint,
        feature_names=list(source.schema.feature_names),
        target_name="target",
        generator_parameters=source.generator_parameters,
        splits=manifests,
        created_at=datetime.now(UTC),
    )

    directory = out_root / version
    if directory.exists() and not force:
        try:
            verification = verify_dataset(directory)
        except (DataError, DatasetValidationError) as exc:
            raise DataError(
                f"existing dataset directory {directory} does not match a regenerated "
                f"artifact: {exc}\nRe-run with --force to rewrite it."
            ) from exc
        if (
            verification.manifest.fingerprint != fingerprint
            or verification.manifest.dataset_version != version
        ):
            raise DataError(
                f"existing dataset directory {directory} holds a different fingerprint "
                f"({verification.manifest.fingerprint}) than the regenerated dataset "
                f"({fingerprint}); re-run with --force to rewrite it"
            )
        return PreparedDataset(
            manifest=verification.manifest,
            directory=directory,
            already_prepared=True,
        )

    _write_artifact(directory, payload, manifest, splits)
    return PreparedDataset(manifest=manifest, directory=directory, already_prepared=False)


def _read_manifest(directory: Path) -> DatasetManifest:
    path = directory / "manifest.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataError(f"{path}: manifest.json is missing") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"{path}: cannot read manifest: {exc}") from exc
    try:
        return DatasetManifest.model_validate_json(text)
    except PydanticValidationError as exc:
        raise DataError(f"{path}: manifest does not match DatasetManifest {exc}") from exc


def _read_splits(directory: Path) -> dict[str, list[int]]:
    path = directory / "splits.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataError(f"{path}: splits.json is missing") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"{path}: cannot read splits.json: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != set(SPLIT_NAMES):
        raise DataError(
            f"{path}: expected an object with exactly {list(SPLIT_NAMES)} keys, "
            f"found {sorted(raw) if isinstance(raw, dict) else type(raw).__name__}"
        )
    splits: dict[str, list[int]] = {}
    for name in SPLIT_NAMES:
        values = raw[name]
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise DataError(f"{path}: {name} must be a list of integer row ids")
        if values != sorted(values):
            raise DataError(f"{path}: {name} row ids are not in ascending canonical order")
        if len(set(values)) != len(values):
            raise DataError(f"{path}: {name} contains duplicate row ids")
        splits[name] = values
    return splits


def verify_dataset(directory: Path) -> DatasetVerification:
    """Verify an artifact directory without trusting any of its contents.

    Recomputes the dataset fingerprint, re-serializes to prove the bytes are
    canonical, re-validates the schema, and recomputes every split fingerprint,
    row count, and class count. Any disagreement is a typed :class:`DataError`.
    """
    if not directory.is_dir():
        raise DataError(f"{directory}: dataset directory does not exist")
    manifest = _read_manifest(directory)
    schema = SCHEMAS_BY_SOURCE_KIND.get(manifest.source_kind)
    if schema is None:
        raise DataError(
            f"{directory}: unsupported source_kind {manifest.source_kind!r} in manifest"
        )
    if list(manifest.feature_names) != list(schema.feature_names):
        raise DataError(
            f"{directory}: manifest feature_names do not match the declared {schema.name!r} schema"
        )
    if manifest.data_schema_version != schema.data_schema_version:
        raise DataError(
            f"{directory}: manifest data_schema_version "
            f"{manifest.data_schema_version!r} does not match the declared "
            f"{schema.data_schema_version!r}"
        )
    expected_version = dataset_version_for(
        manifest.source_kind, schema.data_schema_version, manifest.fingerprint
    )
    if manifest.dataset_version != expected_version:
        raise DataError(
            f"{directory}: manifest dataset_version {manifest.dataset_version!r} does not "
            f"match its fingerprint-derived version {expected_version!r}"
        )
    if manifest.target_name != "target":
        raise DataError(
            f"{directory}: manifest target_name {manifest.target_name!r} is not 'target'"
        )

    dataset_path = directory / "dataset.csv"
    try:
        payload = dataset_path.read_bytes()
    except FileNotFoundError as exc:
        raise DataError(f"{dataset_path}: dataset.csv is missing") from exc
    except OSError as exc:
        raise DataError(f"{dataset_path}: cannot read dataset.csv: {exc}") from exc
    actual_fingerprint = fingerprint_bytes(payload)
    if actual_fingerprint != manifest.fingerprint:
        raise DataError(
            f"{dataset_path}: fingerprint mismatch: bytes hash to {actual_fingerprint}, "
            f"manifest records {manifest.fingerprint}"
        )

    frame = read_dataset_csv(dataset_path, schema)
    if len(frame) != manifest.row_count:
        raise DataError(
            f"{dataset_path}: {len(frame)} rows but manifest records {manifest.row_count}"
        )
    rebuilt = canonical_csv_bytes(frame, schema)
    if rebuilt != payload:
        raise DataError(
            f"{dataset_path}: file is not in canonical serialized form even though its "
            "hash matches the manifest"
        )

    splits = _read_splits(directory)
    all_ids = [row_id for name in SPLIT_NAMES for row_id in splits[name]]
    if len(set(all_ids)) != len(all_ids):
        raise DataError(f"{directory / 'splits.json'}: row ids appear in more than one split")
    dataset_ids = set(int(value) for value in frame["row_id"].tolist())
    if set(all_ids) != dataset_ids:
        missing = len(dataset_ids - set(all_ids))
        unknown = len(set(all_ids) - dataset_ids)
        raise DataError(
            f"{directory / 'splits.json'}: split ids do not cover dataset.csv "
            f"({missing} dataset rows unassigned, {unknown} unknown ids)"
        )

    manifests_by_name = {split.name: split for split in manifest.splits}
    for name in SPLIT_NAMES:
        recorded = manifests_by_name[name]
        if recorded.row_count != len(splits[name]):
            raise DataError(
                f"{directory}: split {name!r} has {len(splits[name])} row ids but the "
                f"manifest records {recorded.row_count}"
            )
        recomputed = _split_manifest(frame, schema, name, splits[name])
        if recomputed.fingerprint != recorded.fingerprint:
            raise DataError(
                f"{directory}: split {name!r} fingerprint mismatch: recomputed "
                f"{recomputed.fingerprint}, manifest records {recorded.fingerprint}"
            )
        if recomputed.class_counts != recorded.class_counts:
            raise DataError(
                f"{directory}: split {name!r} class counts mismatch: recomputed "
                f"{recomputed.class_counts}, manifest records {recorded.class_counts}"
            )
    return DatasetVerification(manifest=manifest, splits=splits, schema=schema, directory=directory)
