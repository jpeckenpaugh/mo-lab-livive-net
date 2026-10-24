"""Determinism, canonical serialization, splits, regimes, and loader tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelops_lab.config import load_training_config
from modelops_lab.contracts import DatasetManifest
from modelops_lab.data import (
    SPLIT_NAMES,
    assign_splits,
    canonical_csv_bytes,
    generate_synthetic,
    label_probabilities,
    load_breast_cancer_dataset,
    prepare_dataset,
    split_row_counts,
)
from modelops_lab.dataset_schema import (
    BREAST_CANCER_SCHEMA,
    SYNTHETIC_CATEGORICAL_LEVELS,
    SYNTHETIC_NUMERIC_FEATURES,
    SYNTHETIC_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[2]
TRAINING = load_training_config(ROOT / "configs" / "training.yaml")


def dataset_config(**overrides: object):
    return TRAINING.dataset.model_copy(update=overrides)


def split_config(**overrides: object):
    return TRAINING.split.model_copy(update=overrides)


def prepare(tmp_path: Path, name: str = "artifacts", **overrides: object):
    return prepare_dataset(dataset_config(**overrides), TRAINING.split, out_root=tmp_path / name)


def subset(frame: pd.DataFrame, row_ids: list[int]) -> pd.DataFrame:
    return frame.loc[frame["row_id"].isin(row_ids)]


def logistic_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(SYNTHETIC_NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(SYNTHETIC_CATEGORICAL_LEVELS),
            ),
        ]
    )
    return make_pipeline(
        preprocessing,
        LogisticRegression(max_iter=1000, random_state=42),
    )


def fit_and_score(train_frame: pd.DataFrame, evaluation_frame: pd.DataFrame) -> float:
    columns = list(SYNTHETIC_SCHEMA.feature_names)
    model = logistic_pipeline()
    model.fit(train_frame[columns], train_frame["target"])
    probabilities = model.predict_proba(evaluation_frame[columns])[:, 1]
    return float(roc_auc_score(evaluation_frame["target"], probabilities))


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_share = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), 1e-6, None)
    current_share = np.clip(np.histogram(current, bins=edges)[0] / len(current), 1e-6, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def test_identical_inputs_produce_identical_artifacts(tmp_path: Path) -> None:
    first = prepare(tmp_path, "first")
    second = prepare(tmp_path, "second")

    assert first.manifest.fingerprint == second.manifest.fingerprint
    assert first.manifest.dataset_version == second.manifest.dataset_version
    assert first.directory.name == second.directory.name
    for name in SPLIT_NAMES:
        first_split = next(split for split in first.manifest.splits if split.name == name)
        second_split = next(split for split in second.manifest.splits if split.name == name)
        assert first_split.fingerprint == second_split.fingerprint
        assert first_split.row_count == second_split.row_count
    assert (first.directory / "dataset.csv").read_bytes() == (
        second.directory / "dataset.csv"
    ).read_bytes()
    assert (first.directory / "splits.json").read_bytes() == (
        second.directory / "splits.json"
    ).read_bytes()
    # created_at is the only intentionally non-deterministic manifest field.
    assert first.manifest.model_dump(exclude={"created_at"}) == second.manifest.model_dump(
        exclude={"created_at"}
    )


def test_seed_and_regime_change_the_fingerprint(tmp_path: Path) -> None:
    fingerprints = {
        prepare(tmp_path, "baseline", seed=42).manifest.fingerprint,
        prepare(tmp_path, "other-seed", seed=43).manifest.fingerprint,
        prepare(tmp_path, "shifted", regime="covariate_shift").manifest.fingerprint,
        prepare(tmp_path, "improved", regime="improved_signal").manifest.fingerprint,
    }

    assert len(fingerprints) == 4


def test_dataset_version_is_human_readable_and_deterministic(tmp_path: Path) -> None:
    result = prepare(tmp_path)

    assert result.manifest.dataset_version == f"synthetic-v1.0.0-{result.manifest.fingerprint[:12]}"


def test_canonical_serialization_is_row_order_independent() -> None:
    frame = generate_synthetic(300, seed=11)
    shuffled = frame.sample(frac=1.0, random_state=3, ignore_index=True)

    assert canonical_csv_bytes(frame, SYNTHETIC_SCHEMA) == canonical_csv_bytes(
        shuffled, SYNTHETIC_SCHEMA
    )


def test_canonical_serialization_uses_lf_and_declared_header() -> None:
    payload = canonical_csv_bytes(generate_synthetic(120, seed=5), SYNTHETIC_SCHEMA)

    assert b"\r" not in payload
    text = payload.decode("utf-8")
    assert text.endswith("\n")
    assert text.splitlines()[0] == ",".join(SYNTHETIC_SCHEMA.columns)


def test_float_formatting_round_trips_exactly() -> None:
    frame = generate_synthetic(200, seed=7)
    tricky = [0.1, 1 / 3, 1e-300, 5e-324, 123456789.123456789, -0.0]
    for index, value in enumerate(tricky):
        frame.loc[index, "bmi"] = value

    lines = canonical_csv_bytes(frame, SYNTHETIC_SCHEMA).decode("utf-8").splitlines()
    positions = {name: index for index, name in enumerate(lines[0].split(","))}
    rows = {int(parts[0]): parts for parts in (line.split(",") for line in lines[1:])}
    for index, value in enumerate(tricky):
        token = rows[index + 1][positions["bmi"]]
        assert float(token) == value
        assert math.copysign(1.0, float(token)) == math.copysign(1.0, value)


def test_splits_are_disjoint_complete_and_exactly_sized() -> None:
    frame = generate_synthetic(2000, seed=42)
    row_ids = frame["row_id"].to_numpy()

    splits = assign_splits(row_ids, TRAINING.split)

    assert {name: len(ids) for name, ids in splits.items()} == {
        "train": 1200,
        "validation": 400,
        "test": 400,
    }
    combined = [row_id for name in SPLIT_NAMES for row_id in splits[name]]
    assert len(combined) == len(set(combined)) == 2000
    assert set(combined) == set(int(value) for value in row_ids)


def test_split_counts_use_largest_remainder() -> None:
    assert split_row_counts(2001, TRAINING.split) == {
        "train": 1201,
        "validation": 400,
        "test": 400,
    }


def test_splits_are_reproducible_and_seed_sensitive() -> None:
    row_ids = np.arange(1, 2001, dtype=np.int64)

    first = assign_splits(row_ids, TRAINING.split)
    second = assign_splits(row_ids, TRAINING.split)
    different_seed = assign_splits(row_ids, split_config(seed=43))

    assert first == second
    assert first["validation"] != different_seed["validation"]


def test_artifacts_contain_three_verifiable_files(tmp_path: Path) -> None:
    result = prepare(tmp_path)

    for name in ("dataset.csv", "manifest.json", "splits.json"):
        assert (result.directory / name).is_file()
    manifest = DatasetManifest.model_validate_json(
        (result.directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == result.manifest
    assert (
        manifest.fingerprint
        == hashlib.sha256((result.directory / "dataset.csv").read_bytes()).hexdigest()
    )
    for split in manifest.splits:
        # Zero-count classes are recorded explicitly, never omitted.
        assert set(split.class_counts) == {"0", "1"}
        assert sum(split.class_counts.values()) == split.row_count
    persisted = json.loads((result.directory / "splits.json").read_text(encoding="utf-8"))
    assert set(persisted) == set(SPLIT_NAMES)
    assert sum(len(ids) for ids in persisted.values()) == 2000


def test_repeated_prepare_is_idempotent(tmp_path: Path) -> None:
    first = prepare(tmp_path)
    second = prepare(tmp_path)

    assert second.already_prepared is True
    assert second.manifest == first.manifest
    assert second.directory == first.directory


def test_breast_cancer_loader_shape_and_labels() -> None:
    frame = load_breast_cancer_dataset()

    assert tuple(frame.columns) == BREAST_CANCER_SCHEMA.columns
    assert len(frame) == 569
    assert set(frame["target"].unique()) == {0, 1}
    assert int(frame["target"].sum()) == 212
    assert list(frame["row_id"]) == list(range(1, 570))


def test_breast_cancer_artifacts_are_deterministic_and_provenanced(tmp_path: Path) -> None:
    first = prepare(tmp_path, "first", kind="breast_cancer", n_samples=1000)
    second = prepare(tmp_path, "second", kind="breast_cancer", n_samples=1000)

    assert first.manifest.source_kind == "breast_cancer"
    assert first.manifest.row_count == 569
    assert first.manifest.fingerprint == second.manifest.fingerprint
    assert first.manifest.generator_parameters["loader"].startswith(
        "sklearn.datasets.load_breast_cancer"
    )
    assert "Wisconsin" in first.manifest.generator_parameters["provenance"]
    assert "regime" not in first.manifest.generator_parameters or (
        first.manifest.generator_parameters["regime"] == "not_applicable"
    )
    assert sum(split.row_count for split in first.manifest.splits) == 569


def test_breast_cancer_rejects_shift_regimes(tmp_path: Path) -> None:
    from modelops_lab.data import DataError

    with pytest.raises(DataError, match="only supported for the synthetic generator"):
        prepare(tmp_path, kind="breast_cancer", regime="covariate_shift")


def test_baseline_is_learnable() -> None:
    frame = generate_synthetic(2000, seed=42, regime="baseline")
    splits = assign_splits(frame["row_id"].to_numpy(), TRAINING.split)

    auc = fit_and_score(subset(frame, splits["train"]), subset(frame, splits["validation"]))

    assert 0.82 <= auc <= 0.98


def test_improved_signal_beats_a_baseline_model_on_the_same_slice() -> None:
    baseline = generate_synthetic(2000, seed=42, regime="baseline")
    improved = generate_synthetic(2000, seed=42, regime="improved_signal")
    baseline_splits = assign_splits(baseline["row_id"].to_numpy(), TRAINING.split)
    improved_splits = assign_splits(improved["row_id"].to_numpy(), TRAINING.split)

    # P(X) is held fixed: only the label mechanism differs.
    assert_frame_equal(
        baseline.drop(columns="target"), improved.drop(columns="target"), check_exact=True
    )

    validation = subset(improved, improved_splits["validation"])
    baseline_model_auc = fit_and_score(subset(baseline, baseline_splits["train"]), validation)
    improved_model_auc = fit_and_score(subset(improved, improved_splits["train"]), validation)

    assert improved_model_auc - baseline_model_auc >= 0.02


def test_covariate_shift_moves_several_features_but_not_the_mechanism() -> None:
    baseline = generate_synthetic(2000, seed=42, regime="baseline")
    shifted = generate_synthetic(2000, seed=42, regime="covariate_shift")

    drifted = [
        name
        for name in SYNTHETIC_NUMERIC_FEATURES
        if population_stability_index(
            baseline[name].to_numpy(dtype=float), shifted[name].to_numpy(dtype=float)
        )
        > 0.20
    ]
    assert len(drifted) >= 6

    # The conditional mechanism is identical for the same feature row.
    np.testing.assert_array_equal(
        label_probabilities(baseline, "baseline"),
        label_probabilities(baseline, "covariate_shift"),
    )

    # ... while the improved regime deliberately changes it.
    assert not np.array_equal(
        label_probabilities(baseline, "baseline"),
        label_probabilities(baseline, "improved_signal"),
    )

    splits = assign_splits(baseline["row_id"].to_numpy(), TRAINING.split)
    shifted_splits = assign_splits(shifted["row_id"].to_numpy(), TRAINING.split)
    train = subset(baseline, splits["train"])
    baseline_auc = fit_and_score(train, subset(baseline, splits["validation"]))
    shifted_auc = fit_and_score(train, subset(shifted, shifted_splits["validation"]))

    assert shifted_auc < baseline_auc - 0.02
    assert shifted_auc > 0.70
    assert 0.35 <= float(shifted["target"].mean()) <= 0.60
