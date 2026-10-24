"""Validation rules, violation collection, and CSV coercion tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from modelops_lab.data import canonical_csv_bytes, generate_synthetic
from modelops_lab.dataset_schema import SYNTHETIC_SCHEMA
from modelops_lab.validation import (
    DatasetValidationError,
    read_dataset_csv,
    validate_dataset,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return generate_synthetic(400, seed=42)


def violation_message(frame: pd.DataFrame) -> str:
    with pytest.raises(DatasetValidationError) as error:
        validate_dataset(frame, SYNTHETIC_SCHEMA)
    return str(error.value)


def test_valid_generated_frame_passes(frame: pd.DataFrame) -> None:
    validate_dataset(frame, SYNTHETIC_SCHEMA)


def test_missing_column_is_named(frame: pd.DataFrame) -> None:
    message = violation_message(frame.drop(columns=["bmi"]))
    assert "bmi" in message
    assert "required column is missing" in message


def test_extra_column_is_named(frame: pd.DataFrame) -> None:
    message = violation_message(frame.assign(surprise=1))
    assert "surprise" in message
    assert "unexpected column" in message


def test_reordered_columns_are_rejected(frame: pd.DataFrame) -> None:
    reordered = frame[list(reversed(SYNTHETIC_SCHEMA.columns))]
    message = violation_message(reordered)
    assert "declared column order" in message


def test_wrong_identifier_dtype_is_rejected_without_coercion(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken["target"] = broken["target"].astype("float64")
    message = violation_message(broken)
    assert "target" in message
    assert "integer dtype" in message


def test_numeric_string_is_rejected_when_read_from_csv() -> None:
    frame = generate_synthetic(200, seed=8)
    payload = canonical_csv_bytes(frame, SYNTHETIC_SCHEMA).decode("utf-8")
    lines = payload.splitlines()
    header = lines[0].split(",")
    age_position = header.index("age")
    parts = lines[1].split(",")
    parts[age_position] = "not-a-number"
    lines[1] = ",".join(parts)
    # Write the corrupted bytes through the strict reader.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "dataset.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(DatasetValidationError) as error:
            read_dataset_csv(path, SYNTHETIC_SCHEMA)
    message = str(error.value)
    assert "age" in message
    assert "invalid numeric literal" in message


def test_empty_csv_is_rejected_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_bytes(b"")

    with pytest.raises(DatasetValidationError, match="cannot read dataset CSV"):
        read_dataset_csv(path, SYNTHETIC_SCHEMA)


def test_nan_is_rejected_with_the_column_name(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[5, "glucose"] = np.nan
    message = violation_message(broken)
    assert "glucose" in message
    assert "missing value" in message


def test_infinity_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[5, "bmi"] = np.inf
    message = violation_message(broken)
    assert "bmi" in message
    assert "non-finite" in message


def test_non_binary_target_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[5, "target"] = 2
    message = violation_message(broken)
    assert "target" in message
    assert "not in {0, 1}" in message


def test_out_of_range_numeric_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[5, "age"] = 250
    message = violation_message(broken)
    assert "age" in message
    assert "outside the declared range" in message


def test_unknown_category_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[5, "sex"] = "unknown"
    message = violation_message(broken)
    assert "sex" in message
    assert "unknown category" in message


def test_class_imbalance_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken["target"] = 0
    broken.loc[:39, "target"] = 1
    message = violation_message(broken)
    assert "target" in message
    assert "minimum per class" in message


def test_base_rate_outside_bounds_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken["target"] = 1
    message = violation_message(broken)
    assert "target" in message
    assert "base rate" in message


def test_too_few_rows_is_rejected() -> None:
    small = generate_synthetic(80, seed=4)
    message = violation_message(small)
    assert "minimum is 100" in message


def test_duplicate_row_ids_are_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[10, "row_id"] = broken.loc[9, "row_id"]
    message = violation_message(broken)
    assert "row_id" in message
    assert "duplicate" in message


def test_non_positive_row_id_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[10, "row_id"] = 0
    message = violation_message(broken)
    assert "row_id" in message
    assert "must be positive" in message


def test_violations_are_capped_with_a_remainder_line(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[:39, "age"] = 250

    with pytest.raises(DatasetValidationError) as error:
        validate_dataset(broken, SYNTHETIC_SCHEMA)

    assert error.value.total_violations >= 40
    assert len(error.value.violations) == 20
    assert "... and 20 more" in str(error.value)


def test_offending_example_includes_row_id_and_value(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[11, "cholesterol"] = 999
    message = violation_message(broken)
    assert "row_id=" in message
    assert "999" in message
