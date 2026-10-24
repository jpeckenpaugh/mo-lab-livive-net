"""Declared-schema validation for dataset frames.

Validation is deliberately strict and non-coercing: a numeric token that is not
a plain decimal literal, a float-typed identifier, or a category outside its
declared levels is a violation rather than something pandas may quietly cast
away. All violations are collected before one typed error is raised, so the
operator sees the full damage in one run instead of fixing one cell at a time.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from modelops_lab.dataset_schema import DatasetSchema

MAX_REPORTED_VIOLATIONS = 20
MIN_ROWS = 100
MIN_ROWS_PER_CLASS = 50
BASE_RATE_BOUNDS = (0.20, 0.60)
TARGET_VALUES = (0, 1)

_INT_PATTERN = re.compile(r"[+-]?\d+")
_FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NULL_TOKENS = frozenset({"", "nan", "+nan", "-nan", "null", "none", "na", "n/a"})
_NON_FINITE_TOKENS = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})


class DatasetValidationError(ValueError):
    """Raised when a dataset violates its declared schema.

    ``violations`` holds the first :data:`MAX_REPORTED_VIOLATIONS` messages and
    ``total_violations`` the full count, so long malformed files stay readable.
    """

    def __init__(self, source: str, violations: list[str], total: int) -> None:
        self.source = source
        self.violations = list(violations)
        self.total_violations = total
        lines = [f"{source}: {total} dataset validation violation(s)"]
        lines.extend(f"  - {message}" for message in self.violations)
        hidden = total - len(self.violations)
        if hidden > 0:
            lines.append(f"  ... and {hidden} more")
        super().__init__("\n".join(lines))


class _Collector:
    def __init__(self) -> None:
        self.samples: list[str] = []
        self.total = 0

    def add(self, message: str) -> None:
        self.total += 1
        if len(self.samples) < MAX_REPORTED_VIOLATIONS:
            self.samples.append(message)


def _is_text_dtype(dtype: object) -> bool:
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


def _is_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULL_TOKENS
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _labels(frame: pd.DataFrame) -> list[str]:
    if "row_id" in frame.columns:
        return [str(value) for value in frame["row_id"].tolist()]
    return [f"<row {index}>" for index in range(len(frame))]


def _example(label: str, value: object) -> str:
    return f"row_id={label} value={value!r}"


def _check_columns(frame: pd.DataFrame, schema: DatasetSchema, errors: _Collector) -> None:
    actual = list(frame.columns)
    expected = list(schema.columns)
    missing = [name for name in expected if name not in actual]
    unexpected = [name for name in actual if name not in expected]
    for name in missing:
        errors.add(f"{name}: required column is missing from schema {schema.name!r}")
    for name in unexpected:
        errors.add(f"{name}: unexpected column is not declared by schema {schema.name!r}")
    if not missing and not unexpected and actual != expected:
        errors.add(
            "columns: declared column order is "
            f"[{', '.join(expected)}]; found [{', '.join(actual)}]"
        )


def _check_missing(series: pd.Series, column: str, labels: list[str], errors: _Collector) -> None:
    for index, value in enumerate(series.tolist()):
        if _is_null(value):
            errors.add(f"{column}: missing value (null/NaN): {_example(labels[index], value)}")


def _read_identifiers(
    series: pd.Series, column: str, labels: list[str], errors: _Collector
) -> np.ndarray:
    count = len(series)
    values = np.full(count, np.nan, dtype=float)
    if pd.api.types.is_integer_dtype(series.dtype):
        values = series.to_numpy(dtype=float, na_value=np.nan)
        _check_missing(series, column, labels, errors)
        return values
    if not _is_text_dtype(series.dtype):
        errors.add(
            f"{column}: column must have integer dtype, found {series.dtype} (no silent coercion)"
        )
        return values
    for index, token in enumerate(series.tolist()):
        if _is_null(token):
            errors.add(f"{column}: missing value (null/NaN): {_example(labels[index], token)}")
            continue
        text = str(token)
        if _INT_PATTERN.fullmatch(text) is None:
            errors.add(f"{column}: invalid integer literal: {_example(labels[index], token)}")
            continue
        number = int(text)
        if number < -(2**63) or number > 2**63 - 1:
            errors.add(
                f"{column}: integer literal is outside the signed 64-bit range: "
                f"{_example(labels[index], token)}"
            )
            continue
        values[index] = number
    return values


def _read_numeric(
    series: pd.Series, column: str, labels: list[str], errors: _Collector
) -> np.ndarray:
    count = len(series)
    values = np.full(count, np.nan, dtype=float)
    if pd.api.types.is_float_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype):
        values = series.to_numpy(dtype=float, na_value=np.nan)
        for index in range(count):
            value = values[index]
            if math.isnan(value):
                errors.add(
                    f"{column}: missing value (null/NaN): "
                    f"{_example(labels[index], series.iloc[index])}"
                )
            elif not math.isfinite(value):
                errors.add(f"{column}: non-finite value: {_example(labels[index], value)}")
        return values
    if not _is_text_dtype(series.dtype):
        errors.add(
            f"{column}: column must have a numeric dtype, found {series.dtype} (no silent coercion)"
        )
        return values
    for index, token in enumerate(series.tolist()):
        if _is_null(token):
            errors.add(f"{column}: missing value (null/NaN): {_example(labels[index], token)}")
            continue
        text = str(token)
        if text.strip().lower() in _NON_FINITE_TOKENS:
            errors.add(f"{column}: non-finite value: {_example(labels[index], token)}")
            continue
        if _FLOAT_PATTERN.fullmatch(text) is None:
            errors.add(f"{column}: invalid numeric literal: {_example(labels[index], token)}")
            continue
        value = float(text)
        if not math.isfinite(value):
            errors.add(f"{column}: non-finite value: {_example(labels[index], value)}")
            continue
        values[index] = value
    return values


def _read_categorical(
    series: pd.Series,
    column: str,
    labels: list[str],
    allowed: tuple[str, ...],
    errors: _Collector,
) -> np.ndarray:
    count = len(series)
    values = np.full(count, None, dtype=object)
    if not _is_text_dtype(series.dtype):
        errors.add(
            f"{column}: column must contain string categories, found {series.dtype} "
            f"(allowed levels: {list(allowed)})"
        )
        return values
    for index, token in enumerate(series.tolist()):
        if _is_null(token):
            errors.add(f"{column}: missing value (null/NaN): {_example(labels[index], token)}")
            continue
        text = str(token)
        if text not in allowed:
            errors.add(
                f"{column}: unknown category; allowed levels are {list(allowed)}: "
                f"{_example(labels[index], token)}"
            )
            continue
        values[index] = text
    return values


def _check_row_ids(values: np.ndarray, labels: list[str], errors: _Collector) -> None:
    finite = ~np.isnan(values)
    for index in np.flatnonzero(finite & (values <= 0)):
        errors.add(f"row_id: value must be positive: {_example(labels[index], values[index])}")
    if finite.all():
        unique, counts = np.unique(values, return_counts=True)
        for value, count in zip(unique[counts > 1], counts[counts > 1], strict=True):
            first = int(np.flatnonzero(values == value)[0])
            errors.add(
                f"row_id: duplicate value {int(value)} appears {int(count)} times: "
                f"{_example(labels[first], value)}"
            )


def _check_target(
    values: np.ndarray, row_count: int, labels: list[str], errors: _Collector
) -> None:
    valid = ~np.isnan(values)
    invalid = np.flatnonzero(valid & ~np.isin(values, TARGET_VALUES))
    for index in invalid:
        errors.add(
            f"target: value {values[index]:g} is not in {{0, 1}}: "
            f"{_example(labels[index], values[index])}"
        )
    if len(invalid):
        return
    counts = {value: int(np.count_nonzero(values == value)) for value in TARGET_VALUES}
    for value, count in counts.items():
        if count < MIN_ROWS_PER_CLASS:
            errors.add(
                f"target: class {value} has {count} rows; "
                f"the minimum per class is {MIN_ROWS_PER_CLASS}"
            )
    if valid.all() and row_count:
        base_rate = counts[1] / row_count
        low, high = BASE_RATE_BOUNDS
        if not low <= base_rate <= high:
            errors.add(
                f"target: base rate {base_rate:.4f} is outside the allowed "
                f"range [{low:.2f}, {high:.2f}]"
            )


def _check_ranges(
    columns: dict[str, np.ndarray], schema: DatasetSchema, labels: list[str], errors: _Collector
) -> None:
    for name in schema.numeric_features:
        values = columns.get(name)
        if values is None:
            continue
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = schema.numeric_ranges[name]
        outside = np.flatnonzero(finite & ((values < low) | (values > high)))
        for index in outside:
            errors.add(
                f"{name}: value {values[index]:g} is outside the declared range "
                f"[{low:g}, {high:g}]: {_example(labels[index], values[index])}"
            )


def collect_dataset_violations(frame: pd.DataFrame, schema: DatasetSchema) -> tuple[list[str], int]:
    """Return the reported violations and the total count for ``frame``."""
    errors = _Collector()
    _check_columns(frame, schema, errors)
    if len(frame) < MIN_ROWS:
        errors.add(f"dataset: has {len(frame)} rows; the minimum is {MIN_ROWS}")
    labels = _labels(frame)
    columns: dict[str, np.ndarray] = {}
    for name in schema.columns:
        if name not in frame.columns:
            continue
        series = frame[name]
        kind = schema.kind_of(name)
        if kind in {"row_id", "target"}:
            columns[name] = _read_identifiers(series, name, labels, errors)
        elif kind == "numeric":
            columns[name] = _read_numeric(series, name, labels, errors)
        else:
            columns[name] = _read_categorical(
                series, name, labels, schema.categorical_levels[name], errors
            )
    if "row_id" in columns:
        _check_row_ids(columns["row_id"], labels, errors)
    if "target" in columns:
        _check_target(columns["target"], len(frame), labels, errors)
    _check_ranges(columns, schema, labels, errors)
    return errors.samples, errors.total


def validate_dataset(frame: pd.DataFrame, schema: DatasetSchema) -> None:
    """Validate ``frame`` against ``schema`` and raise one typed error on failure."""
    violations, total = collect_dataset_violations(frame, schema)
    if total:
        raise DatasetValidationError(schema.name, violations, total)


def parse_dataset(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Validate a raw (string) frame and return a typed copy.

    The typed copy uses ``int64`` for ``row_id``/``target``, ``float64`` for
    numeric features, and text for categorical features. No value is coerced
    implicitly: every conversion is preceded by a declaration check.
    """
    violations, total = collect_dataset_violations(frame, schema)
    if total:
        raise DatasetValidationError(schema.name, violations, total)
    typed = pd.DataFrame(index=frame.index)
    typed["row_id"] = frame["row_id"].astype("int64")
    typed["target"] = frame["target"].astype("int64")
    for name in schema.numeric_features:
        typed[name] = frame[name].astype("float64")
    for name in schema.categorical_features:
        typed[name] = frame[name].astype("str")
    return typed[list(schema.columns)]


def read_dataset_csv(path: Path, schema: DatasetSchema) -> pd.DataFrame:
    """Read a CSV as raw text, validate it, and return a typed frame.

    Reading every cell as text first is what makes the dtype rules meaningful:
    pandas cannot infer ``int64`` from a categorical column that happens to
    contain numeric-looking codes, and numeric literals are parsed by us.
    """
    try:
        raw = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise DatasetValidationError(str(path), [f"dataset file not found: {path}"], 1) from exc
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise DatasetValidationError(str(path), [f"cannot read dataset CSV: {exc}"], 1) from exc
    if raw.empty:
        raise DatasetValidationError(str(path), ["dataset contains no rows"], 1)
    return parse_dataset(raw, schema)
