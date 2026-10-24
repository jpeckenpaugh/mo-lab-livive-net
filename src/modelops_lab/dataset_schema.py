"""Declared dataset schemas shared by the generator and the validator.

A schema fixes the exact column set and order, the column kind (identifier,
numeric feature, categorical feature, target), the plausible numeric ranges,
and the allowed categorical levels. Nothing here is inferred from data: the
validator compares an incoming frame against these declarations so that a
missing, extra, reordered, or retyped column fails loudly instead of being
silently coerced.

The synthetic schema is the primary workload (ADR-001). The breast-cancer
schema describes the optional loader and deliberately declares wide, documented
ranges so that a corrupted export is still caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnKind = Literal["row_id", "numeric", "categorical", "target"]

DATA_SCHEMA_VERSION = "1.0.0"
ROW_ID_COLUMN = "row_id"
TARGET_COLUMN = "target"

SYNTHETIC_NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "glucose",
    "resting_heart_rate",
    "max_heart_rate",
    "st_depression",
    "exercise_minutes",
)

SYNTHETIC_CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "sex": ("female", "male"),
    "chest_pain_type": ("typical", "atypical", "non_anginal", "asymptomatic"),
    "smoker": ("never", "former", "current"),
}

SYNTHETIC_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (18.0, 90.0),
    "bmi": (15.0, 55.0),
    "systolic_bp": (80.0, 220.0),
    "diastolic_bp": (40.0, 140.0),
    "cholesterol": (100.0, 400.0),
    "glucose": (50.0, 300.0),
    "resting_heart_rate": (40.0, 140.0),
    "max_heart_rate": (60.0, 220.0),
    "st_depression": (0.0, 7.0),
    "exercise_minutes": (0.0, 300.0),
}

BREAST_CANCER_NUMERIC_FEATURES: tuple[str, ...] = (
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
)

# Upper bounds are the observed WDBC maxima with headroom; all WDBC features
# are non-negative measurements. A value above these bounds is a corrupt row.
BREAST_CANCER_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "mean radius": (0.0, 31.0),
    "mean texture": (0.0, 43.3),
    "mean perimeter": (0.0, 207.4),
    "mean area": (0.0, 2751.2),
    "mean smoothness": (0.0, 0.2),
    "mean compactness": (0.0, 0.4),
    "mean concavity": (0.0, 0.5),
    "mean concave points": (0.0, 0.3),
    "mean symmetry": (0.0, 0.4),
    "mean fractal dimension": (0.0, 0.2),
    "radius error": (0.0, 3.2),
    "texture error": (0.0, 5.4),
    "perimeter error": (0.0, 24.2),
    "area error": (0.0, 596.5),
    "smoothness error": (0.0, 0.1),
    "compactness error": (0.0, 0.2),
    "concavity error": (0.0, 0.5),
    "concave points error": (0.0, 0.1),
    "symmetry error": (0.0, 0.1),
    "fractal dimension error": (0.0, 0.1),
    "worst radius": (0.0, 39.7),
    "worst texture": (0.0, 54.5),
    "worst perimeter": (0.0, 276.4),
    "worst area": (0.0, 4679.5),
    "worst smoothness": (0.0, 0.3),
    "worst compactness": (0.0, 1.2),
    "worst concavity": (0.0, 1.4),
    "worst concave points": (0.0, 0.4),
    "worst symmetry": (0.0, 0.8),
    "worst fractal dimension": (0.0, 0.3),
}


@dataclass(frozen=True)
class DatasetSchema:
    """Declared shape and constraints for one dataset source."""

    name: str
    data_schema_version: str
    numeric_features: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    numeric_ranges: dict[str, tuple[float, float]]

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(self.categorical_levels)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    @property
    def columns(self) -> tuple[str, ...]:
        return (ROW_ID_COLUMN, *self.feature_names, TARGET_COLUMN)

    def kind_of(self, column: str) -> ColumnKind:
        if column == ROW_ID_COLUMN:
            return "row_id"
        if column == TARGET_COLUMN:
            return "target"
        if column in self.numeric_ranges:
            return "numeric"
        if column in self.categorical_levels:
            return "categorical"
        raise KeyError(f"column {column!r} is not declared by schema {self.name!r}")


SYNTHETIC_SCHEMA = DatasetSchema(
    name="synthetic",
    data_schema_version=DATA_SCHEMA_VERSION,
    numeric_features=SYNTHETIC_NUMERIC_FEATURES,
    categorical_levels=SYNTHETIC_CATEGORICAL_LEVELS,
    numeric_ranges=SYNTHETIC_NUMERIC_RANGES,
)

BREAST_CANCER_SCHEMA = DatasetSchema(
    name="breast_cancer",
    data_schema_version=DATA_SCHEMA_VERSION,
    numeric_features=BREAST_CANCER_NUMERIC_FEATURES,
    categorical_levels={},
    numeric_ranges=BREAST_CANCER_NUMERIC_RANGES,
)

SCHEMAS_BY_SOURCE_KIND: dict[str, DatasetSchema] = {
    "synthetic": SYNTHETIC_SCHEMA,
    "breast_cancer": BREAST_CANCER_SCHEMA,
}
