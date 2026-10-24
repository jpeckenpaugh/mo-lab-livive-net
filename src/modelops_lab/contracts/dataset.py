"""Dataset manifest contract recorded when a dataset version is prepared."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from modelops_lab.contracts.base import ContractModel, DatasetKind, Sha256Hex, VersionedContract


class SplitManifest(ContractModel):
    """Fingerprint and class balance for one deterministic dataset partition."""

    name: Literal["train", "validation", "test"]
    row_count: int = Field(ge=1, description="Number of rows in this partition.")
    fingerprint: Sha256Hex = Field(
        description="SHA-256 of the canonical serialized rows in this partition."
    )
    class_counts: dict[str, int] = Field(
        min_length=1,
        description="Stringified target label to row count, including zero counts when known.",
    )

    @model_validator(mode="after")
    def _class_counts_match_rows(self) -> Self:
        total = sum(self.class_counts.values())
        if total != self.row_count:
            raise ValueError(
                f"class_counts sum to {total}, but row_count is {self.row_count}"
            )
        if any(count < 0 for count in self.class_counts.values()):
            raise ValueError("class_counts values must be non-negative")
        return self


class DatasetManifest(VersionedContract):
    """Immutable record of a generated or loaded dataset and its split manifest.

    ``fingerprint`` covers the complete dataset; each split carries its own
    fingerprint so later waves can detect accidental re-partitioning.
    """

    dataset_version: str = Field(min_length=1, description="Human-readable dataset version.")
    source_kind: DatasetKind = Field(description="Generator or loader that produced the data.")
    seed: int = Field(ge=0, description="Seed used for data generation or sampling.")
    split_seed: int = Field(ge=0, description="Seed used to assign rows to splits.")
    data_schema_version: str = Field(
        min_length=1,
        description="Version of the feature/target column schema, distinct from the contract version.",
    )
    row_count: int = Field(ge=1, description="Total rows across all splits.")
    fingerprint: Sha256Hex = Field(description="SHA-256 of the complete dataset.")
    feature_names: list[str] = Field(min_length=1, description="Ordered model input column names.")
    target_name: str = Field(min_length=1, description="Target column name.")
    generator_parameters: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Source-specific parameters needed to reproduce the dataset.",
    )
    splits: list[SplitManifest] = Field(min_length=3, description="Train, validation, and test splits.")
    created_at: AwareDatetime = Field(description="Time the dataset version was prepared.")

    @field_validator("feature_names")
    @classmethod
    def _feature_names_are_unique_and_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [name.strip() for name in value]
        if any(not name for name in cleaned):
            raise ValueError("feature_names must not contain empty names")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("feature_names must be unique")
        return cleaned

    @model_validator(mode="after")
    def _splits_are_complete_and_consistent(self) -> Self:
        names = [split.name for split in self.splits]
        if sorted(names) != ["test", "train", "validation"]:
            raise ValueError(
                "splits must contain exactly one each of train, validation, and test"
            )
        split_rows = sum(split.row_count for split in self.splits)
        if split_rows != self.row_count:
            raise ValueError(
                f"split row counts sum to {split_rows}, but row_count is {self.row_count}"
            )
        if self.target_name in self.feature_names:
            raise ValueError("target_name must not also appear in feature_names")
        return self
