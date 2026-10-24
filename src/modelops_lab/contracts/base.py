"""Shared primitives for the frozen contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SchemaVersion = Literal["1.0.0"]
SCHEMA_VERSION: SchemaVersion = "1.0.0"

DatasetKind = Literal["synthetic", "breast_cancer"]

Sha256Hex = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase hexadecimal SHA-256 digest.",
    ),
]


class ContractModel(BaseModel):
    """Base class for immutable, strict cross-wave value objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class VersionedContract(ContractModel):
    """Base class for the five top-level contracts exported as JSON Schema."""

    schema_version: SchemaVersion = Field(
        default=SCHEMA_VERSION,
        description="Version of the contract schema that produced this record.",
    )
