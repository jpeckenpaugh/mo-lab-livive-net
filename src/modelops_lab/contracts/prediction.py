"""Prediction event contract persisted for monitoring and audit."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from modelops_lab.contracts.base import VersionedContract


class PredictionEvent(VersionedContract):
    """One inference request with the version and latency needed to audit serving."""

    request_id: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(description="UTC-aware time the request was handled.")
    features: dict[str, float | str] = Field(
        min_length=1,
        description=(
            "Validated model input features keyed by training column name. "
            "Numeric features are floats; categorical features are their known string level."
        ),
    )
    prediction: Literal[0, 1] = Field(description="Binary class label returned to the caller.")
    probability: float = Field(
        ge=0.0, le=1.0, description="Probability assigned to the positive class."
    )
    model_version: str = Field(min_length=1, description="Approved model version that served the request.")
    latency_ms: float = Field(ge=0.0, description="Server-side inference latency in milliseconds.")
