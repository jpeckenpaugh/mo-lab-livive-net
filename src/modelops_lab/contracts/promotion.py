"""Promotion decision contract recording every gate check and its outcome."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from modelops_lab.contracts.base import ContractModel, VersionedContract


class PromotionVerdict(StrEnum):
    """Outcome of an explicit, audited promotion operation."""

    APPROVED = "approved"
    REJECTED = "rejected"


class PromotionCheck(ContractModel):
    """One named gate check with a machine-readable pass flag and human detail."""

    name: str = Field(min_length=1, description="Stable check identifier, unique per decision.")
    passed: bool
    detail: str = Field(min_length=1, description="Human-readable result, including values compared.")
    observed_value: float | None = Field(
        default=None, description="Numeric value observed by a threshold check, when applicable."
    )
    threshold_value: float | None = Field(
        default=None, description="Threshold the observed value was compared against, when applicable."
    )


class PromotionDecision(VersionedContract):
    """Auditable record of a candidate-versus-active promotion evaluation.

    The contract enforces consistency: an approved decision cannot contain a
    failed check, and a rejected decision must state at least one reason.
    """

    decision_id: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    active_version: str | None = Field(
        default=None, description="Version active before this decision; null for first promotion."
    )
    checks: list[PromotionCheck] = Field(min_length=1)
    verdict: PromotionVerdict
    rejection_reasons: list[str] = Field(default_factory=list)
    actor: str = Field(min_length=1, description="Human or service that executed the decision.")
    reason: str = Field(min_length=1, description="Why the actor made this decision.")
    decided_at: AwareDatetime

    @field_validator("checks")
    @classmethod
    def _check_names_are_unique(cls, value: list[PromotionCheck]) -> list[PromotionCheck]:
        names = [check.name for check in value]
        if len(set(names)) != len(names):
            raise ValueError("check names must be unique within a promotion decision")
        return value

    @field_validator("rejection_reasons")
    @classmethod
    def _rejection_reasons_are_meaningful(cls, value: list[str]) -> list[str]:
        if any(not reason.strip() for reason in value):
            raise ValueError("rejection_reasons must not contain empty strings")
        if len(set(value)) != len(value):
            raise ValueError("rejection_reasons must be unique")
        return value

    @model_validator(mode="after")
    def _verdict_is_consistent(self) -> Self:
        failed = [check.name for check in self.checks if not check.passed]
        if self.verdict is PromotionVerdict.APPROVED:
            if failed:
                raise ValueError(
                    "an approved decision cannot contain failed checks: " + ", ".join(failed)
                )
            if self.rejection_reasons:
                raise ValueError("an approved decision must not include rejection_reasons")
        elif self.verdict is PromotionVerdict.REJECTED and not self.rejection_reasons:
            raise ValueError("a rejected decision must include at least one rejection reason")
        return self
