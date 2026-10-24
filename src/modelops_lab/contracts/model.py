"""Model descriptor contract for trained and approved model versions."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from modelops_lab.contracts.base import ContractModel, Sha256Hex, VersionedContract


class ConfusionMatrix(ContractModel):
    """Binary confusion matrix counts at a fixed decision threshold."""

    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_positive: int = Field(ge=0)


class EvaluationMetrics(ContractModel):
    """Metrics and sample accounting for one evaluation partition."""

    roc_auc: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=1, description="Rows included in this metric block.")
    class_counts: dict[str, int] = Field(
        min_length=1, description="Stringified observed label to row count."
    )
    confusion_matrix: ConfusionMatrix

    @model_validator(mode="after")
    def _sample_counts_agree(self) -> Self:
        matrix_total = (
            self.confusion_matrix.true_negative
            + self.confusion_matrix.false_positive
            + self.confusion_matrix.false_negative
            + self.confusion_matrix.true_positive
        )
        if matrix_total != self.sample_count:
            raise ValueError(
                f"confusion matrix totals {matrix_total}, but sample_count is {self.sample_count}"
            )
        class_total = sum(self.class_counts.values())
        if any(count < 0 for count in self.class_counts.values()):
            raise ValueError("class_counts values must be non-negative")
        if class_total != self.sample_count:
            raise ValueError(
                f"class_counts total {class_total}, but sample_count is {self.sample_count}"
            )
        return self


class ApprovalMetadata(ContractModel):
    """Record of the human approval that made an artifact servable."""

    approved_by: str = Field(min_length=1)
    approved_at: AwareDatetime
    decision_id: str = Field(min_length=1, description="Promotion decision that authorized approval.")
    reason: str = Field(min_length=1)
    registry_alias: str = Field(min_length=1, description="MLflow alias that marks the artifact.")


class ModelDescriptor(VersionedContract):
    """Everything needed to identify, audit, and load a trained model version."""

    model_version: str = Field(min_length=1, description="Version label written to the registry.")
    mlflow_run_id: str = Field(min_length=1)
    dataset_fingerprint: Sha256Hex
    artifact_uri: str = Field(min_length=1, description="Resolvable MLflow artifact URI.")
    evaluation_split: Literal["validation", "test"] = Field(
        description="Partition the metrics were computed on; promotion uses validation."
    )
    metrics: EvaluationMetrics
    approval: ApprovalMetadata | None = Field(
        default=None,
        description="Absent for candidates that have not been approved for serving.",
    )
    created_at: AwareDatetime
