"""Frozen, versioned data contracts shared across lifecycle waves.

Each contract inherits :class:`ContractModel`, which rejects unknown fields and
carries a schema version. Schema changes require a PM decision and a version
bump; regenerated JSON Schemas live in the repository-level ``contracts/``
directory.
"""

from modelops_lab.contracts.base import (
    ContractModel,
    DatasetKind,
    SchemaVersion,
    Sha256Hex,
    VersionedContract,
)
from modelops_lab.contracts.dataset import DatasetManifest, SplitManifest
from modelops_lab.contracts.drift import (
    DriftReport,
    DriftThresholds,
    DriftVerdict,
    FeatureDriftStatistic,
    PerformanceSnapshot,
    SignalVerdict,
    WindowSummary,
)
from modelops_lab.contracts.model import (
    ApprovalMetadata,
    ConfusionMatrix,
    EvaluationMetrics,
    ModelDescriptor,
)
from modelops_lab.contracts.prediction import PredictionEvent
from modelops_lab.contracts.promotion import PromotionCheck, PromotionDecision, PromotionVerdict

__all__ = [
    "ApprovalMetadata",
    "ConfusionMatrix",
    "ContractModel",
    "DatasetKind",
    "DatasetManifest",
    "DriftReport",
    "DriftThresholds",
    "DriftVerdict",
    "EvaluationMetrics",
    "FeatureDriftStatistic",
    "ModelDescriptor",
    "PerformanceSnapshot",
    "PredictionEvent",
    "PromotionCheck",
    "PromotionDecision",
    "PromotionVerdict",
    "SchemaVersion",
    "Sha256Hex",
    "SignalVerdict",
    "SplitManifest",
    "VersionedContract",
    "WindowSummary",
]
