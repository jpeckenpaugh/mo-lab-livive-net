"""Round-trip, invariant, and enum-validity tests for the frozen contracts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from modelops_lab.contracts import (
    ApprovalMetadata,
    ConfusionMatrix,
    DatasetManifest,
    DriftReport,
    DriftThresholds,
    DriftVerdict,
    EvaluationMetrics,
    FeatureDriftStatistic,
    ModelDescriptor,
    PerformanceSnapshot,
    PredictionEvent,
    PromotionCheck,
    PromotionDecision,
    PromotionVerdict,
    SignalVerdict,
    SplitManifest,
    WindowSummary,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
DATASET_FINGERPRINT = hashlib.sha256(b"dataset-v1").hexdigest()
SPLIT_FINGERPRINT = hashlib.sha256(b"split-v1").hexdigest()


def make_dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_version="synthetic-v1",
        source_kind="synthetic",
        seed=42,
        split_seed=42,
        data_schema_version="1.0.0",
        row_count=1000,
        fingerprint=DATASET_FINGERPRINT,
        feature_names=["feature_0", "feature_1", "feature_2"],
        target_name="target",
        generator_parameters={"n_samples": 1000, "informative_features": 3},
        splits=[
            SplitManifest(
                name="train",
                row_count=600,
                fingerprint=SPLIT_FINGERPRINT,
                class_counts={"0": 360, "1": 240},
            ),
            SplitManifest(
                name="validation",
                row_count=200,
                fingerprint=SPLIT_FINGERPRINT,
                class_counts={"0": 120, "1": 80},
            ),
            SplitManifest(
                name="test",
                row_count=200,
                fingerprint=SPLIT_FINGERPRINT,
                class_counts={"0": 120, "1": 80},
            ),
        ],
        created_at=NOW,
    )


def make_evaluation_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        roc_auc=0.93,
        precision=0.875,
        recall=0.875,
        f1=0.875,
        accuracy=0.9,
        sample_count=200,
        class_counts={"0": 120, "1": 80},
        confusion_matrix=ConfusionMatrix(
            true_negative=110,
            false_positive=10,
            false_negative=10,
            true_positive=70,
        ),
    )


def make_model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        model_version="2",
        mlflow_run_id="0123456789abcdef0123456789abcdef",
        dataset_fingerprint=DATASET_FINGERPRINT,
        artifact_uri="mlflow-artifacts:/1/0123456789abcdef0123456789abcdef/artifacts/model",
        evaluation_split="validation",
        metrics=make_evaluation_metrics(),
        approval=ApprovalMetadata(
            approved_by="release-manager",
            approved_at=NOW,
            decision_id="decision-1",
            reason="Passed all promotion checks.",
            registry_alias="champion",
        ),
        created_at=NOW,
    )


def make_promotion_decision() -> PromotionDecision:
    return PromotionDecision(
        decision_id="decision-1",
        candidate_version="2",
        active_version="1",
        checks=[
            PromotionCheck(
                name="quality_floor.roc_auc",
                passed=True,
                detail="roc_auc 0.93 >= floor 0.85",
                observed_value=0.93,
                threshold_value=0.85,
            ),
            PromotionCheck(
                name="regression.roc_auc",
                passed=True,
                detail="roc_auc drop 0.00 <= tolerance 0.02",
                observed_value=0.0,
                threshold_value=0.02,
            ),
        ],
        verdict=PromotionVerdict.APPROVED,
        rejection_reasons=[],
        actor="release-manager",
        reason="Candidate passed all gates.",
        decided_at=NOW,
    )


def make_prediction_event() -> PredictionEvent:
    return PredictionEvent(
        request_id="req-0001",
        timestamp=NOW,
        features={"feature_0": 0.12, "feature_1": -1.4, "feature_2": 2.0},
        prediction=1,
        probability=0.83,
        model_version="2",
        latency_ms=12.5,
    )


def make_drift_report() -> DriftReport:
    return DriftReport(
        report_id="drift-0001",
        generated_at=NOW,
        model_version="2",
        reference_window=WindowSummary(
            label="reference",
            source="training_split",
            row_count=1000,
        ),
        current_window=WindowSummary(
            label="current",
            source="prediction_events",
            row_count=500,
            start=datetime(2026, 1, 15, 11, 0, tzinfo=UTC),
            end=NOW,
        ),
        feature_statistics=[
            FeatureDriftStatistic(
                feature="feature_0",
                reference_missing_rate=0.0,
                current_missing_rate=0.01,
                psi=0.31,
                ks_statistic=0.12,
                ks_pvalue=0.001,
                drifted=True,
            ),
            FeatureDriftStatistic(
                feature="feature_1",
                reference_missing_rate=0.0,
                current_missing_rate=0.0,
                psi=0.02,
                ks_statistic=0.03,
                ks_pvalue=0.4,
                drifted=False,
            ),
        ],
        thresholds=DriftThresholds(
            psi_threshold=0.2,
            ks_pvalue_threshold=0.05,
            missing_rate_delta_threshold=0.05,
            aggregate_feature_share_threshold=0.3,
            min_sample_size=100,
            performance_degradation_threshold=0.05,
        ),
        data_drift=SignalVerdict(
            verdict=DriftVerdict.DRIFTED,
            detail="1 of 2 features exceeded drift thresholds",
            method="psi+ks",
            score=0.5,
        ),
        concept_drift=SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED,
            detail="No delayed labels were available for concept-drift analysis.",
        ),
        performance_degradation=SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED,
            detail="No delayed labels were available for performance evaluation.",
        ),
        performance=None,
    )


@pytest.mark.parametrize(
    "instance",
    [
        make_dataset_manifest(),
        make_model_descriptor(),
        make_promotion_decision(),
        make_prediction_event(),
        make_drift_report(),
    ],
    ids=lambda value: type(value).__name__,
)
def test_contract_round_trips_through_json(instance: object) -> None:
    payload = instance.model_dump(mode="json")  # type: ignore[attr-defined]
    restored = type(instance).model_validate(payload)
    assert restored == instance


def test_schema_version_is_only_on_top_level_contracts() -> None:
    payload = make_promotion_decision().model_dump(mode="json")
    assert payload["schema_version"] == "1.1.0"
    assert "schema_version" not in payload["checks"][0]
    assert "schema_version" not in make_model_descriptor().model_dump(mode="json")["metrics"]


def test_prediction_event_accepts_categorical_string_levels() -> None:
    payload = make_prediction_event().model_dump(mode="json")
    payload["features"] = {
        "age": 54.0,
        "bmi": 27.3,
        "sex": "female",
        "chest_pain_type": "asymptomatic",
        "smoker": "former",
    }

    event = PredictionEvent.model_validate(payload)

    assert event.schema_version == "1.1.0"
    assert event.features["sex"] == "female"
    assert event.features["age"] == 54.0


def test_1_0_0_dataset_manifest_still_parses() -> None:
    payload = make_dataset_manifest().model_dump(mode="json")
    payload["schema_version"] = "1.0.0"

    manifest = DatasetManifest.model_validate(payload)

    assert manifest.schema_version == "1.0.0"
    assert manifest.dataset_version == "synthetic-v1"


def test_1_0_0_promotion_decision_still_parses() -> None:
    payload = make_promotion_decision().model_dump(mode="json")
    payload["schema_version"] = "1.0.0"

    decision = PromotionDecision.model_validate(payload)

    assert decision.schema_version == "1.0.0"
    assert decision.verdict is PromotionVerdict.APPROVED


def test_schema_version_rejects_an_unknown_version() -> None:
    payload = make_prediction_event().model_dump(mode="json")
    payload["schema_version"] = "2.0.0"

    with pytest.raises(ValidationError, match="schema_version"):
        PredictionEvent.model_validate(payload)


def test_contracts_reject_unknown_fields() -> None:
    payload = make_prediction_event().model_dump(mode="json")
    payload["unexpected"] = "value"
    with pytest.raises(ValidationError, match="unexpected"):
        PredictionEvent.model_validate(payload)


def test_promotion_verdict_enum_rejects_unknown_value() -> None:
    payload = make_promotion_decision().model_dump(mode="json")
    payload["verdict"] = "maybe"
    with pytest.raises(ValidationError, match="verdict"):
        PromotionDecision.model_validate(payload)


def test_drift_verdict_enum_rejects_unknown_value() -> None:
    payload = make_drift_report().model_dump(mode="json")
    payload["concept_drift"]["verdict"] = "probably"
    with pytest.raises(ValidationError, match="concept_drift.verdict"):
        DriftReport.model_validate(payload)


def test_dataset_kind_rejects_unknown_value() -> None:
    payload = make_dataset_manifest().model_dump(mode="json")
    payload["source_kind"] = "csv"
    with pytest.raises(ValidationError, match="source_kind"):
        DatasetManifest.model_validate(payload)


def test_rejected_promotion_requires_reasons() -> None:
    payload = make_promotion_decision().model_dump(mode="json")
    payload["verdict"] = "rejected"
    payload["rejection_reasons"] = []
    with pytest.raises(ValidationError, match="rejection reason"):
        PromotionDecision.model_validate(payload)


def test_approved_promotion_cannot_have_failed_checks() -> None:
    payload = make_promotion_decision().model_dump(mode="json")
    payload["checks"][0]["passed"] = False
    with pytest.raises(ValidationError, match="failed checks"):
        PromotionDecision.model_validate(payload)


def test_drift_signals_are_independent() -> None:
    report = make_drift_report()
    assert report.data_drift.verdict is DriftVerdict.DRIFTED
    assert report.concept_drift.verdict is DriftVerdict.NOT_EVALUATED
    assert report.performance_degradation.verdict is DriftVerdict.NOT_EVALUATED


def test_small_current_window_cannot_report_data_drift_verdict() -> None:
    payload = make_drift_report().model_dump(mode="json")
    payload["current_window"]["row_count"] = 10
    with pytest.raises(ValidationError, match="insufficient_data"):
        DriftReport.model_validate(payload)


def test_evaluated_performance_requires_snapshot() -> None:
    payload = make_drift_report().model_dump(mode="json")
    payload["performance_degradation"] = {
        "verdict": "not_drifted",
        "detail": "roc_auc held steady",
        "method": "delayed_labels",
    }
    with pytest.raises(ValidationError, match="performance snapshot"):
        DriftReport.model_validate(payload)


def test_evaluated_performance_accepts_snapshot() -> None:
    payload = make_drift_report().model_dump(mode="json")
    payload["performance_degradation"] = {
        "verdict": "not_drifted",
        "detail": "roc_auc held steady",
        "method": "delayed_labels",
        "score": 0.0,
    }
    payload["performance"] = PerformanceSnapshot(
        metric="roc_auc",
        reference_value=0.93,
        current_value=0.92,
        delta=-0.01,
        labeled_sample_count=300,
    ).model_dump(mode="json")
    report = DriftReport.model_validate(payload)
    assert report.performance is not None
    assert report.data_drift.verdict is DriftVerdict.DRIFTED
