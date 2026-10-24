"""Unit tests for the closed retraining loop (Docker-free, fakes only)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modelops_lab.config import TrainingConfig, load_promotion_config
from modelops_lab.contracts import (
    DriftReport,
    DriftThresholds,
    DriftVerdict,
    FeatureDriftStatistic,
    SignalVerdict,
    WindowSummary,
)
from modelops_lab.data import LoadedDataset
from modelops_lab.db import ActiveModelRow
from modelops_lab.pointer import PointerStore
from modelops_lab.promotion import CandidateRun
from modelops_lab.registry import RegistryError
from modelops_lab.retrain import (
    RetrainError,
    RetrainRequest,
    effective_training_config,
    perform_retraining,
)
from modelops_lab.tracking import TrainingOutcome
from modelops_lab.train import Metrics

ROOT = Path(__file__).resolve().parents[2]
PROMOTION_CONFIG = ROOT / "configs" / "promotion.yaml"
NOW = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)


def make_metrics(
    roc_auc: float = 0.94,
    precision: float = 0.86,
    recall: float = 0.85,
    f1: float = 0.85,
) -> Metrics:
    return Metrics(
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=0.86,
        log_loss=0.35,
        true_negative=210,
        false_positive=20,
        false_negative=25,
        true_positive=145,
        support=170,
        predicted_positive=165,
        sample_count=400,
    )


class FakeModel:
    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics


class FakeRegistry:
    """Minimal ModelRegistry double with deterministic versions."""

    def __init__(self) -> None:
        self.registered_model_name = "modelops-lab-classifier"
        self.alias = "champion"
        self.alias_version: str | None = None
        self.versions: dict[str, str] = {}
        self.models: dict[str, Any] = {}
        self._next = 0

    def find_version_for_run(self, run_id: str) -> str | None:
        return self.versions.get(run_id)

    def register_run(self, run_id: str, model_id: str) -> str:
        existing = self.versions.get(run_id)
        if existing is not None:
            return existing
        self._next += 1
        version = str(self._next)
        self.versions[run_id] = version
        self.models[version] = FakeModel(make_metrics())
        return version

    def seed_version(self, run_id: str, metrics: Metrics) -> str:
        version = self.register_run(run_id, f"m-{run_id}")
        self.models[version] = FakeModel(metrics)
        return version

    def set_alias(self, version: str) -> None:
        if version not in self.models:
            raise RegistryError(f"unknown version {version}")
        self.alias_version = version

    def delete_alias(self) -> None:
        self.alias_version = None

    def get_alias_version(self) -> str | None:
        return self.alias_version

    def load_version(self, version: str) -> Any:
        if version not in self.models:
            raise RegistryError(f"cannot load version {version}")
        return self.models[version]

    def version_uri(self, version: str) -> str:
        return f"models:/{self.registered_model_name}/{version}"


def seed_active(store: PointerStore, registry: FakeRegistry, loaded: LoadedDataset) -> None:
    registry.alias_version = registry.seed_version("run-active", make_metrics())
    with store.session() as session, session.begin():
        session.add(
            ActiveModelRow(
                id=1,
                model_name=registry.registered_model_name,
                model_version="1",
                mlflow_run_id="run-active",
                mlflow_model_id="m-run-active",
                dataset_version=loaded.manifest.dataset_version,
                dataset_fingerprint=loaded.manifest.fingerprint,
                decision_id="dec-active",
                promoted_at=NOW,
                updated_at=NOW,
            )
        )


def candidate_for(loaded: LoadedDataset, metrics: Metrics) -> CandidateRun:
    return CandidateRun(
        run_id="run-candidate",
        status="FINISHED",
        tags={
            "dataset_version": loaded.manifest.dataset_version,
            "dataset_fingerprint": loaded.manifest.fingerprint,
            "evaluation_split": "validation",
        },
        model_id="m-candidate",
        model_uri="models:/m-candidate",
        model=FakeModel(metrics),
        model_error=None,
        evaluation={
            "evaluation_split": "validation",
            "dataset_fingerprint": loaded.manifest.fingerprint,
            "dataset_version": loaded.manifest.dataset_version,
            "metrics": metrics.as_dict(),
        },
        evaluation_error=None,
    )


def make_drift_report(
    *,
    report_id: str = "report-1",
    data: DriftVerdict = DriftVerdict.DRIFTED,
    model_version: str = "1",
) -> DriftReport:
    drifted = data is DriftVerdict.DRIFTED
    thresholds = DriftThresholds(
        psi_threshold=0.2,
        ks_pvalue_threshold=0.05,
        missing_rate_delta_threshold=0.05,
        aggregate_feature_share_threshold=0.3,
        min_sample_size=100,
    )
    window = WindowSummary(label="current", source="prediction_events", row_count=400)
    reference = WindowSummary(label="reference", source="validation_split", row_count=400)
    return DriftReport(
        report_id=report_id,
        generated_at=NOW,
        model_version=model_version,
        reference_window=reference,
        current_window=window,
        feature_statistics=[
            FeatureDriftStatistic(
                feature="age",
                reference_missing_rate=0.0,
                current_missing_rate=0.0,
                psi=0.4 if drifted else 0.01,
                ks_statistic=0.3 if drifted else 0.01,
                ks_pvalue=0.001 if drifted else 0.9,
                drifted=drifted,
            )
        ],
        thresholds=thresholds,
        data_drift=SignalVerdict(verdict=data, detail="test evidence", method="ks+psi"),
        concept_drift=SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED, detail="labels unavailable"
        ),
        performance_degradation=SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED, detail="labels unavailable"
        ),
    )


class Harness:
    def __init__(self, loaded: LoadedDataset, *, metrics: Metrics, tmp_path: Path) -> None:
        self.loaded = loaded
        self.metrics = metrics
        self.store = PointerStore(f"sqlite:///{tmp_path / 'retrain.db'}")
        self.store.create_schema()
        self.registry = FakeRegistry()
        seed_active(self.store, self.registry, loaded)
        self.promotion_config = load_promotion_config(PROMOTION_CONFIG)

    def trainer(
        self,
        config: TrainingConfig,
        loaded: LoadedDataset,
        tracking_uri: str,
        run_name: str | None,
    ) -> TrainingOutcome:
        assert loaded.manifest.fingerprint == self.loaded.manifest.fingerprint
        assert config.dataset.regime == "baseline"
        return TrainingOutcome(
            run_id="run-candidate",
            experiment_id="1",
            experiment_name="modelops-lab",
            tracking_uri=tracking_uri,
            model_id="m-candidate",
            model_uri="models:/m-candidate",
            metrics=self.metrics,
            dataset_fingerprint=loaded.manifest.fingerprint,
        )

    def inspector(self, run_id: str, tracking_uri: str) -> CandidateRun:
        assert run_id == "run-candidate"
        return candidate_for(self.loaded, self.metrics)

    def evaluator(self, model: Any, loaded: LoadedDataset, split: str) -> Metrics:
        return model.metrics

    def predictor(self, model: Any, frame: Any, features: list[str]) -> np.ndarray:
        return np.full(len(frame), 0.5)

    def run(
        self,
        tmp_path: Path,
        *,
        trigger: str = "manual",
        drift_report_path: Path | None = None,
        config: TrainingConfig | None = None,
    ):
        request = RetrainRequest(
            dataset_dir=tmp_path / "dataset",
            trigger=trigger,  # type: ignore[arg-type]
            drift_report_path=drift_report_path,
            actor="tester",
            reason="unit retraining",
        )
        return perform_retraining(
            request,
            training_config=config or self._training_config(),
            promotion_config=self.promotion_config,
            tracking_uri="http://tracking.invalid",
            store=self.store,
            registry=self.registry,
            decision_dir=tmp_path / "decisions",
            retrain_dir=tmp_path / "retraining",
            dataset_loader=lambda path: self.loaded,
            trainer=self.trainer,
            inspector=self.inspector,
            evaluator=self.evaluator,
            predictor=self.predictor,
            now=NOW,
        )

    @staticmethod
    def _training_config() -> TrainingConfig:
        from modelops_lab.config import load_training_config

        return load_training_config(ROOT / "configs" / "training.yaml")


@pytest.fixture()
def loaded(loaded_dataset: LoadedDataset) -> LoadedDataset:
    return loaded_dataset


def test_eligible_retrain_writes_audit_without_moving_state(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    harness = Harness(loaded, metrics=make_metrics(), tmp_path=tmp_path)

    result = harness.run(tmp_path)

    assert result.verdict == "eligible"
    assert result.candidate_version == "2"
    assert result.active_before == "1"
    assert result.dataset_version == loaded.manifest.dataset_version
    assert result.dataset_regime == "baseline"
    assert result.dataset_fingerprint == loaded.manifest.fingerprint
    assert result.drift_report_id is None
    assert result.next_command.startswith(
        "modelops promote --candidate-run-id run-candidate --approve --reason "
    )
    assert result.audit_path.exists()

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["retrain_id"] == result.retrain_id
    assert audit["trigger"] == "manual"
    assert audit["drift_report_id"] is None
    assert audit["drift_verdicts"] is None
    assert audit["dataset_version"] == loaded.manifest.dataset_version
    assert audit["dataset_fingerprint"] == loaded.manifest.fingerprint
    assert audit["dataset_regime"] == "baseline"
    assert audit["run_id"] == "run-candidate"
    assert audit["candidate_version"] == "2"
    assert audit["active_version_before"] == "1"
    assert audit["verdict"] == "eligible"
    assert audit["actor"] == "tester"
    assert audit["reason"] == "unit retraining"
    assert audit["next_command"] == result.next_command
    assert {check["name"] for check in audit["checks"]} >= {
        "run_finished",
        "quality_floor",
        "regression_vs_active",
        "registry_consistency",
    }

    # Evaluation-only means exactly this: no decision row, no pointer move, no
    # alias write, and no decision artifact.
    assert harness.store.active() is not None
    assert harness.store.active().model_version == "1"
    assert harness.store.decisions() == []
    assert harness.store.transitions() == []
    assert harness.registry.alias_version == "1"
    assert list((tmp_path / "decisions").glob("*.json")) == []


def test_rejected_retrain_records_rejected_audit_and_keeps_active(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    harness = Harness(
        loaded,
        metrics=make_metrics(roc_auc=0.5, precision=0.0, recall=0.0, f1=0.0),
        tmp_path=tmp_path,
    )

    result = harness.run(tmp_path)

    assert result.verdict == "rejected"
    assert harness.store.active() is not None
    assert harness.store.active().model_version == "1"
    assert harness.store.decisions() == []
    assert harness.registry.alias_version == "1"
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["verdict"] == "rejected"
    failed = {check["name"] for check in audit["checks"] if not check["passed"]}
    assert "quality_floor" in failed
    assert audit["next_command"].endswith(
        "--dry-run --reason 'inspect the rejected retrained candidate'"
    )


def test_drift_trigger_requires_a_report(loaded: LoadedDataset, tmp_path: Path) -> None:
    harness = Harness(loaded, metrics=make_metrics(), tmp_path=tmp_path)

    with pytest.raises(RetrainError, match="requires --drift-report"):
        harness.run(tmp_path, trigger="drift")


def test_drift_trigger_refuses_a_report_without_drift(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    harness = Harness(loaded, metrics=make_metrics(), tmp_path=tmp_path)
    report_path = tmp_path / "no-drift.json"
    report_path.write_text(
        make_drift_report(data=DriftVerdict.NOT_DRIFTED).model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(RetrainError, match="no drifted signal"):
        harness.run(tmp_path, trigger="drift", drift_report_path=report_path)


def test_drift_trigger_records_report_id_and_verdicts(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    harness = Harness(loaded, metrics=make_metrics(), tmp_path=tmp_path)
    report_path = tmp_path / "drifted.json"
    report_path.write_text(
        make_drift_report(data=DriftVerdict.DRIFTED).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = harness.run(tmp_path, trigger="drift", drift_report_path=report_path)

    assert result.verdict == "eligible"
    assert result.drift_report_id == "report-1"
    assert result.drift_verdicts == {
        "data_drift": "drifted",
        "concept_drift": "not_evaluated",
        "performance_degradation": "not_evaluated",
    }
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["trigger"] == "drift"
    assert audit["drift_report_id"] == "report-1"
    assert audit["drift_verdicts"]["data_drift"] == "drifted"
    assert harness.store.active().model_version == "1"
    assert harness.store.decisions() == []


def test_drift_report_for_a_different_model_warns_but_proceeds(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    harness = Harness(loaded, metrics=make_metrics(), tmp_path=tmp_path)
    report_path = tmp_path / "stale-model.json"
    report_path.write_text(
        make_drift_report(data=DriftVerdict.DRIFTED, model_version="99").model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = harness.run(tmp_path, trigger="drift", drift_report_path=report_path)

    assert result.verdict == "eligible"
    assert result.drift_report_id == "report-1"
    assert len(result.warnings) == 1
    assert "produced for model version 99" in result.warnings[0]
    assert "active version at retraining time is 1" in result.warnings[0]
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["drift_report_model_version"] == "99"
    assert audit["warnings"] == result.warnings
    assert harness.store.active().model_version == "1"
    assert harness.store.decisions() == []


def test_effective_config_comes_from_the_dataset_manifest(
    loaded: LoadedDataset, training_config: TrainingConfig
) -> None:
    stale = training_config.model_copy(
        update={
            "dataset": training_config.dataset.model_copy(
                update={"seed": 999, "n_samples": 100, "regime": "concept_shift"}
            ),
            "split": training_config.split.model_copy(update={"seed": 999}),
        }
    )

    effective = effective_training_config(stale, loaded)

    assert effective.dataset.regime == "baseline"
    assert effective.dataset.seed == loaded.manifest.seed
    assert effective.dataset.n_samples == loaded.manifest.row_count
    assert effective.dataset.kind == loaded.manifest.source_kind
    assert effective.split.seed == loaded.manifest.split_seed
    # Model and tracking settings still come from the operator's config.
    assert effective.model == stale.model
    assert effective.tracking == stale.tracking
