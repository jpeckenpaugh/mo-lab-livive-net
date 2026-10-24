"""CLI exit-code tests for ``modelops retrain`` using injected fakes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import modelops_lab.cli as cli
import modelops_lab.retrain as retrain_module
from modelops_lab.config import TrainingConfig
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
from modelops_lab.tracking import TrainingOutcome
from modelops_lab.train import Metrics

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)
runner = CliRunner()


def make_metrics(passing: bool = True) -> Metrics:
    values = (0.94, 0.86, 0.85, 0.85) if passing else (0.50, 0.00, 0.00, 0.00)
    return Metrics(
        roc_auc=values[0],
        precision=values[1],
        recall=values[2],
        f1=values[3],
        accuracy=0.86 if passing else 0.40,
        log_loss=0.35,
        true_negative=210,
        false_positive=20,
        false_negative=25,
        true_positive=145,
        support=170,
        predicted_positive=165 if passing else 0,
        sample_count=400,
    )


class FakeModel:
    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics


class FakeRegistry:
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


def make_drift_report(data: DriftVerdict, *, model_version: str = "1") -> DriftReport:
    drifted = data is DriftVerdict.DRIFTED
    return DriftReport(
        report_id="report-cli",
        generated_at=NOW,
        model_version=model_version,
        reference_window=WindowSummary(label="reference", source="validation_split", row_count=400),
        current_window=WindowSummary(label="current", source="prediction_events", row_count=400),
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
        thresholds=DriftThresholds(
            psi_threshold=0.2,
            ks_pvalue_threshold=0.05,
            missing_rate_delta_threshold=0.05,
            aggregate_feature_share_threshold=0.3,
            min_sample_size=100,
        ),
        data_drift=SignalVerdict(verdict=data, detail="test evidence"),
        concept_drift=SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED, detail="labels unavailable"
        ),
        performance_degradation=SignalVerdict(
            verdict=DriftVerdict.NOT_EVALUATED, detail="labels unavailable"
        ),
    )


class CliHarness:
    def __init__(
        self,
        *,
        loaded: LoadedDataset,
        dataset_dir: Path,
        tmp_path: Path,
        passing: bool = True,
    ) -> None:
        self.loaded = loaded
        self.dataset_dir = dataset_dir
        self.tmp_path = tmp_path
        self.metrics = make_metrics(passing)
        self.database_path = tmp_path / "state.db"
        self.store = PointerStore(f"sqlite:///{self.database_path}")
        self.store.create_schema()
        self.registry = FakeRegistry()
        self.registry.alias_version = self.registry.seed_version("run-active", make_metrics(True))
        with self.store.session() as session, session.begin():
            session.add(
                ActiveModelRow(
                    id=1,
                    model_name=self.registry.registered_model_name,
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

    def trainer(
        self,
        config: TrainingConfig,
        loaded: LoadedDataset,
        tracking_uri: str,
        run_name: str | None,
    ) -> TrainingOutcome:
        return TrainingOutcome(
            run_id="run-cli-candidate",
            experiment_id="1",
            experiment_name="modelops-lab",
            tracking_uri=tracking_uri,
            model_id="m-cli-candidate",
            model_uri="models:/m-cli-candidate",
            metrics=self.metrics,
            dataset_fingerprint=loaded.manifest.fingerprint,
        )

    def inspector(self, run_id: str, tracking_uri: str) -> CandidateRun:
        return CandidateRun(
            run_id=run_id,
            status="FINISHED",
            tags={
                "dataset_version": self.loaded.manifest.dataset_version,
                "dataset_fingerprint": self.loaded.manifest.fingerprint,
                "evaluation_split": "validation",
            },
            model_id="m-cli-candidate",
            model_uri="models:/m-cli-candidate",
            model=FakeModel(self.metrics),
            model_error=None,
            evaluation={
                "evaluation_split": "validation",
                "dataset_fingerprint": self.loaded.manifest.fingerprint,
                "dataset_version": self.loaded.manifest.dataset_version,
                "metrics": self.metrics.as_dict(),
            },
            evaluation_error=None,
        )

    def evaluator(self, model: Any, loaded: LoadedDataset, split: str) -> Metrics:
        return model.metrics

    def predictor(self, model: Any, frame: Any, features: list[str]) -> np.ndarray:
        return np.full(len(frame), 0.5)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = retrain_module.perform_retraining

        def wrapped(request, **kwargs):
            return real(
                request,
                training_config=kwargs["training_config"],
                promotion_config=kwargs["promotion_config"],
                tracking_uri=kwargs["tracking_uri"],
                store=kwargs["store"],
                registry=self.registry,
                decision_dir=self.tmp_path / "decisions",
                retrain_dir=kwargs["retrain_dir"],
                dataset_loader=lambda path: self.loaded,
                trainer=self.trainer,
                inspector=self.inspector,
                evaluator=self.evaluator,
                predictor=self.predictor,
                now=NOW,
            )

        monkeypatch.setattr(cli, "perform_retraining", wrapped)

    def invoke(self, *extra: str):
        return runner.invoke(
            cli.app,
            [
                "retrain",
                "--dataset",
                str(self.dataset_dir),
                "--database-url",
                f"sqlite:///{self.database_path}",
                "--tracking-uri",
                "http://tracking.invalid",
                "--retrain-dir",
                str(self.tmp_path / "retraining"),
                "--actor",
                "tester",
                "--reason",
                "cli retraining test",
                *extra,
            ],
        )


def test_cli_retrain_eligible_exits_zero_and_writes_audit(
    loaded_dataset: LoadedDataset,
    prepared_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CliHarness(
        loaded=loaded_dataset,
        dataset_dir=prepared_dataset,
        tmp_path=tmp_path,
        passing=True,
    )
    harness.install(monkeypatch)

    result = harness.invoke()

    assert result.exit_code == 0, result.output + result.stderr
    assert "gate: eligible" in result.stdout
    assert "active: 1 (unchanged)" in result.stdout
    assert "next_command: modelops promote --candidate-run-id run-cli-candidate" in result.stdout
    audits = list((tmp_path / "retraining").glob("*.json"))
    assert len(audits) == 1
    audit = json.loads(audits[0].read_text(encoding="utf-8"))
    assert audit["verdict"] == "eligible"
    assert audit["candidate_version"] == "2"
    assert harness.store.active().model_version == "1"
    assert harness.store.decisions() == []
    assert harness.registry.alias_version == "1"


def test_cli_retrain_rejected_exits_three(
    loaded_dataset: LoadedDataset,
    prepared_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CliHarness(
        loaded=loaded_dataset,
        dataset_dir=prepared_dataset,
        tmp_path=tmp_path,
        passing=False,
    )
    harness.install(monkeypatch)

    result = harness.invoke()

    assert result.exit_code == 3, result.output + result.stderr
    assert "gate: rejected" in result.stdout
    assert "active: 1 (unchanged)" in result.stdout
    audit = json.loads(next((tmp_path / "retraining").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["verdict"] == "rejected"
    assert harness.store.active().model_version == "1"


def test_cli_retrain_drift_without_a_report_is_an_error(
    loaded_dataset: LoadedDataset,
    prepared_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CliHarness(
        loaded=loaded_dataset,
        dataset_dir=prepared_dataset,
        tmp_path=tmp_path,
    )
    harness.install(monkeypatch)

    result = harness.invoke("--trigger", "drift")

    assert result.exit_code == 1
    assert "requires --drift-report" in result.stderr
    assert list((tmp_path / "retraining").glob("*.json")) == []


def test_cli_retrain_drift_report_without_drift_is_refused(
    loaded_dataset: LoadedDataset,
    prepared_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CliHarness(
        loaded=loaded_dataset,
        dataset_dir=prepared_dataset,
        tmp_path=tmp_path,
    )
    harness.install(monkeypatch)
    report_path = tmp_path / "no-drift.json"
    report_path.write_text(
        make_drift_report(DriftVerdict.NOT_DRIFTED).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = harness.invoke("--trigger", "drift", "--drift-report", str(report_path))

    assert result.exit_code == 1
    assert "no drifted signal" in result.stderr
    assert list((tmp_path / "retraining").glob("*.json")) == []


def test_cli_retrain_drift_report_for_a_different_model_prints_a_warning(
    loaded_dataset: LoadedDataset,
    prepared_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CliHarness(
        loaded=loaded_dataset,
        dataset_dir=prepared_dataset,
        tmp_path=tmp_path,
    )
    harness.install(monkeypatch)
    report_path = tmp_path / "stale-model.json"
    report_path.write_text(
        make_drift_report(DriftVerdict.DRIFTED, model_version="99").model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = harness.invoke("--trigger", "drift", "--drift-report", str(report_path))

    assert result.exit_code == 0, result.output + result.stderr
    assert "warning:" in result.stderr
    assert "model version 99" in result.stderr
    audit = json.loads(next((tmp_path / "retraining").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["drift_report_model_version"] == "99"
    assert audit["warnings"]
    assert harness.store.active().model_version == "1"


def test_cli_retrain_drift_report_with_evidence_proceeds(
    loaded_dataset: LoadedDataset,
    prepared_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = CliHarness(
        loaded=loaded_dataset,
        dataset_dir=prepared_dataset,
        tmp_path=tmp_path,
    )
    harness.install(monkeypatch)
    report_path = tmp_path / "drifted.json"
    report_path.write_text(
        make_drift_report(DriftVerdict.DRIFTED).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = harness.invoke("--trigger", "drift", "--drift-report", str(report_path))

    assert result.exit_code == 0, result.output + result.stderr
    assert "drift_report_id: report-cli" in result.stdout
    assert "data_drift=drifted" in result.stdout
    audit = json.loads(next((tmp_path / "retraining").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["trigger"] == "drift"
    assert audit["drift_report_id"] == "report-cli"
    assert audit["drift_verdicts"]["data_drift"] == "drifted"
    assert harness.store.active().model_version == "1"
