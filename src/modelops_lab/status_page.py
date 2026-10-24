"""Generated static lifecycle status page plus a machine-readable snapshot.

The page is a rendered artifact, never a control surface (ADR-004). It shows
the active model and its approval, recent transitions and decisions, the latest
drift verdicts, event counts, and report paths. It contains no JavaScript and
no interactive controls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from modelops_lab.contracts import DriftReport
from modelops_lab.monitoring import TEMPLATE_DIR, SqlMonitoringRepository
from modelops_lab.pointer import PointerStore

DEFAULT_STATUS_PATH = Path("artifacts/status/status.html")


class StatusPageError(RuntimeError):
    """Raised when the status page cannot read state or write its artifacts."""


@dataclass(frozen=True)
class StatusSnapshot:
    generated_at: datetime
    active: dict[str, Any] | None
    approval: dict[str, Any] | None
    transitions: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    event_total: int
    events_by_version: dict[str, int]
    report_dir: str


def _load_reports(report_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not report_dir.is_dir():
        return []
    reports: list[DriftReport] = []
    for path in sorted(report_dir.glob("drift-*.json")):
        try:
            reports.append(DriftReport.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    reports.sort(key=lambda report: report.generated_at, reverse=True)
    rows: list[dict[str, Any]] = []
    for report in reports[:limit]:
        rows.append(
            {
                "report_id": report.report_id,
                "generated_at": report.generated_at.isoformat(),
                "model_version": report.model_version,
                "data_drift": report.data_drift.verdict.value,
                "performance_degradation": report.performance_degradation.verdict.value,
                "concept_drift": report.concept_drift.verdict.value,
                "current_rows": report.current_window.row_count,
                "labeled_rows": (
                    report.performance.labeled_sample_count if report.performance is not None else 0
                ),
                "json_path": str(report_dir / f"drift-{report.report_id}.json"),
                "html_path": str(report_dir / f"drift-{report.report_id}.html"),
            }
        )
    return rows


def gather_status(
    *,
    database_url: str,
    report_dir: Path,
    now: datetime | None = None,
    limit: int = 10,
) -> StatusSnapshot:
    """Collect pointer, history, event counts, and latest drift reports."""
    store = PointerStore(database_url)
    try:
        active = store.active()
        transitions = store.transitions()[-limit:]
        decisions = store.decisions()[-limit:]
    except Exception as exc:
        raise StatusPageError(f"cannot read lifecycle state: {exc}") from exc
    repository = SqlMonitoringRepository(database_url)
    event_total, events_by_version = repository.event_counts()

    active_payload: dict[str, Any] | None = None
    approval_payload: dict[str, Any] | None = None
    if active is not None:
        active_payload = {
            "model_name": active.model_name,
            "model_version": active.model_version,
            "mlflow_run_id": active.mlflow_run_id,
            "dataset_version": active.dataset_version,
            "dataset_fingerprint": active.dataset_fingerprint,
            "decision_id": active.decision_id,
            "promoted_at": active.promoted_at.isoformat(),
            "updated_at": active.updated_at.isoformat(),
        }
        decision = store.decision(active.decision_id)
        if decision is not None:
            approval_payload = {
                "decision_id": decision.decision_id,
                "actor": decision.actor,
                "reason": decision.reason,
                "decided_at": decision.decided_at.isoformat(),
                "verdict": decision.verdict,
            }

    return StatusSnapshot(
        generated_at=(now or datetime.now(UTC)).astimezone(UTC),
        active=active_payload,
        approval=approval_payload,
        transitions=[
            {
                "id": transition.id,
                "operation": transition.operation,
                "from_model_version": transition.from_model_version,
                "to_model_version": transition.to_model_version,
                "actor": transition.actor,
                "reason": transition.reason,
                "occurred_at": transition.occurred_at.isoformat(),
            }
            for transition in transitions
        ],
        decisions=[
            {
                "decision_id": decision.decision_id,
                "candidate_model_version": decision.candidate_model_version,
                "active_model_version_before": decision.active_model_version_before,
                "verdict": decision.verdict,
                "actor": decision.actor,
                "reason": decision.reason,
                "decided_at": decision.decided_at.isoformat(),
            }
            for decision in decisions
        ],
        reports=_load_reports(report_dir),
        event_total=event_total,
        events_by_version=events_by_version,
        report_dir=str(report_dir),
    )


def snapshot_payload(snapshot: StatusSnapshot) -> dict[str, Any]:
    return {
        "generated_at": snapshot.generated_at.isoformat(),
        "active": snapshot.active,
        "approval": snapshot.approval,
        "transitions": snapshot.transitions,
        "decisions": snapshot.decisions,
        "reports": snapshot.reports,
        "event_counts": {
            "total": snapshot.event_total,
            "by_model_version": snapshot.events_by_version,
        },
        "report_dir": snapshot.report_dir,
    }


def render_status_html(snapshot: StatusSnapshot) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(("html", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template("status.html.j2").render(status=snapshot_payload(snapshot))


def write_status_page(snapshot: StatusSnapshot, output_path: Path) -> tuple[Path, Path]:
    """Write ``status.html`` and its sibling ``status.json``; return both."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    try:
        output_path.write_text(render_status_html(snapshot), encoding="utf-8")
        json_path.write_text(
            json.dumps(snapshot_payload(snapshot), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise StatusPageError(f"cannot write status page: {exc}") from exc
    return output_path, json_path
