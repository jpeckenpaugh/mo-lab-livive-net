"""Deterministic JSON Schema export for the frozen contracts.

The exporter sorts every mapping key and uses LF line endings so regenerated
schemas can be compared byte-for-byte in tests and CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelops_lab.contracts.base import ContractModel
from modelops_lab.contracts.dataset import DatasetManifest
from modelops_lab.contracts.drift import DriftReport
from modelops_lab.contracts.model import ModelDescriptor
from modelops_lab.contracts.prediction import PredictionEvent
from modelops_lab.contracts.promotion import PromotionDecision

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

CONTRACT_MODELS: dict[str, type[ContractModel]] = {
    "dataset_manifest": DatasetManifest,
    "model_descriptor": ModelDescriptor,
    "promotion_decision": PromotionDecision,
    "prediction_event": PredictionEvent,
    "drift_report": DriftReport,
}


def render_schema(model: type[ContractModel]) -> str:
    """Render one contract as stable, human-readable JSON Schema text."""
    schema = {
        "$schema": JSON_SCHEMA_DIALECT,
        **model.model_json_schema(),
    }
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_schemas(output_dir: str | Path) -> list[Path]:
    """Write every contract schema to ``output_dir`` and return the paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in CONTRACT_MODELS.items():
        path = directory / f"{name}.schema.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_schema(model))
        written.append(path)
    return written
