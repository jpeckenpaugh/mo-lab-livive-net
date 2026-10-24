"""Committed JSON Schemas must exactly match generated output."""

from __future__ import annotations

from pathlib import Path

from modelops_lab.contracts.export import CONTRACT_MODELS, write_schemas

ROOT = Path(__file__).resolve().parents[2]
COMMITTED_DIR = ROOT / "contracts"


def test_regenerated_schemas_match_committed_files(tmp_path: Path) -> None:
    written = write_schemas(tmp_path)
    assert {path.name for path in written} == {f"{name}.schema.json" for name in CONTRACT_MODELS}

    for path in written:
        committed = COMMITTED_DIR / path.name
        assert committed.is_file(), f"missing committed schema: {committed}"
        assert path.read_bytes() == committed.read_bytes(), (
            f"{path.name} is stale; run `uv run python scripts/export_schemas.py`"
        )


def test_schema_export_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_schemas(first)
    write_schemas(second)

    for name in CONTRACT_MODELS:
        filename = f"{name}.schema.json"
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
