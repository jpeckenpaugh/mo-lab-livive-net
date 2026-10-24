# Generated JSON Schemas

These files are generated from the Pydantic models in
`src/modelops_lab/contracts/`. Do not hand-edit them.

```bash
uv run python scripts/export_schemas.py
```

`uv run pytest tests/unit/test_schemas.py` fails if a committed schema differs
byte-for-byte from regenerated output.

The five top-level contracts are frozen for later waves. A change requires a PM
decision and a schema version bump; consumers must not treat an older record as
equivalent to a newer one.
