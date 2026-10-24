# ModelOps Lab

ModelOps Lab is a lightweight, self-hosted reference implementation of a
complete machine learning operations lifecycle. The model is deliberately
simple; the product is the operational path from a versioned dataset through
training, promotion, serving, monitoring, drift detection, retraining, and
rollback.

**Current status: W1 dataset pipeline.** The repository contains the project
packaging, validated configuration models, the frozen cross-wave contracts and
their JSON Schemas, a seeded synthetic dataset generator with three drift
regimes, schema validation, canonical serialization and fingerprinting,
reproducible train/validation/test split manifests, and the `modelops data`
commands. Training, serving, and monitoring remain stubs that exit with status
2 and name the wave that will implement them.

## Dataset quick start

```bash
# Generate, validate, fingerprint, and write a versioned artifact.
uv run modelops data prepare --config configs/training.yaml --out artifacts/datasets

# Re-verify an existing artifact (manifest, canonical bytes, split manifests).
uv run modelops data validate artifacts/datasets/synthetic-v1.0.0-<fingerprint>

# Alternate synthetic regimes (the optional loader is documented in docs/dataset.md).
uv run modelops data prepare --config configs/training.yaml --regime covariate_shift
uv run modelops data prepare --config configs/training.yaml --regime improved_signal
```

Preparing is idempotent: re-running against an existing version directory
re-verifies it and exits 0, while a mismatch fails and suggests `--force`. The
artifact layout, feature table, generator math, regime semantics, validation
rules, and fingerprint scheme are documented in
[`docs/dataset.md`](docs/dataset.md).

## Development setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is managed by uv.

```bash
uv sync
uv run modelops --help
uv run pytest
```

Lint and formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

Regenerate the committed contract schemas after an intentional contract change:

```bash
uv run python scripts/export_schemas.py
```

Runtime settings are read from `MODELOPS_*` environment variables or a local
`.env` file. Copy `.env.example` to `.env` for development; the values there are
not secrets.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | Validated YAML for training, promotion, and monitoring |
| `src/modelops_lab/data.py` | Synthetic generator, canonical serialization, splits, artifact preparation |
| `src/modelops_lab/dataset_schema.py` | Declared dataset columns, ranges, and levels |
| `src/modelops_lab/validation.py` | Schema validation and strict CSV reading |
| `src/modelops_lab/contracts/` | Frozen Pydantic contracts and schema exporter |
| `contracts/` | Generated JSON Schemas, committed for review |
| `src/modelops_lab/cli.py` | Typer command surface (`data` implemented; later waves are stubs) |
| `tests/unit/` | Docker-free tests of config, contracts, schemas, data, validation, and CLI |
| `tests/integration/` | Real Postgres/MLflow tests (later waves) |
| `tests/e2e/` | Docker Compose tests (later waves) |

## Documentation

- Product specification: [`concept.md`](concept.md)
- Execution plan, waves, and frozen contract definitions: [`docs/plan.md`](docs/plan.md)
- Architecture decisions: [`docs/decisions.md`](docs/decisions.md)
- Dataset schema, generator, regimes, and versioning: [`docs/dataset.md`](docs/dataset.md)
