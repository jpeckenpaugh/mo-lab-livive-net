# ModelOps Lab

ModelOps Lab is a lightweight, self-hosted reference implementation of a
complete machine learning operations lifecycle. The model is deliberately
simple; the product is the operational path from a versioned dataset through
training, promotion, serving, monitoring, drift detection, retraining, and
rollback.

**Current status: W4 serving, prediction events, rollback, and reconciliation.**
The repository contains the project packaging, validated configuration models,
the frozen cross-wave contracts and their JSON Schemas (schema version `1.1.0`;
`1.0.0` records still parse), a seeded synthetic dataset generator with three
drift regimes, schema validation, canonical serialization and fingerprinting,
reproducible train/validation/test split manifests, the `modelops data`
commands, a deterministic scikit-learn training pipeline that records a
complete MLflow run, a Docker Compose PostgreSQL + MLflow + API stack, the
`modelops train` / `modelops evaluate` commands, a policy-checked promotion gate
with an MLflow registry alias and a transactional PostgreSQL active-model
pointer, a FastAPI inference service with Prometheus metrics and best-effort
prediction events, verified rollback, and registry reconciliation. Monitoring,
retraining, and the status page remain stubs that exit with status 2 and name
the wave that will implement them.

## Quick start

```bash
# 1. Install the locked environment (Python 3.12 is managed by uv).
uv sync

# 2. Start PostgreSQL, MLflow, and the inference API.
cp .env.example .env
docker compose up -d --wait
curl -fsS http://localhost:5001/health
curl -fsS http://localhost:8000/health

# 3. Generate, validate, and fingerprint a dataset artifact.
uv run modelops data prepare --config configs/training.yaml

# 4. Train, record one auditable MLflow run, and inspect it in the UI.
uv run modelops train \
  --config configs/training.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --tracking-uri http://localhost:5001
open http://localhost:5001

# 5. Re-evaluate the logged model on the validation slice (logs nothing).
uv run modelops evaluate \
  --config configs/training.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --run-id <mlflow_run_id> \
  --tracking-uri http://localhost:5001

# 6. Apply the schema once, then promote the candidate through the gate.
uv run modelops db upgrade
uv run modelops promote \
  --candidate-run-id <mlflow_run_id> --approve \
  --reason "baseline passes the calibrated floors" \
  --tracking-uri http://localhost:5001
open http://localhost:5001

# 7. Inspect and score with the promoted version through the container.
curl -fsS http://localhost:8000/model
curl -fsS http://localhost:8000/metrics

# 8. After a second approved promotion, roll back to the prior version.
#    The next request serves it with no restart.
uv run modelops rollback \
  --to-version <previous_version> --reason "demonstrate rollback" \
  --tracking-uri http://localhost:5001
curl -fsS http://localhost:8000/model

# 9. Confirm the pointer and alias agree (or repair them under the lock).
uv run modelops registry reconcile
```

The dataset artifact layout, generator math, regimes, and fingerprint scheme are
documented in [`docs/dataset.md`](docs/dataset.md). The pipeline, leakage
guarantees, metric definitions, exact MLflow run schema, compose commands, and
evaluate workflow are documented in [`docs/training.md`](docs/training.md). The
gate checks, decision records, pointer/alias failure modes, concurrency
guarantee, and the approved/rejected command sequences are documented in
[`docs/promotion.md`](docs/promotion.md).

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

With the Compose stack running, exercise the real PostgreSQL + MLflow integration
tier:

```bash
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m integration
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
| `compose.yaml` | PostgreSQL + MLflow + FastAPI inference stack |
| `Dockerfile` | Non-root API/CLI image: pinned uv, migrations, uvicorn, healthcheck |
| `docker/mlflow.Dockerfile` | Pinned MLflow 3.17.2 image with a stdlib healthcheck |
| `configs/` | Validated YAML for training, promotion, and monitoring |
| `src/modelops_lab/data.py` | Synthetic generator, canonical serialization, splits, artifact preparation, verified loading |
| `src/modelops_lab/dataset_schema.py` | Declared dataset columns, ranges, and levels |
| `src/modelops_lab/validation.py` | Schema validation and strict CSV reading |
| `src/modelops_lab/train.py` | Pipeline, metrics, deterministic fit/evaluate, evaluation document |
| `src/modelops_lab/tracking.py` | MLflow run logging, audit tags, logged-model read-back |
| `src/modelops_lab/registry.py` | MLflow registry protocol: idempotent registration, alias, version loading, reconcile |
| `src/modelops_lab/promotion.py` | Gate checks, decision construction, approval orchestration, descriptor |
| `src/modelops_lab/pointer.py` | Transactional active pointer, transition log, decision records, rollback/reconcile transactions |
| `src/modelops_lab/serving.py` | FastAPI app, pointer-driven model cache, validation, metrics, event logging |
| `src/modelops_lab/rollback.py` | Verified rollback and pointer/alias reconciliation |
| `src/modelops_lab/db.py` | SQLAlchemy schema (promotion state + prediction events) and Alembic entry point |
| `migrations/`, `alembic.ini` | Versioned schema, including `0002_prediction_events` and `0003_prediction_event_id` |
| `src/modelops_lab/contracts/` | Frozen Pydantic contracts and schema exporter |
| `contracts/` | Generated JSON Schemas (1.1.0), committed for review |
| `src/modelops_lab/cli.py` | Typer command surface (`data`, `db`, `train`, `evaluate`, `promote`, `serve`, `rollback`, `registry`; W5/W6 commands are stubs) |
| `tests/unit/` | Docker-free tests of config, contracts, schemas, data, training, tracking, promotion, serving, rollback, and CLI |
| `tests/integration/` | Real MLflow/Postgres tests (skipped unless the test URLs are set) |
| `tests/e2e/` | Compose API tests (skipped unless `MODELOPS_TEST_API_URL` is set) |

## Documentation

- Product specification: [`concept.md`](concept.md)
- Execution plan, waves, and frozen contract definitions: [`docs/plan.md`](docs/plan.md)
- Architecture decisions: [`docs/decisions.md`](docs/decisions.md)
- Dataset schema, generator, regimes, and versioning: [`docs/dataset.md`](docs/dataset.md)
- Training pipeline, metrics, MLflow run schema, and evaluate workflow: [`docs/training.md`](docs/training.md)
- Promotion gate, decisions, active pointer, and concurrency: [`docs/promotion.md`](docs/promotion.md)
- Serving endpoints, events, metrics, rollback, reconciliation, and containers: [`docs/serving.md`](docs/serving.md)
