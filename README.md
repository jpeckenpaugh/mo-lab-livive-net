# ModelOps Lab

ModelOps Lab is a lightweight, self-hosted reference implementation of a
complete machine learning operations lifecycle. The model is deliberately
simple; the product is the operational path from a versioned dataset through
training, promotion, serving, monitoring, drift detection, retraining, and
rollback.

**Current status: W6 retraining loop, registration serialization, and rollback
hardening.** The repository contains the project packaging, validated
configuration models, the frozen cross-wave contracts and their JSON Schemas
(schema version `1.1.0`; `1.0.0` records still parse), a seeded synthetic
dataset generator with four drift regimes (baseline, covariate shift, improved
signal, and concept shift), schema validation, canonical serialization and
fingerprinting, reproducible train/validation/test split manifests, the
`modelops data` commands, a deterministic scikit-learn training pipeline that
records a complete MLflow run, a Docker Compose PostgreSQL + MLflow + API
stack, the `modelops train` / `modelops evaluate` commands, a policy-checked
promotion gate with an MLflow registry alias and a transactional PostgreSQL
active-model pointer, a FastAPI inference service with Prometheus metrics,
best-effort prediction events, delayed labels, and a static status page,
verified rollback, registry reconciliation, in-repo drift statistics with three
independent verdicts, generated JSON + HTML drift reports, a deterministic
traffic simulator, prediction-event retention pruning, and a closed retraining
loop that trains and evaluates a candidate from drift evidence but never
promotes it automatically. Same-run registration is serialized across processes
with a PostgreSQL advisory lock, and promotion-versus-rollback races leave the
pointer, alias, and transitions consistent.

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

# 10. Replay labeled traffic, then compare it with the model's dataset.
uv run modelops simulate-traffic \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --split validation --count 400 --with-labels --seed 7
uv run modelops monitor --config configs/monitoring.yaml --window-size 400

# 11. Show the three-scenario story: clean, covariate shift, concept shift.
uv run modelops data prepare --config configs/training.yaml --regime covariate_shift
uv run modelops data prepare --config configs/training.yaml --regime concept_shift
#   replay <covariate-dataset> then monitor: data drifted, performance drifted,
#   concept not_evaluated (confounded); replay <concept-dataset>: data stable,
#   performance drifted, concept drifted. Exact commands: docs/monitoring.md.

# 12. Retrain from drift evidence. This trains and evaluates only; it never
#     promotes. An eligible result prints the exact promotion command to run.
uv run modelops retrain \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --trigger drift \
  --drift-report reports/drift/drift-<timestamp>-v<version>-<id>.json \
  --reason "drift evidence justifies a new candidate" \
  --tracking-uri http://localhost:5001
# Exit 0 = eligible; run the printed `modelops promote ... --approve` by hand.
# Exit 3 = rejected; active stays unchanged and the audit says why.

# 13. Generate the static lifecycle page and check retention.
uv run modelops status --output artifacts/status/status.html
uv run modelops events prune --dry-run
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
uv run modelops data prepare --config configs/training.yaml --regime concept_shift
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
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops_test \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m integration
```

Integration fixtures refuse to run against a database whose name does not end
in `_test`, because they truncate lifecycle tables between cases. Fresh Compose
volumes create `modelops_test` via `docker/postgres-init.sql`; for an existing
volume run `docker compose exec postgres createdb -U modelops modelops_test`.
Tests write generated artifacts under a temporary `MODELOPS_ARTIFACT_ROOT`, never
under the repository's `artifacts/` or `reports/`.

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
| `src/modelops_lab/db.py` | SQLAlchemy schema (promotion state + prediction events + delayed labels) and Alembic entry point |
| `src/modelops_lab/monitoring.py` | In-repo drift statistics, three independent verdicts, report collection and rendering |
| `src/modelops_lab/traffic.py` | Deterministic dataset replay through `/predict` and `/labels` |
| `src/modelops_lab/status_page.py` | Static lifecycle status page and `status.json` |
| `src/modelops_lab/pruning.py` | Retention pruning for events and their labels |
| `src/modelops_lab/retrain.py` | Drift-justified retraining, evaluation-only gate, audit records |
| `migrations/`, `alembic.ini` | Versioned schema, including `0002_prediction_events`, `0003_prediction_event_id`, and `0004_prediction_labels` |
| `src/modelops_lab/contracts/` | Frozen Pydantic contracts and schema exporter |
| `contracts/` | Generated JSON Schemas (1.1.0), committed for review |
| `src/modelops_lab/cli.py` | Typer command surface (`data`, `db`, `train`, `evaluate`, `retrain`, `promote`, `serve`, `rollback`, `registry`, `monitor`, `simulate-traffic`, `status`, `events`) |
| `tests/unit/` | Docker-free tests of config, contracts, schemas, data, training, tracking, promotion, serving, rollback, monitoring, retraining, registration serialization, labels, traffic, pruning, status, and CLI |
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
- Drift statistics, three verdicts, reports, simulator, status page, and retention: [`docs/monitoring.md`](docs/monitoring.md)
- Retraining loop, drift justification, audit record, registration serialization, and the rollback race: [`docs/retraining.md`](docs/retraining.md)
