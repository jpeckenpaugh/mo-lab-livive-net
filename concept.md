# ModelOps Lab

## Concept

**ModelOps Lab** is a lightweight, self-hosted reference implementation of a complete machine learning operations (MLOps) lifecycle. Its purpose is to demonstrate how a model moves from a validated dataset through reproducible training, experiment tracking, registration, controlled deployment, production monitoring, drift detection, and retraining.

The primary product is **the operational lifecycle**, not the predictive model or a user-facing application. A deliberately simple scikit-learn classifier serves as the workload so that the system's engineering controls remain visible and understandable.

The project should run locally with Docker Compose, require no paid cloud services, and provide a reproducible end-to-end demonstration.

## Objectives

1. Demonstrate an auditable path from source data to a deployed model version.
2. Make training runs reproducible through pinned dependencies, dataset versions, configurations, and recorded metrics.
3. Enforce explicit quality gates before a candidate model can replace the active production model.
4. Monitor inference traffic and incoming data for distribution changes and, when labels are available, predictive performance degradation.
5. Trigger a retraining workflow and evaluate the resulting candidate without automatically assuming that newer is better.
6. Support rollback to a previously approved model.
7. Expose operational state through APIs, logs, experiment records, and a minimal dashboard or generated report.

## Non-goals

- Developing a state-of-the-art predictive model.
- Building a general-purpose enterprise ML platform or Kubernetes operator.
- Requiring GPUs, hosted foundation models, or proprietary infrastructure.
- Automatically promoting every retrained model.
- Treating data drift as proof of model performance degradation.
- Building a sophisticated frontend.

## Reference workload

Use a small, public, tabular binary-classification dataset with an explicit license and documented provenance. The initial candidate is the scikit-learn breast cancer dataset, which is bundled for convenient experimentation; verify its source and usage terms before publishing redistributed data. Alternatively, use a synthetic dataset generated with a fixed seed to simplify reproducibility and controlled drift scenarios.

The baseline model is a scikit-learn `LogisticRegression` pipeline with preprocessing fitted only on training data. A second candidate model or alternative hyperparameter configuration enables meaningful experiment comparisons.

Data is split into training, validation, and held-out test partitions with fixed seeds and recorded split manifests. The held-out test set must not be used repeatedly to tune the promotion policy.

## Core lifecycle

```text
Dataset / Generator
       |
       v
Schema + Quality Validation
       |
       v
Versioned Training Pipeline
       |
       v
MLflow Experiment Tracking
       |
       v
Candidate Evaluation + Promotion Gate
       |
       v
Model Registry / Approved Version
       |
       v
FastAPI Inference Service
       |
       v
Prediction Logs + Data Monitoring
       |
       v
Drift Alert / Retraining Trigger
       |
       +------> New Candidate Training Run

Approved model versions can be rolled back independently of retraining.
```

### 1. Dataset ingestion and validation

- Load a known dataset snapshot or generate a seeded synthetic dataset.
- Validate required columns, types, missing values, target distribution, and basic ranges.
- Record dataset fingerprint, source, schema version, and train/validation/test split configuration.
- Fail clearly when required inputs violate the declared schema.

### 2. Training and experiment tracking

- Execute training from a single CLI command or scheduled job.
- Log parameters, code revision, dependency/environment metadata, dataset fingerprint, metrics, and serialized model artifacts to MLflow.
- Support repeated runs and side-by-side comparison of candidate models.
- Record class balance, confusion matrix, precision, recall, F1, ROC-AUC where applicable, and evaluation sample counts.

### 3. Candidate evaluation and promotion

A candidate is promoted only when it passes documented checks, such as:

- Required data and evaluation artifacts exist.
- The candidate meets a minimum quality floor on validation data.
- The candidate does not regress beyond an allowed tolerance relative to the active production model on the same evaluation slice.
- Inference contract and model-loading smoke tests pass.
- A promotion decision, including rejection reasons, is recorded.

Initial thresholds are configuration values, not universal ML quality standards. Promotion may be manual in the MVP; automated promotion is an optional later mode.

### 4. Model serving

Provide a FastAPI service with endpoints such as:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process health |
| `GET /model` | Active model version and metadata |
| `POST /predict` | Validated inference request and response |
| `GET /metrics` | Basic operational counters and latency metrics |

Load an explicitly approved model version rather than implicitly serving the most recently trained artifact. Include model version and request identifier in inference logs.

### 5. Monitoring and drift detection

Capture input features, model version, prediction, timestamp, and a request identifier, using local storage and a clear retention policy. Avoid collecting sensitive personal information in the demo.

Use Evidently or a comparable statistical implementation to compare a reference dataset with a recent inference window. Report per-feature and aggregate distribution shifts with configurable thresholds and minimum sample sizes.

When delayed ground-truth labels are available, separately calculate predictive performance metrics. Distinguish **data drift**, **concept drift**, and **observed performance degradation**; do not infer one from another without supporting evidence.

### 6. Retraining and rollback

- Support a CLI/API command to launch retraining using an explicitly selected, versioned dataset.
- Produce a new candidate and evaluate it against the currently approved version.
- Retain the production model if the candidate fails its promotion gate.
- Support explicit rollback to a previously approved artifact, with a recorded reason and verification step.
- Prevent concurrent promotion operations from leaving the active model pointer inconsistent.

## Demonstration scenarios

The project should ship with scripted scenarios that can be executed independently:

| Scenario | Expected demonstration |
|---|---|
| Baseline | Validate data, train, evaluate, register, approve, and serve model v1 |
| Reproducibility | Repeat a run with identical inputs/configuration and compare outputs and metadata |
| Input drift | Shift feature distributions in simulated inference traffic and generate a drift report |
| Candidate rejected | Train a deliberately inferior candidate; promotion gate keeps v1 active |
| Candidate approved | Train an improved candidate on an appropriate dataset and promote after checks |
| Rollback | Restore a previously approved version and verify predictions identify the restored version |
| Invalid input | Reject malformed inference or training inputs with actionable errors |

Drift should initiate an investigation or candidate-training workflow, **not** force a production model replacement.

## Proposed architecture

**Runtime:** Python 3.12+, Docker Compose.

**ML:** scikit-learn, pandas, NumPy.

**Experiment tracking and registry:** MLflow with PostgreSQL metadata storage and local artifact volume.

**API:** FastAPI + Pydantic.

**Monitoring:** Evidently for data quality/drift reports; structured logs and basic metrics for serving.

**Orchestration:** Python CLI and explicit jobs initially; a scheduled runner can be added without introducing Airflow or Kubernetes.

**CI:** GitHub Actions for unit tests, pipeline integration tests, container build, and promotion-policy checks.

**UI:** MLflow's existing UI plus a minimal static status page or generated HTML reports if useful. No custom SPA required.

## Suggested repository structure

```text
modelops-lab/
├── concept.md
├── README.md
├── compose.yaml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── configs/
│   ├── training.yaml
│   ├── promotion.yaml
│   └── monitoring.yaml
├── src/modelops_lab/
│   ├── data.py
│   ├── validation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── registry.py
│   ├── promotion.py
│   ├── serving.py
│   ├── monitoring.py
│   └── retrain.py
├── scripts/
│   ├── demo_baseline.py
│   ├── demo_drift.py
│   ├── demo_rejection.py
│   └── demo_rollback.py
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   └── architecture.md
└── .github/workflows/
    └── ci.yaml
```

The structure is a starting point, not a requirement to create every file before its behavior is implemented.

## MVP acceptance criteria

The MVP is complete when a new developer can:

1. Clone the repository and start required services using Docker Compose.
2. Run a documented command that validates data, trains a model, and records a complete MLflow experiment.
3. Evaluate and approve a candidate through a reproducible promotion gate.
4. Call `/predict` and see which approved model version served the request.
5. Generate simulated traffic with a controlled feature shift and obtain a drift report.
6. Run retraining and demonstrate both an accepted and a rejected candidate.
7. Roll back to an earlier approved version without rebuilding the service.
8. Run automated tests locally and in CI.

All eight behaviors should be observable through logs, API responses, MLflow records, or saved reports.

## Implementation phases

**Phase 1 — Reproducible baseline:** Dataset validation, training, evaluation, MLflow tracking, Docker Compose, and tests.

**Phase 2 — Controlled deployment:** Model registry, promotion policy, FastAPI serving, version metadata, and rollback.

**Phase 3 — Operational monitoring:** Prediction logging, reference-vs-current data comparison, drift reports, and controlled traffic generation.

**Phase 4 — Closed-loop demonstration:** Retraining trigger, candidate evaluation, rejection/approval scenarios, end-to-end scripts, and CI validation.

## Success criteria

The repository should clearly demonstrate these MLOps capabilities through executable code and reproducible evidence: **data validation, experiment tracking, artifact versioning, model registry, deployment gating, model serving, monitoring, drift detection, retraining, rollback, and CI/CD**.

The README should lead with a concise architecture diagram, a quick-start command sequence, and a reproducible scenario that shows the active model version before and after a promotion decision. A successful project is one where the operational decisions can be inspected and reproduced, not merely described.

## Open design decisions

- Use a bundled public dataset or a fully synthetic, controllable dataset for the initial release?
- Require human approval for every promotion, or allow a configurable automated promotion policy?
- Persist inference events in PostgreSQL or use a simpler local event store for the MVP?
- Build a minimal dashboard, or rely on MLflow and generated monitoring reports?

**Working principle:** Keep the model simple; make the lifecycle rigorous.
