# Serving, prediction events, rollback, and reconciliation

W4 exposes the model that PostgreSQL says is active, records every prediction,
and makes a rollback visible on the next request without rebuilding or
restarting the service. The served artifact is always an approved model
version; the API never loads "the newest run".

## Endpoints

| Endpoint | Success | Failure |
|---|---|---|
| `GET /health` | always `200` while the process is alive; reports `database_reachable`, `model_loaded`, and `active_model_version` | never fails merely because no model is promoted |
| `GET /model` | `200` with the active descriptor | `503` with an actionable body when no model is active |
| `POST /predict` | `200` with `request_id`, `event_id`, `model_version`, `prediction` (`0`/`1`), `probability` (positive class) | `422` for invalid features, `503` without an active model, `500` if the loaded model cannot score |
| `POST /labels` | `200` with `event_id`, `label`, `labeled_at`, `updated` | `404` for an unknown event, `422` for an invalid body |
| `GET /metrics` | Prometheus text exposition | — |

`GET /health` is a process/liveness signal. A database outage reports
`status: degraded`, `database_reachable: false`, and `model_loaded: false`, but
the endpoint still answers `200`; orchestration should not restart a process
just because the registry or database is temporarily unavailable.

`GET /model` includes the model name/version, MLflow run id, dataset version and
fingerprint, artifact URI, `promoted_at`, `updated_at`, the approval decision
(`decision_id`, `approved_by`, `approved_at`, `reason`), and the feature schema
derived from the fitted pipeline.

Every request may carry `X-Request-ID`. A supplied value of 1–128 printable
characters is echoed. A longer value or one containing non-printable
characters is rejected with `422` naming the `X-Request-ID` header; an absent
or empty header generates a UUID. The identifier is returned in the response
body and in the `X-Request-ID` response header. It is a non-unique correlation
id in the event table: clients may legitimately retry with the same id, and
both attempts are persisted.

## Feature validation

The expected inputs are derived from the loaded pipeline, not from a hardcoded
schema:

- `feature_names_in_` supplies the exact ordered feature names.
- The fitted `OneHotEncoder` supplies the known categorical levels.

`POST /predict` accepts a flat JSON object. Validation rejects:

- a missing feature,
- an extra feature,
- a numeric field that is a string, boolean, `NaN`, `Infinity`, or otherwise
  non-finite,
- a categorical field that is not a string or is not a known level.

Values are never coerced. A numeric string is not parsed and a categorical
number is not stringified. A `422` body names every offending field:

```json
{
  "detail": {
    "error": "invalid_features",
    "message": "feature validation failed",
    "errors": [{"field": "sex", "message": "unknown category 'other'; expected one of: female, male"}]
  }
}
```

## Pointer, alias, and reload

On every request the service reads the singleton PostgreSQL `active_model` row.
The MLflow `champion` alias is used for artifact addressing and reconciliation,
but the pointer is authoritative (ADR-006). The alias is checked at most every
10 seconds (the pointer itself is read on every request) so the hot path stays
cheap. If they disagree, the service serves the pointer, logs a warning, and
increments `modelops_registry_alias_mismatch_total`; repair with
`modelops registry reconcile --fix`.

Loaded pipelines are cached by `(model_name, version, run id)`. A pointer change
loads the new version on the next request and the old version stays cached for a
fast roll-back. No process restart, image rebuild, or deployment is involved.
Model-load failures return `503` and increment
`modelops_model_load_failures_total`.

## Metrics

All series are exposed at `GET /metrics`:

| Metric | Type | Labels |
|---|---|---|
| `modelops_predictions_total` | counter | `model_version`, `outcome` (`success`, `invalid`, `error`) |
| `modelops_prediction_latency_seconds` | histogram | `model_version` |
| `modelops_model_load_failures_total` | counter | — |
| `modelops_registry_alias_mismatch_total` | counter | — |
| `modelops_event_writes_total` | counter | `outcome` (`logged`, `failed`) |
| `modelops_event_log_failures_total` | counter | — |
| `modelops_active_model_info` | gauge (always `1` while active) | `model_version`, `model_name`, `run_id` |

Logs are one JSON object per line with `event`, `request_id`, `model_version`,
`latency_ms`, and `status`. Feature values are never logged, and URI passwords
are redacted before any message reaches a log or client.

## Prediction events and retention

Each successful prediction is validated as a `PredictionEvent` contract and
written to `prediction_events` (Alembic revision `0003_prediction_event_id`):
`event_id` (bigint identity primary key), `request_id` (indexed, non-unique),
`timestamp` (indexed), `features` (JSONB in PostgreSQL), `prediction`,
`probability`, `model_version` (indexed), and `latency_ms`.

Event logging is best-effort. A failed insert increments
`modelops_event_log_failures_total`, writes a structured error line, and still
returns the inference response; losing an audit row never loses a prediction.
The response's `event_id` is the identity of the row that was written, or
`null` when the write failed. A null `event_id` cannot be labeled.

W5 adds `prediction_labels` (Alembic revision `0004_prediction_labels`):
`event_id` (bigint primary key, cascading foreign key to
`prediction_events.event_id`), `label` (`smallint` constrained to 0/1), and
`labeled_at` (indexed `timestamptz`). `POST /labels` performs an idempotent
upsert: the first write returns `updated: false`, a relabel returns
`updated: true` and replaces the stored label. The endpoint has no auth
because this is a local demonstration service.

```bash
curl -fsS -X POST http://localhost:8000/labels \
  -H 'content-type: application/json' \
  -d '{"event_id": 42, "label": 1}'
```

`MODELOPS_EVENT_RETENTION_DAYS` (default `30`) records the retention policy.
`modelops events prune [--days N] [--dry-run]` deletes events older than the
window and then the labels attached to those events; labels for surviving
events are never removed. W5 implements the pruning job, so the policy now has
behavior. See [`monitoring.md`](monitoring.md).

## Rollback

```bash
uv run modelops rollback \
  --to-version 1 \
  --reason "precision regressed after v2" \
  --actor release-manager \
  --database-url postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
  --tracking-uri http://localhost:5001
```

Semantics:

1. Only a version that was previously active is a valid target. The command
   searches `model_transitions` for a matching `to_model_version`; a registered
   but never-approved version is refused with exit `1`.
2. The target is resolved from MLflow, loaded, and smoke-scored on one
   schema-valid row **before** anything moves. A load or scoring failure aborts
   with exit `1` and no writes.
3. The move uses the W3 critical section: `pg_advisory_xact_lock`, a re-read of
   the pointer, and `PromotionConflict` if it changed since the command began.
   The alias is not touched before the conflict check.
4. Inside the transaction: set the alias, append a `rollback` transition, write
   `artifacts/rollbacks/<rollback_id>.json`, and update the pointer last.
5. A target equal to the current version is a no-op: exit `0`, no transition,
   no alias write, no audit record.
6. Exit codes: `0` success or no-op, `1` conflict / verification failure / any
   operational error, `2` usage error.

The audit record contains the rollback id, model name, from/to versions, MLflow
run id, actor, reason, timestamp, the original authorizing decision id, and the
pre-commit verification result (sample count, latency, detail).

## Registry reconciliation

```bash
uv run modelops registry reconcile            # compare only
uv run modelops registry reconcile --fix      # move the alias onto the pointer
```

The command prints both versions. It exits `0` when they agree and `1` on a
mismatch. With `--fix`, the alias is set to the pointer inside the same advisory
lock and expected-version check; a concurrent pointer move aborts without
touching the alias. With no active model, `--fix` removes a stray alias. This
discharges the W3/W4 reconciliation obligation from ADR-012.

## Container usage

```bash
cp .env.example .env
docker compose up -d --wait
docker compose ps
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/model
curl -fsS -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"age": 54, "bmi": 27.3, "systolic_bp": 130, "diastolic_bp": 84,
       "cholesterol": 210, "glucose": 95, "resting_heart_rate": 70,
       "max_heart_rate": 155, "st_depression": 1.2, "exercise_minutes": 30,
       "sex": "female", "chest_pain_type": "asymptomatic", "smoker": "former"}'
```

The `api` image runs as a non-root user, applies pending Alembic migrations on
start, then serves uvicorn. It depends on healthy PostgreSQL and MLflow. The
published port is `${API_PORT:-8000}`; the container port is fixed at `8000`.

## No-restart rollback proof

```text
POST /predict            -> 200 {"model_version": "2", ...}
modelops rollback --to-version 1 --reason "demo"
POST /predict            -> 200 {"model_version": "1", ...}   # same process
GET  /model              -> 200 {"model_version": "1", ...}
```

`GET /health` continues to report `model_loaded: true`; no container restart or
image rebuild occurs.

## Tests

```bash
uv run pytest -q -m "not integration and not e2e"    # SQLite/fakes
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m integration                         # real Postgres + MLflow
MODELOPS_TEST_API_URL=http://localhost:8000 \
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m e2e                                 # requires Compose
```
