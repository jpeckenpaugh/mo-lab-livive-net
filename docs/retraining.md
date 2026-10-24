# Retraining, registration serialization, and the rollback race

W6 closes the lifecycle loop. Drift can start a retraining run, but retraining
never promotes: it trains a candidate, registers it, evaluates the full
promotion gate in evaluation-only mode, and writes an audit record containing
the exact command a human must run next (ADR-002). The wave also makes same-run
registration safe across processes (ADR-012) and proves that a promotion and a
rollback racing from the same expected version cannot both win.

## The loop

```text
eligible drift evidence          operator command
reports/drift/*.json  ────────>  modelops retrain --trigger drift ...
                                            |
                                            v
                        effective config from the dataset manifest
                                            |
                                            v
                     existing training pipeline -> one MLflow run
                                            |
                                            v
                         idempotent registration (run id key)
                                            |
                                            v
                 full promotion gate, evaluation-only (dry-run)
                        /                               \
                   eligible                          rejected
                 exit 0, audit                     exit 3, audit
                        |                               |
             human runs `modelops promote`      fix data/config, repeat
```

`modelops retrain` does not write a promotion decision, does not write a
`model_transitions` row, does not change the MLflow alias, and does not move the
PostgreSQL pointer. The evaluator is the same `evaluate_gate` used by
`modelops promote`; only the mode differs.

```bash
uv run modelops retrain \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --config configs/training.yaml \
  --trigger manual \
  --reason "routine retrain after a dataset refresh" \
  --actor ml-engineer \
  --tracking-uri http://localhost:5001 \
  --database-url postgresql+psycopg://modelops:modelops@localhost:5432/modelops_test
```

Options: `--dataset` (required), `--config` (training YAML), `--trigger
manual|drift`, `--drift-report`, `--actor`, `--reason`, `--run-name`,
`--tracking-uri`, `--database-url`, `--promotion-config`, `--retrain-dir`.

### Training reflects the dataset artifact

The run is trained from the verified artifact's splits, and the effective
training config is rebound to the manifest before anything is logged:

| Setting | Source |
|---|---|
| `dataset.kind` | `manifest.source_kind` |
| `dataset.seed` | `manifest.seed` |
| `dataset.regime` | `manifest.generator_parameters.regime` (baseline for the real loader) |
| `dataset.n_samples` | `manifest.generator_parameters.n_samples`, else `manifest.row_count` |
| `split.seed` | `manifest.split_seed` |
| model hyperparameters, tracking URI | operator's `--config` |

This prevents a stale YAML value (for example `regime: baseline`) from labelling
an improved-signal artifact as baseline in MLflow. The run tags still come from
the manifest, so `dataset_fingerprint`, `dataset_version`, `regime`, and the
three split fingerprints describe the bytes that were actually trained.

## Drift justification

`--trigger drift` requires `--drift-report PATH` pointing at a JSON file written
by `modelops monitor` (a frozen `DriftReport` contract). At least one of
`data_drift`, `concept_drift`, or `performance_degradation` must be `drifted`.

A report whose verdicts are `not_drifted`, `not_evaluated`, or
`insufficient_data` is refused with an actionable error and exit `1`. Routine
retraining uses `--trigger manual`; drift-triggered retraining must be justified
by evidence. `not_evaluated` is never treated as "no drift" and never as
"drift": it is simply not evidence. Supplying `--drift-report` with
`--trigger manual` is allowed for the audit trail, but the report does not have
to show drift because the run is not claiming drift as its justification.

```bash
uv run modelops retrain \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --trigger drift \
  --drift-report reports/drift/drift-<timestamp>-v1-<id>.json \
  --reason "concept drift asserted on the current window" \
  --actor ml-engineer
```

The audit record stores `drift_report_id`, `drift_report_model_version`, and all
three verdicts exactly as the report stated. It also records a `warnings` list
when the report was produced for a model version other than the active version
at retraining time. That case is **warn-only** per the PM decision: the run
proceeds, the warning is printed, and the audit makes the stale identity
visible. Drift evidence is not refused merely because a later version is active.

## Explicit non-promotion

`perform_retraining` uses the gate's `dry-run` mode after registering the
candidate. It then re-reads the pointer and refuses to continue if it moved,
which makes "retraining cannot promote" a checked property rather than a
convention. The audit verdict is `eligible` or `rejected`; it is not an approval.

- Eligible means: the candidate would pass the same checks a human approval
  would run. The operator decides.
- Rejected means: at least one check failed. The candidate stays registered and
  unaliased, exactly like a rejected promotion candidate (ADR-012).
- `auto_promote` remains rejected by configuration validation (ADR-002).

## Exit codes and output

| Code | Meaning |
|---|---|
| `0` | Training completed and the gate is `eligible`. |
| `3` | Training completed and the gate is `rejected`; the audit record was written. |
| `1` | Any error: invalid/absent drift evidence, unreadable dataset, training or tracking failure, registration failure, gate operational failure. |
| `2` | Usage error (Typer). |

The command prints the run id, candidate version, gate verdict, the unchanged
active version, every check, the audit path, and the next command. It never
prints a traceback.

### Eligible next command

```text
modelops promote --candidate-run-id <run_id> --approve --reason '<your reason>'
```

Run it only after reviewing the audit. It goes through the normal promotion
critical section, writes a decision and transition, sets the alias, and updates
the pointer last.

### Rejected next command

```text
modelops promote --candidate-run-id <run_id> --dry-run --reason 'inspect the rejected retrained candidate'
```

Inspect the failed checks, correct the data or configuration, and retrain. Do
not approve a rejected candidate: the gate refuses it again and records another
rejection.

## Audit record

Each retraining run writes `artifacts/retraining/<retrain_id>.json`
(`--retrain-dir` overrides the directory; the default is
`<MODELOPS_ARTIFACT_ROOT>/retraining`). Decision artifacts are written under
`<MODELOPS_ARTIFACT_ROOT>/decisions` by the gate, so no test or alternate
environment writes into the repository checkout. The record is:

```json
{
  "retrain_id": "retrain-...",
  "created_at": "2026-10-24T18:04:11.123456+00:00",
  "trigger": "drift",
  "drift_report_id": "20261024T154746Z-v1-d34e6e32",
  "drift_report_model_version": "1",
  "drift_verdicts": {
    "data_drift": "not_drifted",
    "concept_drift": "drifted",
    "performance_degradation": "drifted"
  },
  "warnings": [],
  "dataset_version": "synthetic-v1.0.0-f96aa1f3bdf5",
  "dataset_fingerprint": "<64 hex>",
  "dataset_regime": "baseline",
  "dataset_source_kind": "synthetic",
  "dataset_seed": 42,
  "run_id": "<mlflow run id>",
  "candidate_version": "2",
  "active_version_before": "1",
  "checks": [
    {"name": "run_finished", "passed": true, "detail": "...", "observed_value": null, "threshold_value": null}
  ],
  "verdict": "eligible",
  "actor": "ml-engineer",
  "reason": "concept drift asserted on the current window",
  "next_command": "modelops promote --candidate-run-id <run_id> --approve --reason '...'"
}
```

The record contains identifiers, fingerprints, verdicts, check details, and the
human reason. It contains no secret, tracking URI password, or model payload.

## Registration serialization (ADR-012)

Registration is idempotent per MLflow run id. Before W6 the find-or-create
sequence was serialized only inside one Python process; two processes could both
observe "no version" and create duplicates.

`MlflowModelRegistry.register_run` now takes a serialization guard keyed by the
run id:

1. PostgreSQL: open a transaction on `MODELOPS_DATABASE_URL` and take
   `pg_advisory_xact_lock(k)` where `k` is a signed 64-bit hash of
   `modelops-registration:<run_id>`. Inside the lock, search the registered
   model for a version carrying that `run_id`; reuse it if present, otherwise
   create exactly one. The transaction-scoped lock releases on commit/rollback.
2. SQLite or no lock URL (unit tier): an in-process per-run `threading.Lock`
   under a registry-wide guard.

The promotion CLI, rollback, reconciliation, and retraining all construct the
registry with the database URL, so the cross-process lock is active in normal
operation. Integration coverage starts six threads with six separate registry
clients against the same run and asserts one version and one returned number.

The registration lock is namespaced away from the pointer advisory lock, so
registration and promotion cannot block each other by accidental key collision.

## Concurrent rollback guarantee

Promotion and rollback share the same critical section
(`PointerStore.apply_promotion` / `apply_rollback`):

1. take the pointer advisory lock;
2. re-read `active_model`;
3. compare with the version captured before the gate/verification began;
4. raise `PromotionConflict` before any alias write on a mismatch;
5. otherwise set the alias, append the transition (and decision/audit), and
   update the pointer last.

Therefore, if a promotion and a rollback both started from version `2`, exactly
one commits. The loser writes nothing: no alias move, no transition, no pointer
update. The pointer, alias, and append-only transition log remain mutually
consistent. An integration test races the two operations with a barrier and a
recording registry, asserts exactly one `set_alias` call, then deliberately
corrupts the alias and proves `modelops registry reconcile --fix` repairs it
under the same lock and expected-version check.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q                     # unit tier: fakes + SQLite
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops_test \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -q -m integration     # real Postgres + MLflow
```

The unit tier covers eligible and rejected retraining, audit contents, the
no-decision/no-pointer/no-alias guarantee, drift refusal (missing and no-drift
reports), drift evidence recording, manifest-derived config, and concurrent
registration with an unsynchronized fake. The integration tier adds the real
improved/inferior scenarios, one-version concurrent registration, and the
rollback-versus-promotion race with alias reconciliation.
