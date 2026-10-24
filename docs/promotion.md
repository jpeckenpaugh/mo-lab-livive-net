# Candidate evaluation, promotion, and the active pointer

W3 turns a finished training run into either a rejected candidate with recorded
reasons or an approved model that both the MLflow alias and the PostgreSQL
pointer agree on. Promotion is manual (ADR-002): the gate evaluates and records
everything, but a human supplies the decision and its reason.

The model is not served here. W4 loads the active version through the alias and
the pointer; W3 guarantees those two stores agree after every committed
promotion.

## What the gate checks

`modelops promote` runs the checks below in this order. Every check is recorded
with `name`, `passed`, a human-readable `detail`, and numeric
`observed_value`/`threshold_value` where a threshold applies.

| Check | Data source | Pass condition |
|---|---|---|
| `run_finished` | MLflow run status | status is `FINISHED`. |
| `model_logged` | run `outputs.model_outputs[0].model_id` | a non-empty logged-model id exists. A failed artifact load is reported in the detail and caught by later checks. |
| `evaluation_artifact` | `runs:/<run_id>/evaluation.json` | the artifact exists, downloads, and parses as a JSON object. |
| `dataset_identity` | run tag `dataset_fingerprint`, `evaluation.json`, resolved dataset manifest | the run tag and `evaluation.json` both match the manifest fingerprint, and the evaluation split is `validation` (ADR-011). |
| `required_artifacts` | `configs/promotion.yaml` logical names | `model` maps to a logged model; `evaluation.json` maps to the run artifact. The dataset manifest lives in the dataset artifact, not the run. |
| `quality_floor` | live candidate evaluation on the resolved dataset's validation split | every configured floor is met. One detail line per metric. |
| `regression_vs_active` | candidate **and** active evaluated on the **candidate's** validation slice | `candidate >= active - tolerance` for ROC-AUC, precision, recall, and F1. With no active model the check passes as `no active model (baseline promotion)` (ADR-011). |
| `smoke_test` | candidate model loaded from the run | `smoke_test.sample_count` validation rows score finite probabilities in `[0, 1]`; measured latency is within `smoke_test.max_latency_ms`. |
| `registry_consistency` | Postgres pointer vs MLflow alias | when an active model exists, the alias holder equals the pointer's version. A mismatch fails with explicit reconciliation instructions. |

Floors and tolerances come from `configs/promotion.yaml`; no threshold is
hardcoded. The calibrated floors (ROC-AUC 0.83, precision 0.70, recall 0.65,
F1 0.67; tolerance 0.02) are the values recorded in ADR-010.

A failed check always produces a `rejected` verdict. `--approve` on a failing
candidate is refused: the decision is still recorded, but named as rejected, the
failed checks become rejection reasons, and the command exits `1`. The frozen
contract independently refuses an approved decision containing a failed check.

## Human override

A human may reject a candidate whose checks all pass by passing `--reject` and a
reason (ADR-010). The verdict stays binary; the reason is stored in both
`reason` and `rejection_reasons`, and any failed check details are appended to
the reasons as well.

## Decision record

Every real decision is written twice:

1. **PostgreSQL `promotion_decisions`** — `decision_id`, `candidate_run_id`,
   `candidate_model_version`, `active_model_version_before`, `verdict`,
   `checks` (JSON), `rejection_reasons` (JSON), `actor`, `reason`, `decided_at`,
   `dataset_version`, `dataset_fingerprint`, `evaluation_split`.
2. **`artifacts/decisions/<decision_id>.json`** — exactly the frozen
   `PromotionDecision` contract (`schema_version 1.0.0`), so it round-trips
   through `PromotionDecision.model_validate`. The JSON and the database row
   carry the same decision id, verdict, checks, actor, and reason. The file is
   written before the transaction commits and removed if the commit fails.

The candidate's MLflow run is tagged (additively; training tags are untouched)
with `promotion_decision_id`, `promotion_verdict`, and `promotion_actor`.
Rejected decisions are tagged too.

No secret is stored in either record: there is no tracking URI, database URL,
password, or model payload, only identifiers, fingerprints, check details, and
the human reason.

## Pointer, alias, and failure modes

ADR-006 splits responsibilities:

- **MLflow alias** (`registry.alias`, default `champion`) addresses approved
  artifacts. It is how a server or evaluator loads an immutable model version.
- **PostgreSQL `active_model`** is the authoritative served pointer. It is a
  singleton row (`id = 1` with a check constraint), so a second row cannot
  exist. It records the model name/version, run id, logged-model id, dataset
  identity, authorizing decision id, and timestamps.
- **`model_transitions`** is append-only: `bootstrap` for the first promotion
  and `promotion` afterwards, with from/to versions, run, decision, actor, and
  reason. W4 will add `rollback`.

Approval order inside the transaction is: set the alias, insert the decision,
append the transition, update the pointer **last**. The pointer is written last
because it is the store that decides what is served.

Failure modes and their handling:

| Failure | Behavior |
|---|---|
| Another promotion moved the pointer after the gate captured its expected version | Typed `PromotionConflict`; the transaction aborts before the alias is touched. Nothing is written. Re-run the gate. |
| Alias update fails before any database write | Error; transaction rolls back; no state changes. |
| Database write fails after the alias moved | Transaction rolls back, the previous alias holder is restored, and the error says no state was committed. |
| Commit fails after the alias moved | Same restore attempt; if the restore also fails, the error explicitly says reconciliation is required and that the alias may point at the candidate. |
| Process crashes after the alias moved but before the database commit | The transaction rolls back, but the alias can point at the uncommitted candidate. The next promotion's `registry_consistency` check fails with reconciliation instructions; no pointer has moved. |
| Pointer update writes but the process dies before responding | The database state is authoritative. Re-running the same promotion is a no-op (below). |
| Re-promoting the currently active candidate | Idempotent no-op: no new decision, no new transition, no pointer write, no alias write. The CLI prints `<version> is already active; no decision or transition written` and exits `0`. |
| Rejected candidate | Decision recorded; pointer and alias unchanged. |
| `--dry-run` | Checks printed; no registration, decision, transition, pointer, or alias write. |

Registration is idempotent per run id: a run already present in the registry is
reused, never duplicated. Registration happens outside the critical section, so
a conflict can leave a harmless unaliased candidate version behind but can never
move the pointer.

### Concurrency guarantee

The critical section opens a transaction, takes
`pg_advisory_xact_lock(1297697616)`, and re-reads the pointer. If the observed
version differs from the gate's expected version, it raises
`PromotionConflict` before setting the alias or writing anything. Therefore two
concurrent promotions of different candidates against the same active version
produce exactly one winner and exactly one new transition. SQLite (the unit
tier) has no advisory lock; the expected-version check still prevents a lost
update, and only the integration tier asserts the concurrent race.

`reconcile()` compares the pointer with the alias holder and returns an
actionable detail string; the gate uses the same comparison in
`registry_consistency`.

## CLI

```bash
# Apply the migration once per database.
uv run modelops db upgrade \
  --database-url postgresql+psycopg://modelops:modelops@localhost:5432/modelops

# Preview the gate (no writes).
uv run modelops promote \
  --candidate-run-id <run_id> --dry-run --reason "preview" \
  --tracking-uri http://localhost:5001

# Approve, with a mandatory reason.
uv run modelops promote \
  --candidate-run-id <run_id> --approve \
  --reason "beats champion on the validation slice" \
  --actor release-manager \
  --tracking-uri http://localhost:5001
```

Options: `--candidate-run-id`, exactly one of `--approve` / `--reject` /
`--dry-run`, `--reason`, `--actor` (defaults to the OS user), `--dataset`
(defaults to `artifacts/datasets/<run dataset_version>`), `--config`
(`configs/promotion.yaml`), `--training-config`
(`configs/training.yaml`, for the tracking URI), `--tracking-uri`,
`--database-url` (defaults to `MODELOPS_DATABASE_URL` or the local Compose
stack).

Output prints the decision id, candidate run/version, `active: before -> after`
(or `unchanged`), the verdict, and every check with `PASS`/`FAIL` and its
detail. Exit codes: `0` approved or no-op, `1` rejected / conflict / any
operational error (one actionable line, no traceback), `2` usage error.

## Exact approved scenario

```bash
uv run modelops data prepare --config configs/training.yaml
uv run modelops data prepare --config configs/training.yaml --regime improved_signal
uv run modelops db upgrade

uv run modelops train --config configs/training.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<baseline-fingerprint> \
  --tracking-uri http://localhost:5001 --run-name baseline
uv run modelops promote --candidate-run-id <baseline-run> --approve \
  --reason "baseline passes calibrated floors" --actor demo

# Improved candidate: same features, stronger label signal (ADR-011).
MODELOPS_DATASET__REGIME=improved_signal uv run modelops train \
  --config configs/training.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<improved-fingerprint> \
  --tracking-uri http://localhost:5001 --run-name improved
uv run modelops promote --candidate-run-id <improved-run> --approve \
  --reason "improved candidate beats champion on its validation slice" --actor demo
```

Observed with seed 42: baseline ROC-AUC 0.8699 / precision 0.7576 / recall
0.7440 / F1 0.7508; improved 0.9418 / 0.8605 / 0.8555 / 0.8580. On the improved
candidate's slice the baseline scored 0.8871 / 0.7818 / 0.7457 / 0.7633, so every
delta cleared the 0.02 tolerance. The pointer moved `none -> 1 -> 2`, with
transitions `bootstrap` then `promotion`.

## Exact rejected scenario

`configs/training_inferior.yaml` is `training.yaml` with `C: 1e-4`. It is
deliberately under-fit, not sabotaged: precision, recall, and F1 collapse to 0.0
because the model predicts no positives.

```bash
uv run modelops train --config configs/training_inferior.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<baseline-fingerprint> \
  --tracking-uri http://localhost:5001 --run-name inferior

uv run modelops promote --candidate-run-id <inferior-run> --approve \
  --reason "attempt to promote the weak candidate" --actor demo
# exit 1, verdict: rejected
```

Observed: ROC-AUC 0.8514 (above its floor) but precision/recall/F1 all 0.0000,
so `quality_floor` failed and `regression_vs_active` failed against the active
v2 (precision 0.0000 vs 0.7267 and similar for recall/F1). The pointer stayed at
version 2, the alias stayed `champion -> 2`, the transition count did not change,
and the rejected decision was written to Postgres and
`artifacts/decisions/<decision_id>.json`.

## Tests

```bash
uv run pytest -q                     # unit tier: SQLite + fake registry
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m integration
```

The unit tier covers every check's pass and fail path, refusal of `--approve`
with failed checks, human rejection of an all-pass candidate, bootstrap, the
pointer/transition/decision write, the no-op, stale-version conflict with no
writes, dry-run, decision-JSON round-trip, descriptor approval metadata, and the
migration. The integration tier covers the full lifecycle, a real rejection, and
two threads racing from the same expected version.

The integration tests use a unique registered-model name per run, truncate
`promotion_decisions`, `model_transitions`, and `active_model` between tests, and
apply migrations before starting.
