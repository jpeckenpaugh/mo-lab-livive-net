# Decision Log

Architecture decision records for ModelOps Lab. Newest decisions are appended.
A decision may be revisited only with a recorded reason.

---

## ADR-001 — Seeded synthetic dataset as primary workload

**Date:** 2026-10-24
**Status:** Accepted
**Context:** `concept.md` leaves open whether to use the bundled breast-cancer
dataset or a synthetic generator. The build needs controlled drift, a
repeatable "improved candidate" scenario, and unambiguous licensing.
**Decision:** Ship a seeded synthetic binary-classification generator as the
default workload. Keep a breast-cancer loader as an optional alternate behind
configuration, documented with provenance and usage terms.
**Rationale:** Synthetic data makes drift and improvement constructible and
deterministic, avoids redistribution questions, and removes noise from the
promotion scenarios. A real dataset remains available to show the pipeline is
not synthetic-only.
**Consequences:** We own generator quality and must document its schema and
regime construction. All acceptance scenarios use the synthetic default.
**Revisit if:** The demo needs to prove behavior on externally produced data.

---

## ADR-002 — Manual promotion behind a policy-checked gate

**Date:** 2026-10-24
**Status:** Accepted
**Context:** The concept allows manual promotion for the MVP and treats
automated promotion as optional.
**Decision:** The gate always evaluates and records every check, but promotion
requires an explicit human command with a reason. An `auto_promote` config flag
exists and is validated, defaulting to `false`; enabling it is a later feature.
**Rationale:** Matches the spec's caution against assuming newer is better, and
keeps a single code path for both modes.
**Consequences:** Promotion decisions are auditable artifacts. Automated
promotion must reuse the same gate and lock, not bypass them.
**Revisit if:** Continuous deployment of models becomes a requirement.

---

## ADR-003 — PostgreSQL for prediction events and the active-model pointer

**Date:** 2026-10-24
**Status:** Accepted
**Context:** The concept asks whether inference events belong in Postgres or a
simpler store. The active-model pointer must remain consistent under concurrent
promotion and rollback.
**Decision:** PostgreSQL holds both the prediction-event table and the
active-model pointer with its history. MLflow keeps artifacts and run metadata.
**Rationale:** Postgres is already required by MLflow, provides transactions
and row-level locking, and keeps the concurrency guarantee in one place.
**Consequences:** Unit tests still use a disposable database or repository
double; integration tests exercise real Postgres. Schema migrations are
required and versioned.
**Revisit if:** The event volume outgrows a single Postgres instance.

---

## ADR-004 — Observability surface

**Date:** 2026-10-24
**Status:** Accepted
**Context:** The concept offers MLflow UI, a minimal dashboard, or generated
reports.
**Decision:** Provide all three lightweight options: the stock MLflow UI,
generated HTML + JSON reports for drift and promotion decisions, and one
minimal static status page showing the active model, recent decisions, and the
latest drift verdict.
**Rationale:** The status page is the fastest way to make lifecycle state
legible without building a SPA.
**Consequences:** The status page is generated, not interactive, and must never
become a privileged control surface.
**Revisit if:** Interactive operations become necessary.

---

## ADR-005 — In-repo drift statistics with Evidently optional

**Date:** 2026-10-24
**Status:** Accepted (technical)
**Context:** Evidently's API changes across releases and pulls a large
dependency tree. The concept permits "a comparable statistical implementation."
**Decision:** Implement the statistical core in-repo (per-feature distribution
tests, PSI, missingness, aggregate verdicts, minimum-sample guards) behind a
stable interface. Use Evidently, if installed, only to render an HTML report.
**Rationale:** Keeps tests fast and deterministic, isolates third-party churn,
and guarantees the drift/performance separation is our own explicit logic.
**Consequences:** We own the statistical correctness and its tests. Evidently
becomes optional, not load-bearing.
**Revisit if:** Evidently stabilizes into a clearly better supported surface.

---

## ADR-006 — Registry aliases for artifacts, Postgres for the served pointer

**Date:** 2026-10-24
**Status:** Accepted (technical)
**Context:** MLflow deprecated registry stages in favor of aliases; serving must
load an explicitly approved version rather than the latest artifact.
**Decision:** Use an MLflow alias (for example `champion`) to mark approved
artifacts, and treat the Postgres `active_model` row as authoritative for what
the API serves. Both are updated in the same promotion transaction, with the
Postgres row written last and logged.
**Rationale:** Separates immutable artifact addressing from mutable deployment
state, and gives one transactional place to enforce concurrency.
**Consequences:** Recovery must reconcile the two stores; a reconciliation check
is part of the promotion smoke test.
**Revisit if:** MLflow gains first-class transactional deployment state.

---

## ADR-007 — Python 3.12 via uv

**Date:** 2026-10-24
**Status:** Accepted (technical)
**Context:** The host has Python 3.9.6; the concept requires 3.12+.
**Decision:** Use `uv` to manage a Python 3.12 interpreter and dependency lock;
commit `uv.lock`. The container image uses `python:3.12-slim`.
**Rationale:** Deterministic environments and fast CI, without depending on the
host interpreter.
**Consequences:** All documented commands run through `uv run`.
**Revisit if:** The project standardizes on another toolchain.

---

## ADR-008 — Tiered test strategy

**Date:** 2026-10-24
**Status:** Accepted (technical)
**Context:** Requiring Docker for every test slows the build and makes local
development fragile.
**Decision:** Three tiers: unit (pure, no Docker), integration (real Postgres
and MLflow), e2e (Compose). Markers select the tier; CI runs unit on every push
and the heavier tiers on the integration job.
**Rationale:** Fast feedback without weakening end-to-end evidence.
**Consequences:** Domain logic must be injectable so unit tests can avoid
infrastructure.
**Revisit if:** Integration coverage becomes cheap enough to run everywhere.

---

## ADR-009 — Contract schema versioning

**Date:** 2026-10-24
**Status:** Accepted
**Context:** The five frozen contracts carry `schema_version`. Without a rule,
field changes could silently invalidate stored manifests, decisions, and
events.
**Decision:** Contract fields are frozen at `1.0.0`. Removing, renaming,
retyping, or making an optional field required is a **major** bump. Adding a
new optional field is a **minor** bump. Documentation-only edits are a patch.
Every bump is recorded here and regenerated schemas are committed; the
byte-comparison test enforces the checked-in state.
**Rationale:** Promotion decisions and drift reports are audit artifacts; a
stable, explicit version is how a reader knows which fields to trust.
**Consequences:** Later waves must not edit a frozen contract casually. If a
field is missing during implementation, raise it to the PM rather than
mutating `1.0.0`.

**1.1.0 (W4, 2026-10-24):** PM-approved widening of
`PredictionEvent.features` from `dict[str, float]` to `dict[str, float | str]`
so prediction events can carry the three categorical features of the primary
workload. All five contracts share one global `schema_version`, so the
`SchemaVersion` literal accepts both `1.0.0` and `1.1.0`, newly written records
emit `1.1.0`, and committed JSON Schemas were regenerated. No other contract
field changed. Existing `1.0.0` dataset manifests and promotion decisions
still parse (covered by tests).

---

## ADR-010 — Gate semantics and threshold calibration

**Date:** 2026-10-24
**Status:** Accepted
**Context:** W0 shipped provisional quality floors (ROC-AUC 0.85, P/R/F1 0.75,
tolerance 0.02) before any model was measured. Three related semantics needed
a ruling.
**Decision:**
1. The floors are **provisional** and must be recalibrated in W2 against the
   measured baseline, with the change recorded in this log.
2. A human may **reject an all-pass candidate** by supplying a reason. The
   verdict stays binary; the audit trail carries the rationale.
3. `auto_promote: true` is **rejected by validation** until automated promotion
   is implemented. The flag documents intent without implying capability.
4. Concept-drift and performance thresholds are **designed in W5**, not
   guessed now. A signal without labels must report `not_evaluated`, never
   `not_drifted`.
**Rationale:** Honest defaults beat plausible-looking numbers, and the
distinction between "no drift observed" and "not measured" is central to the
concept's warning against inferring degradation from drift.
**Consequences:** W2 and W5 own explicit threshold-calibration tasks. W3's
promotion output is only meaningful after W2 updates the floors.

**W2 calibration (2026-10-24):** the provisional floors were replaced after
measuring baseline runs across five seeds (42, 7, 123, 2024, 31337). New floors:
ROC-AUC 0.83, precision 0.70, recall 0.65, F1 0.67; tolerance unchanged at 0.02.
The old recall floor of 0.75 was above the measured baseline (0.744) and could
never have been met. The new floors sit 0.026–0.051 below the worst measured
baseline value per metric. Known rejection cases: a severely under-fit model
(`C=1e-4`, precision/recall/F1 all 0.0) and a model trained on the
covariate-shift regime (ROC-AUC 0.7948).

---

## ADR-011 — Dataset regime semantics and comparison slice

**Date:** 2026-10-24
**Status:** Accepted
**Context:** W1 shipped three synthetic regimes used by later waves. Their
meaning determines whether the drift and promotion demos are honest.
**Decision:**
1. `covariate_shift` changes **P(X) only**. Coefficients, standardization
   constants, and the logistic link are shared with `baseline`; the
   implementation proves this by construction and tests `P(y|X)` equality. Any
   AUC drop under this regime is a distributional effect, **not** concept drift.
2. `improved_signal` changes **P(y|X) only**, reusing baseline feature draws.
   Coefficients are reweighted rather than uniformly scaled because AUC is
   invariant to monotone score transforms; uniform scaling could not produce a
   real improvement.
3. The optional breast-cancer loader inverts scikit-learn's labels so
   `target=1` means malignant. The transform is recorded in the manifest and
   the dataset documentation.
4. W3 compares a candidate against the active model **on the candidate's
   validation slice**, never each model on its own slice. This keeps the
   improved-regime comparison fair and prevents a promoted model from being
   judged on data it was tuned against.
**Rationale:** The concept explicitly forbids inferring performance degradation
from drift. That distinction has to be built into the data, not asserted in
prose.
**Consequences:** W5 reports data drift, concept drift, and observed performance
as three separate verdicts. W3's gate must record the evaluation slice in its
decision record.
**Revisit if:** A regime is added that intentionally changes both P(X) and
P(y|X).

---

## ADR-012 — Promotion state machine and concurrency guarantee

**Date:** 2026-10-24
**Status:** Accepted
**Context:** W3 introduced registration, the gate, the active pointer, and
concurrent promotion. The ordering between the registry alias and the database
determines what a failure can leave behind.
**Decision:**
1. Candidates are registered **before** gate evaluation, idempotently per run.
   A rejected candidate stays registered but unaliased; rejection is an audit
   outcome, not a registration rollback.
2. The gate records nine checks into a `PromotionDecision`, persisted to
   Postgres, to `artifacts/decisions/<id>.json`, and as additive tags on the
   candidate run. An approved decision cannot contain a failed check.
3. Approval takes a transaction-scoped Postgres advisory lock, re-reads the
   pointer, and aborts with `PromotionConflict` if it differs from the version
   captured when the gate began. It then sets the alias, writes the decision and
   transition, and updates the pointer **last**.
4. Re-promoting the active candidate is an explicit no-op: exit 0, no decision,
   no transition.
5. A crash after the alias move but before commit attempts to restore the
   previous alias and reports that reconciliation is required. Serving reads the
   pointer, never the alias, so the served model remains correct.
**Rationale:** One serialized critical section makes "exactly one promotion
wins" a property of the database, not of process timing. Writing the pointer
last keeps the least recoverable failure on the least authoritative store.
**Consequences:** W4 rollback must reuse the same lock and expected-version
check, and ship a reconciliation command. W6 must serialize same-run
registration across processes before concurrent retraining is allowed.

**W4 discharge (2026-10-24):** `modelops rollback` and
`modelops registry reconcile --fix` both run inside the same
`pg_advisory_xact_lock` critical section with the expected-version check; a
conflict leaves the alias untouched. Verification (load + smoke prediction)
happens before the lock, so an unservable target aborts with no writes.
**Revisit if:** MLflow offers transactional pointer semantics or a second writer
is introduced.

---

## ADR-013 — PredictionEvent represents categorical inputs (schema 1.1.0)

**Date:** 2026-10-24
**Status:** Accepted
**Context:** `PredictionEvent.features` was `dict[str, float]`, but the primary
synthetic workload has three categorical features. As written, no valid event
could be recorded for the default model, which would have silently undermined
serving audit and W5 monitoring.
**Decision:** Widen features to `dict[str, float | str]`. Because the five
contracts share one global schema version, `SchemaVersion` now accepts both
`1.0.0` and `1.1.0`, emits `1.1.0`, and all committed schemas are regenerated.
Existing `1.0.0` artifacts remain readable and are covered by tests.
**Rationale:** Raw feature values, including category labels, are what drift
monitoring must compare against the training reference. Encoding categories as
floats or one-hot vectors would either distort the comparison or change the
schema invisibly.
**Consequences:** Consumers of prediction events must handle string values. No
prediction event existed before this change, so no data migration is required.

---

## ADR-014 — Prediction events are identified by event_id, not request_id

**Date:** 2026-10-24
**Status:** Accepted
**Context:** W4 initially made `request_id` the primary key. A client retry with
the same `X-Request-ID` therefore failed the best-effort insert and silently
dropped a monitoring event.
**Decision:** `prediction_events` uses a bigint `event_id` identity primary key.
`request_id` is a non-unique indexed column, alongside indexes on timestamp and
model version for window queries. Client request IDs longer than 128 characters
or containing non-printable characters are rejected with a 422 naming the
header. Duplicate request IDs produce distinct events.
**Rationale:** Request IDs identify a caller's logical request, not a storage
row; retries are normal and must not erase monitoring data.
**Consequences:** Exactly-once event semantics are explicitly out of scope.
W5 queries work on events, not on request identity.

---

## ADR-015 — Monitoring verdict semantics

**Date:** 2026-10-24
**Status:** Accepted
**Context:** W5 had to report data drift, concept drift, and observed performance
degradation without inferring one from another. The existing `improved_signal`
regime makes an old model perform *better*, so it cannot supply evidence of
concept drift.
**Decision:**
1. Add a `concept_shift` regime: identical feature draws to `baseline`, with a
   rotated label mechanism that makes a baseline-trained model degrade by more
   than 0.05 ROC-AUC. This is concept drift with stable inputs.
2. Reference window = the active model's own dataset, validation split; current
   window = the most recent events for the active model version.
3. Data drift uses per-feature PSI and KS (numeric) or categorical PSI, plus
   missing-rate deltas, with an aggregate share threshold. Insufficient samples
   yield `insufficient_data`, never `not_drifted`.
4. Performance degradation requires delayed labels, compares ROC-AUC against the
   same model on the reference split, and applies a configured maximum drop and
   minimum labeled sample count.
5. Concept drift is asserted only when performance degrades while data drift is
   **not** flagged. When both are present the verdict is `not_evaluated` with a
   detail saying the cause is confounded; when labels are absent it is
   `not_evaluated`.
6. If the model cannot be loaded, the report is still written with data drift
   evaluated and both dependent verdicts `not_evaluated`, naming the operator
   error without secrets or a traceback.
**Rationale:** A covariate shift changes the input distribution and can degrade
performance without changing P(y|X). Reporting that as concept drift would
violate the concept's central warning. Requiring stable inputs before asserting
concept drift keeps the three signals honest and demonstrable.
**Consequences:** The demo shows three distinct evidence patterns. Event
retention pruning deletes labels for deleted events only. The status page is
generated, read-only, and never a control surface.
**Revisit if:** A reweighted or stratified concept-drift test replaces the
stable-input rule.
