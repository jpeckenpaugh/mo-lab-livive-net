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
