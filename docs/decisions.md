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
