# ModelOps Lab — Build Plan

**Owner:** Project manager (agent)
**Source spec:** [`concept.md`](../concept.md)
**Last updated:** 2026-10-24

This document is the execution plan for the concept. Where it conflicts with
`concept.md`, the concept wins; this file only adds sequencing, contracts, and
working agreements.

## Scope

Deliver the eight MVP acceptance criteria in `concept.md` as executable,
reproducible evidence. The model stays deliberately simple; the lifecycle is
the product.

Out of scope: alternate datasets beyond the optional loader, custom SPA,
Kubernetes, GPUs, hosted services, automated promotion (flag reserved only).

## Locked decisions

See [`decisions.md`](decisions.md) for rationale. Summary:

| ID | Decision |
|---|---|
| ADR-001 | Seeded synthetic dataset is the primary workload; breast-cancer loader is optional |
| ADR-002 | Promotion is manual, gated by a policy-checked, recorded decision; `auto_promote` flag reserved |
| ADR-003 | PostgreSQL stores prediction events and the active-model pointer |
| ADR-004 | Observability = MLflow UI + generated HTML/JSON reports + one minimal static status page |
| ADR-005 | Drift statistics are implemented in-repo; Evidently is an optional report renderer |
| ADR-006 | MLflow registry aliases are authoritative for artifacts; our Postgres table is authoritative for the served pointer |
| ADR-007 | Python 3.12 managed by `uv`; `uv.lock` is committed |
| ADR-008 | Tests are tiered: unit (no Docker), integration (Postgres/MLflow), e2e (Compose) |

## Stages and waves

Every wave ends with something runnable, tested, documented, and demonstrable.
A wave is not complete until its exit gate passes from a clean environment.

| Wave | Stage | Goal | Exit gate | Size |
|---|---|---|---|---|
| W0 | Frame | Contracts, scaffold, CI stub, CLI skeleton | `uv run pytest` + lint green; contracts reviewed; no domain logic | S |
| W1 | Phase 1 | Data generation/loading, fingerprinting, validation, split manifests | Two independent preparations produce identical fingerprints and split hashes; malformed input fails clearly | M |
| W2 | Phase 1 | Training pipeline + MLflow tracking | Complete, inspectable run; no leakage; rerun reproduces metrics/artifact hash | M |
| W3 | Phase 2 | Registry, transactional active pointer, promotion gate | Inferior candidate rejected with reasons; better approved; concurrent promotions → exactly one wins | L |
| W4 | Phase 2 | FastAPI serving, prediction events, rollback | `/predict` reports serving version; 422s are actionable; rollback visible next request, no rebuild | M |
| W5 | Phase 3 | Monitoring, drift reports, traffic simulator, status page | Shift flagged; clean traffic not flagged; small samples guarded; drift ≠ performance loss | L |
| W6 | Phase 4 | Retraining loop, rollback hardening | Failed candidate leaves active untouched; rollback verified; operations idempotent | M |
| W7 | Evidence | Demo scripts, README, architecture/acceptance docs, full CI | Fresh clone → documented sequence → all 8 criteria observed; CI green | M |

Milestones: **W2** = Phase 1, **W4** = Phase 2, **W5** = Phase 3, **W7** = MVP.

## Frozen contracts

Defined as Pydantic models in W0 and exported as JSON Schema. All later waves
consume these; changes require a PM decision and a version bump.

1. **Dataset manifest** — dataset version, source kind, seed, schema version,
   row count, fingerprint, split hashes and row counts.
2. **Model descriptor** — model version, MLflow run ID, dataset fingerprint,
   metrics block, artifact URI, approval metadata.
3. **Promotion decision** — candidate version, active version, individual
   checks with pass/fail and detail, verdict, rejection reasons, actor,
   timestamp.
4. **Prediction event** — request ID, timestamp, features, prediction,
   probability, model version, latency.
5. **Drift report** — reference window, current window, sample counts,
   per-feature statistics, thresholds, and three independent verdicts:
   data drift, concept drift, observed performance degradation.

## Carried obligations

Commitments made now that must be discharged by a later wave.

| Wave | Obligation | Source |
|---|---|---|
| W1 | Manifest records provenance for both the generator and the optional real loader | ADR-001 |
| W2 | Measure the baseline, recalibrate the provisional quality floors, record the change | ADR-010 |
| W3 | Gate uses the calibrated floors; tests cover human rejection of an all-pass candidate | ADR-010 |
| W5 | Define concept-drift and performance-degradation thresholds; absent labels → `not_evaluated` | ADR-010 |
| W7 | Choose and add the repository license before any public announcement | PM open item |

## Verification model

- **Builder → self-review → PM.** Each wave runs as: PM writes the task spec →
  builder implements and adds tests → builder performs a review pass and
  returns open questions with options and a recommendation → PM decides.
- **Independent verification.** The builder's own test run is not sufficient
  evidence. Before a wave is declared done, the exit gate is re-run by a
  separate verifier session (or the PM) against the working tree.
- **Evidence, not claims.** Each gate maps to a command and observable output,
  recorded in `docs/acceptance.md` from W7 onward.
- **Tiered tests.** Unit tests must not require Docker. Integration tests cover
  Postgres and MLflow. End-to-end tests use Compose and run in CI.

## Reproducibility rules

- Fixed seeds for data generation and splits; split manifests are persisted.
- Preprocessing is fit on training data only, enforced by a leakage test.
- MLflow runs record git revision, lockfile hash, Python version, dataset
  fingerprint, and full config.
- Thread counts for BLAS are pinned so floating-point results are stable.
- Reproducibility compares artifact hashes, metrics, and metadata — never
  MLflow run IDs or timestamps.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| MLflow / Evidently API churn | Rework, flaky tests | Pin versions; wrap behind internal interfaces |
| Compose stack slow on macOS | Integration flakiness | Healthchecks, pinned images, Docker-free unit tier |
| Pointer inconsistency under concurrency | Wrong model served | Postgres row lock; two-process test |
| "Improved candidate" not reproducible | Demo looks fabricated | Synthetic generator with a constructible regime shift |
| Determinism polluted by BLAS threads | Reproducibility gate fails | Pinned `random_state` and thread env |
| Scope growth into platform/UI work | Never ships | MLflow UI + generated pages only |

## Working agreements

- `main` is the integration branch. The PM commits and pushes completed work;
  subagents do not commit or push.
- One builder owns a wave; parallel agents are used only where the dependency
  chain genuinely allows it (docs/CI, drift statistics, verification).
- No wave begins before the previous exit gate is recorded as passing.
- Config is YAML validated by Pydantic; secrets come from environment only.
- Every user-visible behavior is observable through logs, API responses,
  MLflow records, or saved reports.
