# Monitoring, drift statistics, and three independent verdicts

W5 implements the monitoring stage behind MVP criterion 5. It captures
prediction events (W4), compares a current inference window with the active
model's own reference split, computes data drift, delayed-label performance
degradation, and concept drift **independently**, and writes JSON + HTML
reports and one static status page.

The statistics are implemented in-repo (ADR-005). There is no Evidently
dependency and no JavaScript in any generated artifact.

## Windows

| Window | Source | Selection |
|---|---|---|
| reference | the active model's verified dataset artifact | `reference.source` split (`validation_split` by default), first `reference.window_size` rows in canonical `row_id` order |
| current | `prediction_events` for the monitored model version | most recent `current.window_size` rows, ordered chronologically |

The reference split is the model's **actual** dataset, resolved from the
pointer's `dataset_version` (or `--reference-dataset`). Validation is the
default because it is the slice the promotion gate used. The performance
reference is always the validation split, even if `reference.source` is
`training_split`: performance must be measured on data the model did not fit.

`configs/monitoring.yaml` uses a 400-row reference window (the validation split
size) and a 500-row current window. A current window below `min_sample_size`
(100) is `insufficient_data`, never `not_drifted`.

## Statistics and formulas

All comparisons use the declared synthetic schema: ten numeric features and
three categorical features. Features are never encoded before comparison;
categorical levels stay as strings.

### Numeric: two-sample KS

`scipy.stats.ks_2samp(reference, current)` returns the maximum empirical CDF
distance `D` and its p-value. The feature is flagged when `p < 0.05`. KS is
sensitive to any distributional difference, so a small shift in scale can be
significant even when ranking is unaffected.

### Numeric: PSI with reference deciles

Bins are the reference deciles plus infinite tails, so the reference is
roughly 10% per bin:

```text
edges = [-inf, q10(ref), q20(ref), ..., q90(ref), +inf]
psi   = sum((cur_share - ref_share) * ln(cur_share / ref_share))
```

Zero shares are floored at `1e-6` in **both** windows, so an empty bin in both
is exactly zero and a bin present only in one window is finite. PSI is
undefined (null) when either window has no numeric values.

### Categorical: PSI over frequency distributions

The category set is the union of reference and current levels. An unseen
current category contributes a positive divergence instead of being dropped.
Missing values are excluded from the PSI and reported separately. There is no
KS test for categorical features.

### Missing-rate delta

`abs(current_missing_rate - reference_missing_rate)`, where missing means
null/NaN/non-numeric for numeric features and null/NaN for categoricals. A
delta of at least `0.05` flags the feature even when the observed values did
not move.

### Per-feature and aggregate rule

A feature is `drifted` when **any** rule fires:

- numeric: `psi >= 0.2` OR `ks_pvalue < 0.05` OR `missing_delta >= 0.05`;
- categorical: `psi >= 0.2` OR `missing_delta >= 0.05`.

The report detail names every rule that fired with its observed value. The
`data_drift` verdict is `drifted` when the share of drifted features is at
least `aggregate_feature_share_threshold` (0.3). `data_drift.score` is that
share.

## The three verdicts

`DriftReport` carries three separate `SignalVerdict` fields. They are never
derived from one another.

### `data_drift`

Computed from the statistics above. `insufficient_data` below
`min_sample_size`; otherwise `drifted` or `not_drifted`.

### `performance_degradation`

Requires delayed labels joined from `prediction_labels`.

1. The reference ROC-AUC is the monitored model scored on the dataset's full
   validation split.
2. The current ROC-AUC is computed from the stored serving probabilities of
   the labeled current-window events.
3. `delta = current - reference`; degradation is `reference - current >
   0.05`.
4. Below `min_labeled_samples` (100) the verdict is `insufficient_data`.
   With no labels at all it is `not_evaluated` with no `PerformanceSnapshot`.
   If the labeled window contains only one observed class, ROC-AUC is
   undefined and the verdict is also `insufficient_data`.

A `PerformanceSnapshot` records `metric`, `reference_value`,
`current_value`, `delta`, and `labeled_sample_count` whenever the AUC is
computable.

**Degradation behavior.** Monitoring must not go dark because the model
artifact is unavailable. If the monitored model cannot be loaded or the
reference validation slice cannot be scored, `modelops monitor` still computes
`data_drift` from the feature windows and writes both report files.
`performance_degradation` and `concept_drift` are then `not_evaluated`, and
their details name the failure in one operator-facing line (exception type and
message, secrets redacted, no traceback). The command exits `0` unless
`--fail-on-drift` is set and data drift is flagged. Exit `1` is reserved for
failures that prevent a report at all, such as an invalid configuration or a
missing reference dataset.

### `concept_drift` — the explicit rule

| Labels | Performance | Data drift | Concept drift |
|---|---|---|---|
| none | — | — | `not_evaluated` |
| fewer than `min_labeled_samples` | — | — | `insufficient_data` |
| enough | degraded | **not** flagged | `drifted` |
| enough | degraded | flagged | `not_evaluated` — cause confounded by data drift; concept drift is not asserted |
| enough | not degraded | any | `not_drifted` |

This is the rule that keeps the covariate-shift scenario honest. In
`covariate_shift`, `P(X)` moves while `P(y|X)` stays identical. The model's
ROC-AUC falls from 0.8699 to 0.7932, so performance is degraded — but data
drift is also flagged and the causal attribution is not available. Reporting
"concept drift" there would be false. In `concept_shift`, the feature rows are
byte-identical to baseline, the same model's ROC-AUC collapses to 0.3300, and
only then is concept drift asserted.

## Threshold calibration

| Setting | Value | Why |
|---|---|---|
| `psi_threshold` | 0.20 | Conventional "significant shift" PSI band; the covariate regime's 10 numeric features exceed it while the concept regime's identical rows score 0. |
| `ks_pvalue_threshold` | 0.05 | Standard two-sample significance level; with 400 rows per window this detects the constructed mean/scale shifts. |
| `missing_rate_delta_threshold` | 0.05 | Five percentage points is a visible collection defect; the demo dataset has no missing values, so this rule is exercised by unit tests. |
| `aggregate_feature_share_threshold` | 0.30 | Requires more than a single feature before calling a window drifted; two of thirteen features is noise, four is a cohort change. |
| `min_sample_size` | 100 | Below this, PSI and label metrics are too unstable to assert either verdict. |
| `performance_degradation.max_drop` | 0.05 | Larger than the W3 promotion tolerance (0.02) because production windows are noisier; the covariate drop is 0.0767 and the concept drop is 0.5398. |
| `min_labeled_samples` | 100 | Same reasoning as the data sample guard, applied to labels. |

These are demonstration policy values, not universal ML standards. Change them
in `configs/monitoring.yaml`, not in code; unknown keys are rejected.

## Labels and the simulator

`POST /labels` accepts `{"event_id": <int>, "label": 0|1}`:

- `200` on success; the body echoes `event_id`, `label`, `labeled_at`, and
  `updated` (`false` on first write, `true` on relabel).
- `404` with `{"error": "unknown_event"}` for an unknown event.
- `422` for a non-integer/zero `event_id`, a label other than 0/1, or extra
  keys.

The write is an idempotent upsert keyed by `event_id`; the newest label wins.
`POST /predict` now returns `event_id` (null when the best-effort event write
failed, in which case labeling is impossible and the simulator skips it).

Replay traffic deterministically:

```bash
uv run modelops simulate-traffic \
  --dataset artifacts/datasets/<dataset-version> \
  --split validation --count 400 \
  --api-url http://localhost:8000 \
  --with-labels --seed 7
```

Rows are chosen by `numpy.random.Generator(PCG64(seed))` from the selected
split; the same `(dataset, split, count, seed)` sends the same requests.
`--count` larger than the split samples with replacement. The command prints
`sent`, `labeled`, and the model versions observed.

## Reports, status, and retention

`modelops monitor` writes two self-contained files per run:

```text
reports/drift/drift-<UTC timestamp>-v<version>-<id>.json   DriftReport contract
reports/drift/drift-<UTC timestamp>-v<version>-<id>.html   Jinja2, no JS
```

It prints the three verdicts, window sizes, and labeled sample count. Exit
codes: `0` success (including a report where data drift was computed but
performance could not be), `3` when `--fail-on-drift` and data or performance
drifted, `1` only when no report can be produced (invalid config, missing
reference artifact, unreadable windows).

`modelops status --output artifacts/status/status.html` writes the HTML page
and a sibling `status.json`. The page shows the active model and its approval,
recent transitions, recent decisions, the latest drift verdicts with report
paths, and event counts by model version. It has no interactive controls.

`modelops events prune [--days N] [--dry-run]` deletes events older than the
retention window (default `MODELOPS_EVENT_RETENTION_DAYS`, 30 days) and the
labels attached to those events. Labels for surviving events are never
deleted; `--dry-run` prints the counts without changing anything.

## Exact three-scenario demonstration

The active model is a baseline-trained version whose dataset artifact is
`synthetic-v1.0.0-f96aa1f3bdf5`.

```bash
# Prepare the two shifted datasets once.
uv run modelops data prepare --config configs/training.yaml --regime covariate_shift
uv run modelops data prepare --config configs/training.yaml --regime concept_shift

# 1. Clean traffic: all three verdicts stable.
uv run modelops simulate-traffic \
  --dataset artifacts/datasets/synthetic-v1.0.0-f96aa1f3bdf5 \
  --split validation --count 400 --with-labels --seed 7
uv run modelops monitor --config configs/monitoring.yaml --window-size 400

# 2. Covariate shift: data drift and performance loss, concept confounded.
uv run modelops simulate-traffic \
  --dataset artifacts/datasets/<covariate-shift-version> \
  --split validation --count 400 --with-labels --seed 7
uv run modelops monitor --config configs/monitoring.yaml --window-size 400

# 3. Concept shift: stable inputs, performance loss, concept drift asserted.
uv run modelops simulate-traffic \
  --dataset artifacts/datasets/<concept-shift-version> \
  --split validation --count 400 --with-labels --seed 7
uv run modelops monitor --config configs/monitoring.yaml --window-size 400

uv run modelops status --output artifacts/status/status.html
uv run modelops events prune --dry-run
```

Measured on the Compose stack with seed 42 / traffic seed 7:

| Scenario | data_drift | performance_degradation | concept_drift | ROC-AUC | Labeled rows |
|---|---|---|---|---|---|
| baseline | `not_drifted` (0/13) | `not_drifted` (+0.0000) | `not_drifted` | 0.8699 → 0.8699 | 400 |
| covariate_shift | `drifted` (10/13, share 0.769) | `drifted` (−0.0767) | `not_evaluated` (confounded) | 0.8699 → 0.7932 | 400 |
| concept_shift | `not_drifted` (0/13) | `drifted` (−0.5398) | `drifted` | 0.8699 → 0.3300 | 400 |

Each run uses `--window-size 400` so the newest scenario's 400 events exactly
fill the current window; earlier scenarios remain in the table (and the status
page) but are outside the window.

## Tests

```bash
# Unit tier: statistical rules, verdict matrix, reports, labels, simulator,
# pruning, status page, and the concept_shift regime. No Docker.
uv run pytest -q -m "not integration and not e2e"

# Integration: real Postgres + MLflow, three replayed scenarios.
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m integration

# E2E: Compose API, simulated traffic, monitor, and a clean report.
MODELOPS_TEST_API_URL=http://localhost:8000 \
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m e2e
```

## Honest limits

- Performance uses the probability stored at serving time, not a re-score.
  That is the score the caller received and is the correct audit value.
- Event logging is best-effort; predictions whose event write failed are
  absent from the window rather than fabricated.
- The reference and current windows need not share row identities; they share
  the declared schema, feature semantics, and model version.
- Drift is an investigation trigger (ADR-010 / the concept), never an
  automatic model replacement. W6 owns the retraining loop.
