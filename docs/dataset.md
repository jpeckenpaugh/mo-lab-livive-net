# Dataset: generation, validation, and versioning

ModelOps Lab's primary workload is a seeded synthetic tabular binary
classification dataset (ADR-001). A bundled real loader is available as an
alternate source. This document describes the schema, the generator math, the
three regimes, the canonical serialization and fingerprint scheme, and the
provenance of the optional loader.

## Declared schema

Every dataset has one row per observation, ordered and identified by `row_id`.
The columns are fixed and declared in this order:

`row_id`, ten numeric features, three categorical features, `target`.

| Column | Kind | Declared range / levels | Role |
|---|---|---|---|
| `row_id` | integer id | positive, unique | canonical row identity and split membership |
| `age` | numeric | 18–90 years | demographic |
| `bmi` | numeric | 15.0–55.0 kg/m² | metabolic |
| `systolic_bp` | numeric | 80–220 mmHg | cardiovascular |
| `diastolic_bp` | numeric | 40–140 mmHg | cardiovascular |
| `cholesterol` | numeric | 100–400 mg/dL | metabolic |
| `glucose` | numeric | 50–300 mg/dL | metabolic |
| `resting_heart_rate` | numeric | 40–140 bpm | cardiovascular |
| `max_heart_rate` | numeric | 60–220 bpm | cardiovascular |
| `st_depression` | numeric | 0.0–7.0 mm | ECG |
| `exercise_minutes` | numeric | 0–300 minutes/week | lifestyle |
| `sex` | categorical | `female`, `male` | demographic |
| `chest_pain_type` | categorical | `typical`, `atypical`, `non_anginal`, `asymptomatic` | symptom |
| `smoker` | categorical | `never`, `former`, `current` | lifestyle |
| `target` | integer target | exactly `0` or `1` | binary label, base rate in `[0.20, 0.60]` |

The schema version is `1.0.0` and is recorded as
`DatasetManifest.data_schema_version`, distinct from the contract
`schema_version`. The breast-cancer loader replaces the feature block with the
30 WDBC measurements and keeps the same `row_id`/`target` contract; see
[the optional loader](#optional-loader-breast-cancer-wdbc).

## Synthetic generator

Generation is deterministic for a given `(seed, n_samples, regime)`. One
NumPy `Generator(PCG64(seed))` stream is consumed in a fixed order:

1. **Correlated numeric features.** Ten standard normal draws are transformed
   with the Cholesky factor of a declared correlation matrix, scaled by each
   feature's baseline standard deviation, shifted by its mean, clipped to the
   declared range, and rounded (`age`, blood pressures, cholesterol, glucose,
   heart rates and exercise to integers; `bmi` to 1 decimal; `st_depression` to
   2 decimals).
2. **Categorical features.** Each categorical is drawn independently with the
   regime's declared probability vector.
3. **Label draws.** One uniform draw per row is compared with the row's
   modelled probability.

Modest correlations are explicit, not emergent from independent noise:
`bmi`/`glucose` 0.35, `age`/`systolic_bp` 0.45, `age`/`diastolic_bp` 0.30,
`systolic_bp`/`diastolic_bp` 0.55, `age`/`max_heart_rate` −0.40,
`bmi`/`exercise_minutes` −0.25, `cholesterol`/`glucose` 0.18,
`age`/`cholesterol` 0.18, `systolic_bp`/`bmi` 0.20,
`resting_heart_rate`/`exercise_minutes` −0.15.

### Label mechanism

Each numeric feature is standardized with **fixed baseline constants** — the
same constants for every regime:

```
z_j = (x_j - mean_j) / scale_j
```

The latent score is linear in those standardized numeric features plus one-hot
categorical effects:

```
score = -0.75
      + 0.825·z_age + 0.42·z_bmi + 0.48·z_systolic_bp + 0.12·z_diastolic_bp
      + 0.45·z_cholesterol + 0.63·z_glucose + 0.18·z_resting_heart_rate
      - 0.48·z_max_heart_rate + 0.72·z_st_depression - 0.33·z_exercise_minutes
      + effect(sex) + effect(chest_pain_type) + effect(smoker)

p = sigmoid(score / temperature)
target ~ Bernoulli(p)
```

Categorical effects: `sex=male` +0.32; `chest_pain_type` typical +0.55,
atypical +0.25, non-anginal 0.0, asymptomatic −0.40; `smoker` never 0.0, former
+0.12, current +0.28.

Because the label is **sampled** from `p`, the target contains irreducible
noise. There is no deterministic threshold on a score that would let a model
recover the label exactly. Using fixed baseline standardization constants also
means the same feature row always maps to the same `P(y|X)` in the baseline and
covariate-shift regimes.

## Regimes

A regime changes either `P(X)` or `P(y|X)`, never both. All three emit the same
schema and are recorded in `DatasetManifest.generator_parameters` (including
the numeric distribution, categorical probabilities, coefficients, link
temperature, and standardization constants).

| Regime | What moves | What is held fixed | What it isolates |
|---|---|---|---|
| `baseline` | nothing (reference) | — | reference distribution and label mechanism |
| `covariate_shift` | numeric means/scales and categorical proportions | latent coefficients, standardization, logistic link | **data drift** with unchanged `P(y|X)` |
| `improved_signal` | label coefficients (reweighted and enlarged) and link temperature | feature draws (identical to baseline) | **concept drift** with unchanged `P(X)` |

### `covariate_shift`

The shifted cohort is older and narrower: mild positive mean shifts on age,
BMI, blood pressure, cholesterol, glucose and resting heart rate, smaller
negative shifts on max heart rate and exercise, and scale multipliers from 0.60
to 0.75. Categorical proportions change as well (for example `chest_pain_type`
asymptomatic 0.38 → 0.52, `smoker` current 0.18 → 0.34).

The same latent coefficients and link are used, so `P(y|X)` is unchanged. The
shift moves the base rate and narrows the feature spread, which measurably
degrades a baseline-trained model without collapsing it. Measured with
`seed=42`, `n_samples=2000` (train 1200 / validation 400 / test 400):

- base rate 0.4075 → 0.4995
- all ten numeric features have PSI ≥ 0.204 (max 0.497) against baseline
- baseline-trained model ROC-AUC 0.8699 (baseline slice) → 0.7932 (shifted slice)
- a model retrained on the shifted slice reaches 0.7948, i.e. the ranking signal
  available under the unchanged mechanism is essentially the same

This is covariate shift, not concept drift. It should not be reported as
"the model got worse because the world changed" without the labels that show
the conditional relationship is still stable.

### `improved_signal`

`P(X)` is held fixed: for a given seed the feature rows are identical to
baseline. The label mechanism changes:

- metabolic/exercise features gain weight (`glucose` ×1.6, `exercise_minutes`
  ×1.6, `st_depression` ×1.5, `max_heart_rate` ×1.4), demographic/symptom
  features lose weight (`age` ×0.6, `cholesterol` ×0.6 and similar);
- all numeric coefficients are scaled by 1.3;
- categorical effects are halved;
- the logistic link is sharpened (`temperature=0.72`).

Uniformly enlarging one coefficient vector would only be a monotone transform
of the score and could not change ROC-AUC ranking. The reweighting is therefore
necessary for the demo to be honest: a model trained on `improved_signal`
learns a genuinely different `P(y|X)` and beats a baseline-trained model on the
same rows.

Measured on the same `seed=42` validation slice: a model trained on
`improved_signal` reaches ROC-AUC **0.9418**, while the baseline-trained model
reaches **0.8871** — a **+0.0547** delta.

## Splits

Rows are partitioned deterministically:

1. Take `row_id` values in ascending order.
2. Permute them with `numpy.random.Generator(PCG64(split.seed))`.
3. Assign the first *n* rows to train, the next to validation, then test.
4. Derive exact counts from the configured fractions with largest-remainder
   rounding; ties go to the earlier split in `train, validation, test` order.

With 2000 rows and 0.6/0.2/0.2 the partition is 1200 / 400 / 400. A split that
rounds to zero rows is an error. The resulting `row_id` lists are written to
`splits.json`; later waves consume those lists instead of re-deriving the
partition, and `data validate` recomputes every split fingerprint.

`SplitManifest.class_counts` always carries both `"0"` and `"1"` keys,
including zero counts, so a missing class is visible in the manifest rather
than implied by an absent key.

## Canonical serialization

One serializer (`canonical_csv_bytes`) is used both for writing and hashing, so
the artifact verifies itself:

- fixed declared column order: `row_id`, features, `target`;
- rows sorted ascending by `row_id`;
- UTF-8, LF line endings, exactly one header row, no index column;
- floats formatted with `%.17g` (round-trip exact); integers in decimal;
- CSV quoting only where a value requires it.

`dataset.csv` is the canonical byte stream. `manifest.json` is the
`DatasetManifest` serialized as JSON with sorted keys. `splits.json` is
`{"train": [...], "validation": [...], "test": [...]}` with ascending ids.

`DatasetManifest.created_at` records preparation time and is the one manifest
field that intentionally differs between two fresh runs. It is excluded from
version identity and from reproducibility comparisons.

## Fingerprint and version

```
fingerprint = sha256(canonical dataset.csv bytes)
split fingerprint = sha256(canonical bytes of that subset, header included)
dataset_version = <source_kind>-v<data_schema_version>-<fingerprint[:12]>
```

Example: `synthetic-v1.0.0-f96aa1f3bdf5`.

The version string contains only inputs that are stable for identical
`(source, seed, n_samples, regime, schema version)`; two runs produce the same
version and the same `dataset.csv` bytes.

## Artifact layout

```
<out>/<dataset_version>/
  dataset.csv      canonical bytes
  manifest.json    DatasetManifest, schema_version 1.1.0
  splits.json      persisted row_id lists
```

`modelops data prepare` is idempotent. If the version directory exists, the
fingerprint and all split manifests are re-verified; a match exits 0 with
"already prepared and verified", a mismatch fails with a clear message. Use
`--force` to rewrite the directory.

## Validation

`modelops data prepare` validates before writing; `modelops data validate`
re-validates and additionally proves the bytes are canonical. All violations
are collected and reported together, capped at 20 with a "... and N more"
line. Rules:

- exact column set and declared order, with no silent dtype coercion;
- no null, `NaN`, or infinite value in any column;
- `row_id` positive, unique, and an integer;
- `target` exactly `{0, 1}`, base rate in `[0.20, 0.60]`;
- numeric features finite and inside the ranges in the table above;
- categorical features restricted to declared levels;
- at least 100 rows and at least 50 rows per class.

Each violation names the column, the rule, and a concrete example
(`row_id` and value). The CLI prints one actionable stderr message and exits 1;
it does not print a traceback for user data errors.

## CLI

```bash
modelops data prepare --config configs/training.yaml \
  [--regime baseline|covariate_shift|improved_signal] \
  [--out artifacts/datasets] [--force]

modelops data validate artifacts/datasets/synthetic-v1.0.0-<fingerprint>
```

Exit codes: `0` success, `1` configuration/data/validation error, `2` usage
error. The optional `--regime` flag overrides `dataset.regime` in the config.

## Optional loader: breast cancer (WDBC)

`dataset.kind: breast_cancer` loads scikit-learn's bundled copy of the Breast
Cancer Wisconsin (Diagnostic) dataset:

- 569 rows, 30 numeric features, no categorical features;
- `n_samples` is ignored (the dataset is fixed at 569 rows);
- `row_id` is 1…569 in loader order;
- scikit-learn encodes malignant as `0` and benign as `1`; ModelOps Lab inverts
  the labels so `target=1` means **malignant** (212/569, base rate 0.3726) and
  the declared base-rate gate stays meaningful.

Provenance and usage: UCI Machine Learning Repository, Breast Cancer Wisconsin
(Diagnostic), W.N. Street, W.H. Wolberg and O.L. Mangasarian, "Nuclear feature
extraction for breast tumor diagnosis", IS&T/SPIE 1993. The data ships with
scikit-learn; this repository loads it at runtime and does not redistribute a
copy. Provenance, license note, label transform, and the ignored configured row
count are recorded in `DatasetManifest.generator_parameters`. Drift regimes are
not defined for this loader; requesting one is a clear error.

## Reproducibility notes

- Perturbations to dependency versions, BLAS/LAPACK, or the declared
  correlation matrix could in principle move a value across a rounding
  boundary. The fingerprints are stable for the pinned environment in
  `uv.lock`; treat a fingerprint change after an intentional generator change
  as expected and record it.
- Never compare artifacts by `created_at` or file modification time.
