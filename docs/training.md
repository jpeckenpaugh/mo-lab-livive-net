# Training and experiment tracking

W2 turns a verified dataset artifact into one auditable MLflow run. The model is
deliberately simple; the guarantees are the product: no leakage, stable feature
order, pinned threads and seeds, and a run record complete enough that a reader
can reproduce the numbers without asking the author.

The training code lives in [`src/modelops_lab/train.py`](../src/modelops_lab/train.py)
(pure, MLflow-free functions) and
[`src/modelops_lab/tracking.py`](../src/modelops_lab/tracking.py) (run logging
and read-back).

## Pipeline

`modelops train` loads the dataset directory through
`data.load_prepared_dataset`, which delegates to the existing `verify_dataset`:
the manifest, canonical `dataset.csv` bytes, split coverage, and every split
fingerprint are re-checked before a row reaches the model.

The scikit-learn pipeline is built in declared schema order:

| Step | Transformer | Input |
|---|---|---|
| `preprocess.numeric` | `StandardScaler` | all declared numeric features |
| `preprocess.categorical` | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | all declared categorical features |
| `classifier` | `LogisticRegression(C, max_iter, solver, random_state)` | encoded features |

`row_id` and `target` are never features. The encoder ignores unseen categories
at scoring time; such a row contributes no one-hot signal rather than raising.

### Leakage guarantees

- `fit_and_evaluate` calls `pipeline.fit` once, on the **train** partition only.
  Standardization constants and one-hot categories therefore come only from
  training rows.
- Validation rows are passed only to `predict_proba` after the pipeline is
  fitted. The held-out test partition is not touched by `train` at all.
- A unit test replaces validation features with `1e6` outliers and asserts the
  scaler means still equal the train means exactly.
- `modelops evaluate` evaluates a saved model on the split named on the command
  line; it never refits, resamples, or re-partitions.

## Metrics and threshold

The decision threshold is fixed at **0.5**. It is recorded as a parameter, in
`evaluation.json`, and beside every metrics block; nothing silently picks a
better threshold from validation.

| Metric | Definition |
|---|---|
| `roc_auc` | Ranking quality from positive-class probabilities. A single-class slice has no defined AUC and is recorded as the no-skill value `0.5`; counts make that visible. |
| `precision` | `tp / (tp + fp)`, `0.0` when nothing is predicted positive. |
| `recall` | `tp / (tp + fn)`, `0.0` when the slice has no observed positives. |
| `f1` | Harmonic mean of precision and recall, `0.0` on either zero-division case. |
| `accuracy` | `(tp + tn) / n`. |
| `log_loss` | Mean negative log-likelihood of the observed label under the predicted probability, with `labels=[0, 1]`. |
| `confusion_matrix_tn/fp/fn/tp` | Counts at threshold 0.5. |
| `support` | Rows whose observed label is `1`. |
| `predicted_positive` | Rows whose predicted label is `1`. |
| `positive_rate` | `predicted_positive / sample_count`. |
| `sample_count` | Rows in the evaluated slice. |

Everything is computed by `evaluate_predictions`, which is a pure function of
`(y_true, probabilities, threshold)` and needs no MLflow. Zero-division
behaviour follows scikit-learn's `zero_division=0` convention.

## MLflow run schema

One `modelops train` invocation creates exactly one MLflow run in the configured
experiment. The experiment is created if absent. **No model is registered**;
W3 owns the registry.

Tracking destination precedence:

1. `--tracking-uri`
2. `MLFLOW_TRACKING_URI`
3. `tracking.uri` in `configs/training.yaml`

### Parameters

Every config value is flattened with dotted names, plus the model type and
threshold:

```
dataset.kind, dataset.seed, dataset.n_samples, dataset.regime
split.seed, split.train_fraction, split.validation_fraction, split.test_fraction
model.type, model.random_state
model.hyperparameters.C, model.hyperparameters.max_iter, model.hyperparameters.solver
tracking.uri, tracking.experiment_name
model_type, threshold
```

`tracking.uri` is credential-redacted (`user:***@host`) so a database password
cannot leak into run metadata. It records the configured value, not the
per-invocation override; the effective destination is where the run exists.

### Tags

| Tag | Meaning |
|---|---|
| `git_revision` | `git rev-parse HEAD`; falls back to `GIT_SHA`, then the literal `unknown` (no silent empty tag). |
| `uv_lock_sha256` | SHA-256 of the committed `uv.lock`, or `unavailable`. |
| `python_version`, `sklearn_version`, `mlflow_version`, `numpy_version`, `pandas_version` | Interpreter and library versions that produced the numbers. |
| `dataset_version`, `dataset_fingerprint` | Verified dataset identity. |
| `split_train_fingerprint`, `split_validation_fingerprint`, `split_test_fingerprint` | Split identity, so a comparison cannot silently use a different partition. |
| `split_<name>_row_count` | Row count per split. |
| `regime` | `manifest.generator_parameters.regime` (`not_applicable` for the real loader, `unknown` if a manifest lacks it). |
| `experiment_name`, `evaluation_split` | Experiment and the split the logged metrics describe. |
| `model_id`, `model_uri` | MLflow 3 logged-model identifiers written after logging. |

### Metrics

All metric values above are logged as MLflow metrics with the same names used
in the table (`confusion_matrix_tn`, `confusion_matrix_fp`,
`confusion_matrix_fn`, `confusion_matrix_tp`, `support`, `predicted_positive`,
`positive_rate`, `sample_count`, and the six continuous metrics).

### Artifacts

| Artifact | Content |
|---|---|
| Logged model (`models:/m-...`) | The fitted pipeline, serialized in MLflow 3's safe `skops` format. |
| `MLmodel` signature | Inputs are the declared features in schema order (numeric doubles, categorical strings); output is a `float64` positive-class probability. |
| `input_example.json` | The first five training rows in declared feature order. |
| `evaluation.json` | Deterministic bytes: metrics, counts, threshold, dataset and split fingerprints, feature list, dataset version. No timestamps, so two identical runs produce identical bytes. |

`evaluation.json` is the stable comparison primitive for W3/W5. It deliberately
excludes run IDs, artifact URIs, and timestamps.

## Reproducibility controls

- Dataset and split seeds come from `configs/training.yaml`; split manifests are
  persisted and re-verified at load time.
- Feature order is the declared schema order, not a pandas-inferred order.
- `LogisticRegression.random_state` is fixed by config.
- `fit` and `predict_proba` run inside
  `threadpoolctl.threadpool_limits(limits=1)` so BLAS/OpenMP pool size cannot
  perturb floating-point reductions.
- `uv.lock` is committed and hashed into every run tag.
- Compare runs by metrics, coefficients, and artifact hashes — never by MLflow
  run ID or timestamps.

Two runs on the same dataset and config produce identical metrics, identical
classifier coefficients, and byte-identical `evaluation.json`. The unit suite
asserts this without Docker:

```bash
uv run pytest tests/unit/test_train.py tests/unit/test_tracking.py
```

MLflow 3 refuses a filesystem tracking store unless
`MLFLOW_ALLOW_FILE_STORE=true` is set. The unit tests opt in explicitly; the
documented workflow uses the PostgreSQL-backed Compose server instead.

## Local MLflow stack (Docker Compose)

`compose.yaml` provides PostgreSQL 16 (metadata) and an MLflow 3.17.2 server
with artifact proxying. Artifacts live in the named volume
`modelops-lab-mlflow-artifacts`; metadata lives in the named volume
`modelops-lab_postgres-data`.

```bash
cp .env.example .env          # development-only defaults
docker compose config -q      # validate the file and interpolation
docker compose up -d --wait   # build mlflow, start postgres, wait for health
docker compose ps
curl -fsS http://localhost:5001/health
```

The server exposes the stock MLflow UI at `http://localhost:5001`. Both services
use `restart: unless-stopped`. PostgreSQL is healthchecked with `pg_isready`;
MLflow waits for a healthy database and is healthchecked through
`python -c "urllib.request.urlopen(.../health)"` (the slim image has no curl).

To reset local state, including artifacts and metadata:

```bash
docker compose down -v
```

The host port defaults to **5001** because macOS AirPlay Receiver commonly binds
5000; inside the container MLflow still listens on 5000. Override the published
port with `MLFLOW_PORT` in `.env` and keep the tracking URI in step. Postgres
credentials come from `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`;
the defaults in `.env.example` are demo values, not secrets.

## Training

```bash
# Prepare a dataset first (W1).
uv run modelops data prepare --config configs/training.yaml

# Train against the Compose stack.
uv run modelops train \
  --config configs/training.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --tracking-uri http://localhost:5001 \
  --run-name baseline

# Or rely on MLFLOW_TRACKING_URI / tracking.uri.
export MLFLOW_TRACKING_URI=http://localhost:5001
uv run modelops train --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint>
```

The command prints the run ID, experiment, dataset version and fingerprint,
validation metrics at threshold 0.5 (including the confusion matrix), and the
logged-model URI. Exit codes: `0` success, `1` configuration/dataset/training/
tracking error (one actionable stderr line, no traceback), `2` usage error.

## Evaluating a logged model

`evaluate` is the comparison primitive W3 uses. It loads the model recorded by a
run, applies it to a verified split, prints a JSON summary, and logs nothing.

```bash
uv run modelops evaluate \
  --config configs/training.yaml \
  --dataset artifacts/datasets/synthetic-v1.0.0-<fingerprint> \
  --run-id <mlflow_run_id> \
  --tracking-uri http://localhost:5001
```

- Default split is `validation`.
- `--split test` prints a visible warning that the held-out set must not be used
  for tuning, model selection, or promotion.
- If the dataset fingerprint differs from the run's `dataset_fingerprint` tag,
  a warning names both fingerprints. Evaluation is still allowed because W3
  intentionally compares candidate and active models **on the candidate's
  validation slice** (ADR-011); the printed fingerprints make that slice
  explicit and auditable.
- The JSON includes `run_id`, `model_uri`, `evaluation_split`, `threshold`,
  dataset and split fingerprints, metrics, confusion counts, and the feature
  list. On an unchanged dataset and model, its metrics equal the logged run
  metrics exactly.

## Measured baseline (W2)

Config `configs/training.yaml`, synthetic `baseline`, seed 42, 2000 rows
(train 1200 / validation 400 / test 400), threshold 0.5:

| Metric | Validation |
|---|---|
| `roc_auc` | 0.869869 |
| `precision` | 0.757576 |
| `recall` | 0.744048 |
| `f1` | 0.750751 |
| `accuracy` | 0.792500 |
| `log_loss` | 0.440553 |
| Confusion matrix | tn 192, fp 40, fn 43, tp 125 |
| Support / predictions positive | 168 / 165 |

The quality floors in `configs/promotion.yaml` are the W2-calibrated values
recorded in ADR-010. W3 reads them at gate time and never hardcodes a threshold;
the gate, decision record, and approved/rejected command sequences are
documented in [`promotion.md`](promotion.md).
