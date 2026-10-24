# Integration tests

This tier exercises real infrastructure: PostgreSQL for prediction events and the
active-model pointer, and an MLflow tracking server for runs and registry
aliases. These tests are marked `integration` and are excluded from the default
`pytest` run because they require services that unit tests must not.

Run them once the corresponding services are available (MLflow from W2,
PostgreSQL from W3):

```bash
uv run pytest -m integration
```

There are intentionally no placeholder tests here. Integration coverage arrives
with the waves that introduce each external dependency.
