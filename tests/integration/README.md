# Integration tests

This tier exercises real infrastructure: PostgreSQL for prediction events and the
active-model pointer, and an MLflow tracking server for runs and registry
aliases. These tests are marked `integration` and are excluded from the default
`pytest` run because they require services that unit tests must not.

Run them once the corresponding services are available:

```bash
MODELOPS_TEST_DATABASE_URL=postgresql+psycopg://modelops:modelops@localhost:5432/modelops \
MODELOPS_TEST_TRACKING_URI=http://localhost:5001 \
  uv run pytest -m integration
```

The promotion tests also apply the Alembic migrations before they start and
truncate the three promotion tables between tests. There are intentionally no
placeholder tests here.
