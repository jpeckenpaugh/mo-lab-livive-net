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

The tests apply the Alembic migrations before they start and truncate the
promotion tables (plus `prediction_events`) between tests. Serving/rollback
coverage promotes two real versions, serves them through `TestClient`, rolls
back with the CLI, and asserts the event rows in PostgreSQL. There are
intentionally no placeholder tests here.
