# End-to-end tests

This tier drives the full local stack from Docker Compose, from dataset
preparation through serving, monitoring, promotion, and rollback. Tests are
marked `e2e` and run only where the Compose services are available; the Compose
file arrives with the first infrastructure wave.

```bash
docker compose up -d
uv run pytest -m e2e
```

There are intentionally no placeholder tests here. End-to-end coverage arrives
in W7, when the documented demo sequence exists to drive it.
