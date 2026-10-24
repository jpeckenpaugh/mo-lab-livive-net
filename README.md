# ModelOps Lab

ModelOps Lab is a lightweight, self-hosted reference implementation of a
complete machine learning operations lifecycle. The model is deliberately
simple; the product is the operational path from a versioned dataset through
training, promotion, serving, monitoring, drift detection, retraining, and
rollback.

**Current status: W0 scaffold.** The repository currently contains the project
packaging, validated configuration models, the frozen cross-wave contracts and
their JSON Schemas, a CLI command surface, and CI. The lifecycle itself is not
implemented yet: every `modelops` lifecycle command exits with status 2 and
names the wave that will implement it. Do not expect data, training, serving,
monitoring, or registry behavior in this revision.

## Development setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is managed by uv.

```bash
uv sync
uv run modelops --help
uv run pytest
```

Lint and formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

Regenerate the committed contract schemas after an intentional contract change:

```bash
uv run python scripts/export_schemas.py
```

Runtime settings are read from `MODELOPS_*` environment variables or a local
`.env` file. Copy `.env.example` to `.env` for development; the values there are
not secrets.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | Validated YAML for training, promotion, and monitoring |
| `src/modelops_lab/contracts/` | Frozen Pydantic contracts and schema exporter |
| `contracts/` | Generated JSON Schemas, committed for review |
| `src/modelops_lab/cli.py` | Typer command surface (stubs until their waves) |
| `tests/unit/` | Docker-free tests of config, contracts, schemas, and CLI |
| `tests/integration/` | Real Postgres/MLflow tests (later waves) |
| `tests/e2e/` | Docker Compose tests (later waves) |

## Documentation

- Product specification: [`concept.md`](concept.md)
- Execution plan, waves, and frozen contract definitions: [`docs/plan.md`](docs/plan.md)
- Architecture decisions: [`docs/decisions.md`](docs/decisions.md)
