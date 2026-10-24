# ModelOps Lab inference service and CLI image.
#
# Builds from the committed lockfile, installs runtime dependencies only, runs
# as a non-root user, and applies schema migrations before starting uvicorn.
FROM python:3.12-slim

# Pinned uv; keep in step with the version used to maintain uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so source edits do not invalidate the layer.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY contracts ./contracts

RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 modelops \
    && chown -R modelops:modelops /app

USER modelops

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# Migrations are idempotent; an existing volume is upgraded to head on start.
CMD ["sh", "-c", "modelops db upgrade && exec modelops serve --host 0.0.0.0 --port 8000"]
