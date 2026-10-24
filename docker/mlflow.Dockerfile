# MLflow tracking server for ModelOps Lab.
#
# The SQL backend is PostgreSQL (ADR-003). Artifacts live on a mounted volume
# and are served through the tracking server's artifact proxy, so clients only
# need the HTTP tracking URI. Dependencies are pinned to the same versions as
# uv.lock; the compose stack is the W3+ integration target, not a floating
# deployment.
FROM python:3.12-slim

ARG MLFLOW_VERSION=3.17.2
ARG PSYCOPG_VERSION=3.3.6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg[binary]==${PSYCOPG_VERSION}"

RUN useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlflow:mlflow /mlflow

USER mlflow
WORKDIR /home/mlflow
EXPOSE 5000

# No curl in the image: the healthcheck uses the standard library only.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=12 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)"]

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
