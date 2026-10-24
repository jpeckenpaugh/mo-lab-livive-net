"""MLflow model registry access behind a small injectable protocol.

MLflow aliases mark approved artifacts (ADR-006); the Postgres pointer records
what is actually active. Registration is idempotent per run id: promoting the
same run twice reuses its model version instead of creating a duplicate.

The :class:`ModelRegistry` protocol keeps the promotion gate unit-testable
without an MLflow registry, which the file-store tracking backend does not
support.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from modelops_lab.db import create_engine_for

# Serializes the find-or-create sequence. On PostgreSQL a transaction-scoped
# advisory lock keyed by (a stable hash of) the run id serializes across
# processes; SQLite and the no-database unit tier use a per-run in-process lock.
_PER_RUN_LOCKS: dict[str, threading.Lock] = {}
_PER_RUN_LOCKS_GUARD = threading.Lock()
_ENGINES: dict[str, Any] = {}
_ENGINES_GUARD = threading.Lock()


class RegistryError(RuntimeError):
    """Raised when the model registry cannot be read or written."""


class ModelRegistry(Protocol):
    """Operations the promotion gate needs from a model registry."""

    def find_version_for_run(self, run_id: str) -> str | None:
        """Return the existing version registered for ``run_id``, if any."""

    def register_run(self, run_id: str, model_id: str) -> str:
        """Register the run's logged model and return its version (idempotent)."""

    def set_alias(self, version: str) -> None:
        """Point the configured alias at ``version``."""

    def delete_alias(self) -> None:
        """Remove the configured alias if it exists."""

    def get_alias_version(self) -> str | None:
        """Return the version the configured alias points at, if any."""

    def load_version(self, version: str) -> Any:
        """Load the model artifact for a registered ``version``."""

    def version_uri(self, version: str) -> str:
        """Return a human-readable artifact location for ``version``."""


@dataclass(frozen=True)
class Reconciliation:
    """Comparison of the pointer's version with the current alias holder."""

    pointer_version: str | None
    alias_version: str | None
    consistent: bool
    detail: str


def registration_lock_key(run_id: str) -> int:
    """Return the signed bigint advisory-lock key for one run id.

    Hashing namespaces the key away from the pointer lock (``ADVISORY_LOCK_KEY``)
    and makes it stable across processes and Python hash seeds.
    """
    digest = hashlib.sha256(f"modelops-registration:{run_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


@contextmanager
def _inprocess_lock(run_id: str) -> Iterator[None]:
    with _PER_RUN_LOCKS_GUARD:
        lock = _PER_RUN_LOCKS.setdefault(run_id, threading.Lock())
    with lock:
        yield


def _engine_for(database_url: str) -> Any:
    with _ENGINES_GUARD:
        engine = _ENGINES.get(database_url)
        if engine is None:
            engine = create_engine_for(database_url)
            _ENGINES[database_url] = engine
    return engine


@contextmanager
def registration_guard(database_url: str | None, run_id: str) -> Iterator[None]:
    """Serialize registration for ``run_id`` across processes where possible.

    PostgreSQL takes ``pg_advisory_xact_lock`` inside a transaction that is held
    until the create-or-reuse decision is complete. SQLite, a missing URL, or a
    non-PostgreSQL URL fall back to a process-wide per-run lock (the unit tier).
    """
    if not database_url or not database_url.startswith("postgresql"):
        with _inprocess_lock(run_id):
            yield
        return

    engine = _engine_for(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": registration_lock_key(run_id)},
            )
            yield
    except SQLAlchemyError as exc:
        raise RegistryError(
            f"cannot take the registration advisory lock for run {run_id!r}: {exc}"
        ) from exc


def register_run_once(
    registry: ModelRegistry,
    run_id: str,
    model_id: str,
    *,
    database_url: str | None = None,
) -> str:
    """Find-or-create one model version for ``run_id`` under a serialization guard.

    This is the standalone form used with an injected registry (and unit tests).
    :meth:`MlflowModelRegistry.register_run` uses the same guard internally.
    """
    if not model_id:
        raise RegistryError(f"run {run_id!r} has no logged model id to register")
    with registration_guard(database_url, run_id):
        existing = registry.find_version_for_run(run_id)
        if existing is not None:
            return existing
        # A real MlflowModelRegistry already owns the guard in ``register_run``;
        # call its unlocked path so the helper cannot take the same lock twice.
        create = getattr(registry, "_register_run_locked", registry.register_run)
        return create(run_id, model_id)


def reconcile(registry: ModelRegistry, pointer_version: str | None) -> Reconciliation:
    """Compare the pointer's version with the current alias holder."""
    alias_version = registry.get_alias_version()
    if pointer_version is None:
        consistent = alias_version is None
        detail = (
            "no active model and no alias (pre-bootstrap)"
            if consistent
            else f"no active model but the alias points at version {alias_version}"
        )
    elif alias_version is None:
        consistent = False
        detail = (
            f"active pointer is version {pointer_version} but the alias is unset; "
            "reconcile by setting the alias to the active version"
        )
    elif alias_version != pointer_version:
        consistent = False
        detail = (
            f"active pointer is version {pointer_version} but the alias points at "
            f"version {alias_version}; reconcile before promoting"
        )
    else:
        consistent = True
        detail = f"active pointer and alias both reference version {pointer_version}"
    return Reconciliation(
        pointer_version=pointer_version,
        alias_version=alias_version,
        consistent=consistent,
        detail=detail,
    )


class MlflowModelRegistry:
    """MLflow 3 registry implementation bound to one model name and alias."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        registered_model_name: str,
        alias: str,
        lock_database_url: str | None = None,
    ) -> None:
        if not registered_model_name.strip():
            raise RegistryError("registered_model_name must not be empty")
        if not alias.strip():
            raise RegistryError("registry alias must not be empty")
        self.tracking_uri = tracking_uri
        self.registered_model_name = registered_model_name
        self.alias = alias
        self.lock_database_url = lock_database_url
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def _ensure_registered_model(self) -> None:
        try:
            self._client.get_registered_model(self.registered_model_name)
            return
        except MlflowException as exc:
            if not _is_missing(exc):
                raise RegistryError(
                    f"cannot read registered model {self.registered_model_name!r}: {exc}"
                ) from exc
        try:
            self._client.create_registered_model(self.registered_model_name)
        except MlflowException as exc:
            if not _is_already_exists(exc):
                raise RegistryError(
                    f"cannot create registered model {self.registered_model_name!r}: {exc}"
                ) from exc

    def _versions_for_run(self, run_id: str) -> list[Any]:
        escaped = self.registered_model_name.replace("'", "''")
        try:
            versions = self._client.search_model_versions(f"name = '{escaped}'")
        except MlflowException as exc:
            if _is_missing(exc):
                return []
            raise RegistryError(
                f"cannot search versions of {self.registered_model_name!r}: {exc}"
            ) from exc
        return [version for version in versions if version.run_id == run_id]

    def find_version_for_run(self, run_id: str) -> str | None:
        versions = self._versions_for_run(run_id)
        if not versions:
            return None
        latest = max(versions, key=lambda version: int(version.version))
        return str(latest.version)

    def register_run(self, run_id: str, model_id: str) -> str:
        """Create one version for the run, or reuse the version already created.

        The find-or-create sequence is serialized across processes by the
        PostgreSQL advisory lock described in :func:`registration_guard`, so two
        concurrent retrains of the same run return the same version.
        """
        if not model_id:
            raise RegistryError(f"run {run_id!r} has no logged model id to register")
        with registration_guard(self.lock_database_url, run_id):
            return self._register_run_locked(run_id, model_id)

    def _register_run_locked(self, run_id: str, model_id: str) -> str:
        existing = self.find_version_for_run(run_id)
        if existing is not None:
            return existing

        self._ensure_registered_model()
        source = f"models:/{model_id}"
        try:
            created = self._client.create_model_version(
                name=self.registered_model_name,
                source=source,
                run_id=run_id,
                model_id=model_id,
            )
        except MlflowException as exc:
            # A concurrent registration of the same run may have won the race.
            existing = self.find_version_for_run(run_id)
            if existing is not None:
                return existing
            raise RegistryError(
                f"cannot register run {run_id!r} as {self.registered_model_name!r}: {exc}"
            ) from exc

        # Record the run id as a tag so the link survives even without run_id
        # support in an older registry implementation.
        with contextlib.suppress(MlflowException):
            self._client.set_model_version_tag(
                self.registered_model_name, created.version, "mlflow_run_id", run_id
            )
        return str(created.version)

    def set_alias(self, version: str) -> None:
        try:
            self._client.set_registered_model_alias(self.registered_model_name, self.alias, version)
        except MlflowException as exc:
            raise RegistryError(
                f"cannot set alias {self.alias!r} to version {version} on "
                f"{self.registered_model_name!r}: {exc}"
            ) from exc

    def delete_alias(self) -> None:
        try:
            self._client.delete_registered_model_alias(self.registered_model_name, self.alias)
        except MlflowException as exc:
            if _is_missing(exc):
                return
            raise RegistryError(
                f"cannot delete alias {self.alias!r} on {self.registered_model_name!r}: {exc}"
            ) from exc

    def get_alias_version(self) -> str | None:
        try:
            version = self._client.get_model_version_by_alias(
                self.registered_model_name, self.alias
            )
        except MlflowException as exc:
            if _is_missing(exc):
                return None
            raise RegistryError(
                f"cannot resolve alias {self.alias!r} on {self.registered_model_name!r}: {exc}"
            ) from exc
        return str(version.version)

    def load_version(self, version: str) -> Any:
        model_uri = f"models:/{self.registered_model_name}/{version}"
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            return mlflow.sklearn.load_model(model_uri)
        except (MlflowException, OSError, ValueError) as exc:
            raise RegistryError(f"cannot load {model_uri}: {exc}") from exc

    def version_uri(self, version: str) -> str:
        try:
            model_version = self._client.get_model_version(self.registered_model_name, version)
        except MlflowException as exc:
            raise RegistryError(
                f"cannot read version {version} of {self.registered_model_name!r}: {exc}"
            ) from exc
        return str(model_version.source)


def _error_code(exc: MlflowException) -> str:
    return getattr(exc, "error_code", "") or ""


def _is_missing(exc: MlflowException) -> bool:
    return _error_code(exc) in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}


def _is_already_exists(exc: MlflowException) -> bool:
    return _error_code(exc) in {"RESOURCE_ALREADY_EXISTS", "ALREADY_EXISTS"}
