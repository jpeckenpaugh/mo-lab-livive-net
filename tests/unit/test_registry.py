"""Unit coverage for cross-process registration serialization."""

from __future__ import annotations

import threading
import time

import pytest

from modelops_lab.registry import RegistryError, register_run_once, registration_lock_key


class UnsynchronizedRegistry:
    """A deliberately racy find-or-create registry double.

    ``find_version_for_run`` and ``register_run`` both sleep so an unguarded
    caller would observe ``None`` twice and create duplicate versions.
    """

    def __init__(self) -> None:
        self.versions: dict[str, str] = {}
        self.creates = 0
        self.find_calls = 0
        self._guard = threading.Lock()

    def find_version_for_run(self, run_id: str) -> str | None:
        time.sleep(0.02)
        with self._guard:
            self.find_calls += 1
            return self.versions.get(run_id)

    def register_run(self, run_id: str, model_id: str) -> str:
        time.sleep(0.02)
        with self._guard:
            self.creates += 1
            version = str(len(self.versions) + 1)
            self.versions[run_id] = version
            return version


def test_registration_lock_key_is_stable_and_run_specific() -> None:
    assert registration_lock_key("run-a") == registration_lock_key("run-a")
    assert registration_lock_key("run-a") != registration_lock_key("run-b")


def test_concurrent_registration_creates_exactly_one_version() -> None:
    registry = UnsynchronizedRegistry()
    barrier = threading.Barrier(8)
    results: list[str] = []
    errors: list[BaseException] = []

    def register() -> None:
        try:
            barrier.wait(timeout=30)
            results.append(register_run_once(registry, "run-shared", "m-shared", database_url=None))
        except BaseException as exc:  # noqa: BLE001 - surfaced in the assertion
            errors.append(exc)

    threads = [threading.Thread(target=register) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(results) == 8
    assert set(results) == {"1"}
    assert registry.creates == 1
    assert registry.versions == {"run-shared": "1"}


def test_register_run_once_requires_a_model_id() -> None:
    with pytest.raises(RegistryError, match="no logged model id"):
        register_run_once(UnsynchronizedRegistry(), "run-a", "", database_url=None)
