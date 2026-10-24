"""Integration-tier guards: never touch the demo database or repo artifacts."""

from __future__ import annotations

import os

import pytest

from modelops_lab.db import validate_test_database_url


@pytest.fixture(scope="session", autouse=True)
def isolated_test_database() -> None:
    """Validate the test URL and make it the default for CLI invocations.

    The guard fails hard when ``MODELOPS_TEST_DATABASE_URL`` names anything other
    than a ``*_test`` database, because the fixtures truncate lifecycle tables.
    """
    url = os.environ.get("MODELOPS_TEST_DATABASE_URL")
    if url is None:
        return
    validated = validate_test_database_url(url)
    previous = os.environ.get("MODELOPS_DATABASE_URL")
    os.environ["MODELOPS_DATABASE_URL"] = validated
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MODELOPS_DATABASE_URL", None)
        else:
            os.environ["MODELOPS_DATABASE_URL"] = previous
