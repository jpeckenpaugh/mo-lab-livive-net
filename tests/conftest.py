"""Shared fixtures for Docker-free unit tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelops_lab.config import TrainingConfig, load_training_config
from modelops_lab.data import LoadedDataset, load_prepared_dataset, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"
WATCHED_OUTPUT_DIRS = ("artifacts", "reports")


def _repo_output_files() -> set[str]:
    """Return every file currently under the repository's runtime output dirs."""
    found: set[str] = set()
    for name in WATCHED_OUTPUT_DIRS:
        base = ROOT / name
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file():
                    found.add(str(path.relative_to(ROOT)))
    return found


@pytest.fixture(scope="session", autouse=True)
def isolated_artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Keep every test-generated artifact outside the repository checkout."""
    root = tmp_path_factory.mktemp("modelops-test-artifacts")
    previous = os.environ.get("MODELOPS_ARTIFACT_ROOT")
    os.environ["MODELOPS_ARTIFACT_ROOT"] = str(root)
    try:
        yield root
    finally:
        if previous is None:
            os.environ.pop("MODELOPS_ARTIFACT_ROOT", None)
        else:
            os.environ["MODELOPS_ARTIFACT_ROOT"] = previous


@pytest.fixture(scope="session", autouse=True)
def no_repo_artifact_writes() -> None:
    """Fail the suite if any test writes under the repo's artifacts/ or reports/."""
    before = _repo_output_files()
    yield
    new_files = sorted(_repo_output_files() - before)
    assert not new_files, (
        "tests wrote generated files into the repository output directories: "
        + ", ".join(new_files)
    )


@pytest.fixture(scope="session")
def training_config() -> TrainingConfig:
    return load_training_config(TRAINING_CONFIG)


@pytest.fixture(scope="session")
def prepared_dataset(
    tmp_path_factory: pytest.TempPathFactory, training_config: TrainingConfig
) -> Path:
    root = tmp_path_factory.mktemp("datasets")
    prepared = prepare_dataset(training_config.dataset, training_config.split, out_root=root)
    return prepared.directory


@pytest.fixture(scope="session")
def loaded_dataset(prepared_dataset: Path) -> LoadedDataset:
    return load_prepared_dataset(prepared_dataset)
