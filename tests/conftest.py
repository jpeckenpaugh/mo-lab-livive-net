"""Shared fixtures for Docker-free unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelops_lab.config import TrainingConfig, load_training_config
from modelops_lab.data import LoadedDataset, load_prepared_dataset, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"


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
