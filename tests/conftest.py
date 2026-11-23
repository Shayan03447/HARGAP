"""Shared pytest fixtures for dataset fixtures, mock WindowStores, and manifests."""

import pytest
import numpy as np
import pandas as pd
from hargap.data.loaders import generate_synthetic_benchmark
from hargap.data.cache import WindowStore
from hargap.protocols.p1 import build_p1_folds
from hargap.protocols.p2 import build_p2_folds
from hargap.protocols.reference import build_official_ucihar_manifest


@pytest.fixture(scope="session")
def synthetic_data():
    """Generates synthetic dataset: 10 subjects, 6 classes, 9 channels, 15 windows/class."""
    windows, meta_df = generate_synthetic_benchmark(
        n_subjects=10,
        n_classes=6,
        n_channels=9,
        windows_per_subject_class=15,
        seed=42
    )
    return windows, meta_df


@pytest.fixture(scope="session")
def synthetic_store(synthetic_data):
    windows, meta_df = synthetic_data
    return WindowStore(windows, meta_df)


@pytest.fixture(scope="session")
def synthetic_p2_manifests(synthetic_store):
    return build_p2_folds(
        synthetic_store.metadata,
        dataset_name="synthetic",
        cache_sha256=synthetic_store.sha256,
        seed=1
    )


@pytest.fixture(scope="session")
def synthetic_p1_manifests(synthetic_store):
    return build_p1_folds(
        synthetic_store.metadata,
        dataset_name="synthetic",
        cache_sha256=synthetic_store.sha256,
        seeds=[1, 2, 3]
    )
