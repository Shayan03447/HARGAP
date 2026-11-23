"""Design check: label permutation test on synthetic data (leakage sentinel)."""

import numpy as np

from hargap.data.loaders import generate_synthetic_benchmark
from hargap.verify.checks import permutation_check


def test_permutation_check_at_chance():
    windows, meta = generate_synthetic_benchmark(
        n_subjects=6, n_classes=4, n_channels=6, windows_per_subject_class=12, seed=7
    )
    # Simple split by subject to mimic LOSO-like partitions
    subs = meta["subject_id"].values
    y = meta["activity_code"].values
    train_mask = np.isin(subs, [1, 2, 3, 4])
    val_mask = np.isin(subs, [5])
    test_mask = np.isin(subs, [6])

    res = permutation_check(
        train_x=windows[train_mask],
        train_y=y[train_mask],
        train_subs=subs[train_mask],
        val_x=windows[val_mask],
        val_y=y[val_mask],
        test_x=windows[test_mask],
        test_y=y[test_mask],
        n_classes=4,
        chance_margin=0.05,  # relaxed for tiny synthetic run
        seed=42,
    )
    assert "test_macro_f1" in res
    assert res["chance_level"] == 0.25
    # On properly permuted labels, performance should not be far above chance
    assert res["test_macro_f1"] <= 0.60
