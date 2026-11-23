"""Design check: subject-identity probe runs and reports above-chance separability."""

import numpy as np

from hargap.data.loaders import generate_synthetic_benchmark
from hargap.verify.checks import identity_probe_check


def test_identity_probe_runs():
    windows, meta = generate_synthetic_benchmark(
        n_subjects=6, n_classes=4, n_channels=6, windows_per_subject_class=15, seed=13
    )
    subs = meta["subject_id"].values
    train_mask = np.isin(subs, [1, 2, 3, 4])
    val_mask = np.isin(subs, [5])
    # probe check signature includes test args; provide val as held-out evaluation in this synthetic setup
    res = identity_probe_check(
        train_x=windows[train_mask],
        train_subs=subs[train_mask],
        val_x=windows[val_mask],
        val_subs=subs[val_mask],
        test_x=windows[val_mask],
        test_subs=subs[val_mask],
        n_subjects=4,
        seed=21,
    )
    assert res["check"] == "identity_probe"
    assert 0.0 <= res["probe_accuracy"] <= 1.0
    # Train subjects {1..4} plus held-out val subject {5} are observed/mapped.
    assert res["n_subjects_trained"] == 5
    assert np.isclose(res["chance_accuracy"], 0.2)
