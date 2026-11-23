"""Design check: reproduction-style training sanity on synthetic data."""

import numpy as np

from hargap.data.loaders import generate_synthetic_benchmark
from hargap.mitigations.normalizer import GlobalNormalizer
from hargap.models.cnn import CNN1D
from hargap.train.trainer import train_fold


def test_cnn_trains_on_synthetic_signal():
    windows, meta = generate_synthetic_benchmark(
        n_subjects=6, n_classes=4, n_channels=6, windows_per_subject_class=20, seed=11
    )
    subs = meta["subject_id"].values
    y = meta["activity_code"].values
    train_mask = np.isin(subs, [1, 2, 3, 4])
    val_mask = np.isin(subs, [5])
    test_mask = np.isin(subs, [6])

    train_x, train_y = windows[train_mask], y[train_mask]
    val_x, val_y = windows[val_mask], y[val_mask]
    test_x, test_y = windows[test_mask], y[test_mask]

    norm = GlobalNormalizer()
    norm.fit(train_x)
    train_x = norm.transform(train_x)
    val_x = norm.transform(val_x)
    test_x = norm.transform(test_x)

    model = CNN1D(n_channels=6, n_classes=4)
    res = train_fold(
        model=model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        n_classes=4,
        epochs_max=12,
        patience=4,
        batch_size=32,
        seeds={"python": 1, "numpy": 11, "torch": 111},
    )
    assert 0.0 <= res["test_macro_f1"] <= 1.0
    assert res["best_epoch"] >= 1
    # Model should learn some structure above uniform chance on this synthetic task
    assert res["test_macro_f1"] >= 0.20
