"""Integrity checks: label permutation test, reproduction check, and identity probe."""

from typing import Dict, Any, Tuple
import numpy as np
import torch
from hargap.train.trainer import train_fold
from hargap.models.cnn import CNN1D
from hargap.models.probe import SubjectIdentityProbe


def permutation_check(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_subs: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_classes: int,
    chance_margin: float = 0.02,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Label permutation test: shuffles activity labels WITHIN each subject.
    Since true movement signal is severed, macro-F1 must sit at chance: <= 1/K + 0.02.
    Any performance materially above chance proves data leakage in the pipeline.
    """
    rng = np.random.RandomState(seed)
    shuffled_train_y = train_y.copy()

    # Permute strictly within subjects
    unique_subs = np.unique(train_subs)
    for s in unique_subs:
        mask = (train_subs == s)
        shuffled_train_y[mask] = rng.permutation(shuffled_train_y[mask])

    model = CNN1D(n_channels=train_x.shape[1], n_classes=n_classes)
    res = train_fold(
        model=model,
        train_x=train_x,
        train_y=shuffled_train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        n_classes=n_classes,
        epochs_max=15,  # Short check
        patience=5,
        seeds={"python": seed, "numpy": seed + 1, "torch": seed + 2}
    )

    test_f1 = res["test_macro_f1"]
    chance_level = 1.0 / n_classes
    threshold = chance_level + chance_margin
    passed = (test_f1 <= threshold)

    return {
        "check": "permutation_test",
        "passed": passed,
        "test_macro_f1": test_f1,
        "chance_level": chance_level,
        "threshold": threshold,
        "margin": chance_margin,
    }


def reproduction_check(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_classes: int = 6,
    expected_band: Tuple[float, float] = (0.85, 0.98),
    seed: int = 1
) -> Dict[str, Any]:
    """
    Official UCI-HAR split reproduction check.
    Asserts that 1D-CNN achieves accuracy in the established published band [0.85, 0.98].
    """
    model = CNN1D(n_channels=train_x.shape[1], n_classes=n_classes)
    res = train_fold(
        model=model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        n_classes=n_classes,
        epochs_max=30,
        patience=10,
        seeds={"python": seed, "numpy": seed + 10, "torch": seed + 100}
    )

    acc = res["test_accuracy"]
    f1 = res["test_macro_f1"]
    passed = (expected_band[0] <= acc <= expected_band[1])

    return {
        "check": "official_split_reproduction",
        "passed": passed,
        "accuracy": acc,
        "macro_f1": f1,
        "expected_band": list(expected_band),
    }


def identity_probe_check(
    train_x: np.ndarray,
    train_subs: np.ndarray,
    val_x: np.ndarray,
    val_subs: np.ndarray,
    test_x: np.ndarray,
    test_subs: np.ndarray,
    n_subjects: int,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Subject identity probe check.
    Trains a compact classifier to predict subject identity, quantifying biometric
    separability in inertial windows. Reported, not thresholded.
    """
    # Map all observed subject IDs to contiguous labels so held-out subjects are valid.
    all_sub_arrays = [np.asarray(train_subs), np.asarray(val_subs), np.asarray(test_subs)]
    all_sub_arrays = [a for a in all_sub_arrays if a.size > 0]
    all_subs = np.unique(np.concatenate(all_sub_arrays)) if all_sub_arrays else np.array([], dtype=np.int64)
    sub_map = {int(s): idx for idx, s in enumerate(sorted(all_subs.tolist()))}

    def _map(arr):
        return np.array([sub_map[int(s)] for s in arr], dtype=np.int64)

    mapped_train = _map(train_subs)
    mapped_val = _map(val_subs) if len(val_subs) else mapped_train
    mapped_test = _map(test_subs) if len(test_subs) else mapped_val

    eval_x = test_x if len(test_x) else val_x
    eval_y = mapped_test if len(test_subs) else mapped_val

    n_probe_classes = max(len(sub_map), int(n_subjects), 1)
    model = SubjectIdentityProbe(n_channels=train_x.shape[1], n_subjects=n_probe_classes)
    res = train_fold(
        model=model,
        train_x=train_x,
        train_y=mapped_train,
        val_x=val_x if len(val_x) else train_x,
        val_y=mapped_val if len(val_subs) else mapped_train,
        test_x=eval_x,
        test_y=eval_y,
        n_classes=n_probe_classes,
        epochs_max=20,
        patience=5,
        seeds={"python": seed, "numpy": seed + 1, "torch": seed + 2},
    )

    return {
        "check": "identity_probe",
        "probe_accuracy": float(res.get("test_accuracy", 0.0)),
        "probe_macro_f1": float(res.get("test_macro_f1", 0.0)),
        "n_subjects_trained": int(len(sub_map)),
        "chance_accuracy": 1.0 / float(len(sub_map)) if len(sub_map) else 0.0,
        "best_epoch": int(res.get("best_epoch", 0)),
    }
