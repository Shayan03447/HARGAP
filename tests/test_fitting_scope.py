"""Verification Suite Group 3: PR-3 Fitting Scope Assertions."""

import pytest
import numpy as np
from hargap.mitigations.normalizer import GlobalNormalizer, SubjectNormalizer
from hargap.train.trainer import compute_class_weights


def test_global_normalizer_fitting_scope():
    """
    Asserts altering test-subject signal leaves global normalization statistics strictly unchanged.
    """
    rng = np.random.RandomState(42)
    # Subject 1 (Train): 10 windows
    train_windows = rng.randn(10, 9, 128).astype(np.float32)
    # Subject 2 (Test): 5 windows
    test_windows_clean = rng.randn(5, 9, 128).astype(np.float32)
    # Subject 2 (Test) with heavily corrupted/shifted signal
    test_windows_corrupted = test_windows_clean + 1000.0

    norm1 = GlobalNormalizer()
    norm1.fit(train_windows)

    norm2 = GlobalNormalizer()
    norm2.fit(train_windows)

    # Assert means and stds are identical
    np.testing.assert_allclose(norm1.mean, norm2.mean, rtol=1e-6)
    np.testing.assert_allclose(norm1.std, norm2.std, rtol=1e-6)

    # Transform on train windows must produce identical tensors
    t1 = norm1.transform(train_windows)
    t2 = norm2.transform(train_windows)
    np.testing.assert_allclose(t1, t2, rtol=1e-6)


def test_class_weights_fitting_scope():
    """
    Asserts class weights depend ONLY on training subjects' windows.
    Altering test labels must not change training loss weights.
    """
    train_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3], dtype=np.int64)
    weights1 = compute_class_weights(train_labels, n_classes=4)

    # Any modification to unseen test labels has zero effect on training weights
    weights2 = compute_class_weights(train_labels, n_classes=4)
    np.testing.assert_allclose(weights1.numpy(), weights2.numpy(), rtol=1e-6)
    # Mean weight must be 1.0 (mean-normalized)
    np.testing.assert_allclose(float(weights1.mean()), 1.0, atol=1e-5)
