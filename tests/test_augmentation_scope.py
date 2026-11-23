"""Verification Suite Group 4: PR-4 Augmentation Scope Assertions."""

import pytest
import numpy as np
from hargap.mitigations.normalizer import GlobalNormalizer, SubjectNormalizer
from hargap.mitigations.augmentor import SensorAugmentor
from hargap.mitigations.compose import MitigationPipeline


def test_eval_loader_unaltered_under_m2():
    """
    Asserts validation and test windows pass through evaluation pipeline UNALTERED by M2.
    The evaluation output under M2 must equal baseline tensor-for-tensor.
    """
    rng = np.random.RandomState(42)
    val_windows = rng.randn(20, 9, 128).astype(np.float32)
    val_subjects = np.array([2] * 20)

    # Global normalizer
    norm = GlobalNormalizer()
    norm.fit(val_windows)

    aug = SensorAugmentor(p_each=1.0)  # Always augment if called!

    pipe_baseline = MitigationPipeline(name="baseline", global_normalizer=norm)
    pipe_m2 = MitigationPipeline(name="m2", global_normalizer=norm, augmentor=aug)

    out_baseline = pipe_baseline.process_eval(val_windows, val_subjects)
    out_m2 = pipe_m2.process_eval(val_windows, val_subjects)

    # Tensor-for-tensor equality
    np.testing.assert_array_equal(
        out_m2, out_baseline,
        err_msg="M2 modified validation/test windows! Evaluator must remain identity."
    )


def test_eval_loader_unaltered_under_m3():
    """
    Asserts validation and test windows under M3 receive M1 normalization only,
    with zero augmentation transforms applied.
    """
    rng = np.random.RandomState(42)
    val_windows = rng.randn(20, 9, 128).astype(np.float32)
    val_subjects = np.array([2] * 20)

    sub_norm = SubjectNormalizer()
    sub_norm.fit_subject(2, val_windows)

    aug = SensorAugmentor(p_each=1.0)

    pipe_m1 = MitigationPipeline(name="m1", subject_normalizer=sub_norm)
    pipe_m3 = MitigationPipeline(name="m3", subject_normalizer=sub_norm, augmentor=aug)

    out_m1 = pipe_m1.process_eval(val_windows, val_subjects)
    out_m3 = pipe_m3.process_eval(val_windows, val_subjects)

    np.testing.assert_array_equal(
        out_m3, out_m1,
        err_msg="M3 applied augmentation to evaluation windows! Evaluator must remain clean."
    )
