"""Verification Suite Group 1: PR-1 Subject Disjointness Assertions."""

import pytest
from hargap.protocols.p2 import build_p2_folds
from hargap.protocols.reference import build_official_ucihar_manifest


def test_p2_subject_pairwise_disjointness(synthetic_p2_manifests):
    """
    Asserts for every P2 fold:
    train_subjects, val_subjects, and test_subjects sets are strictly pairwise disjoint.
    Any intersection immediately fails the integrity check.
    """
    for manifest in synthetic_p2_manifests:
        s_train = set(manifest.train_subjects)
        s_val = set(manifest.val_subjects)
        s_test = set(manifest.test_subjects)

        # Pairwise intersections must be empty
        inter_train_val = s_train.intersection(s_val)
        inter_train_test = s_train.intersection(s_test)
        inter_val_test = s_val.intersection(s_test)

        assert len(inter_train_val) == 0, f"Train and Val intersect in fold {manifest.fold_id}: {inter_train_val}"
        assert len(inter_train_test) == 0, f"Train and Test intersect in fold {manifest.fold_id}: {inter_train_test}"
        assert len(inter_val_test) == 0, f"Val and Test intersect in fold {manifest.fold_id}: {inter_val_test}"

        # Test set must contain exactly 1 subject under LOSO
        assert len(s_test) == 1, f"Fold {manifest.fold_id} test set has {len(s_test)} subjects, expected 1."


def test_official_ucihar_subject_disjointness(synthetic_store):
    """
    Asserts official UCI-HAR split preserves strict subject disjointness (21 train / 9 test).
    """
    manifest = build_official_ucihar_manifest(synthetic_store.metadata, synthetic_store.sha256)
    s_train = set(manifest.train_subjects)
    s_val = set(manifest.val_subjects)
    s_test = set(manifest.test_subjects)

    assert len(s_train.intersection(s_test)) == 0, "Official train and test intersect"
    assert len(s_val.intersection(s_test)) == 0, "Official val and test intersect"
    assert len(s_train.intersection(s_val)) == 0, "Official train and val intersect"
