"""Verification Suite Group 5: Fold Manifest Reproducibility and Hashing."""

import pytest
import json
import tempfile
from pathlib import Path
from hargap.protocols.manifest import FoldManifest
from hargap.protocols.p2 import build_p2_folds


def test_manifest_determinism_and_hashing(synthetic_store):
    """
    Asserts regenerating fold manifests reproduces exact JSON bytes and identical sha256 hash.
    """
    manifests_run1 = build_p2_folds(synthetic_store.metadata, "synthetic", synthetic_store.sha256, seed=1)
    manifests_run2 = build_p2_folds(synthetic_store.metadata, "synthetic", synthetic_store.sha256, seed=1)

    assert len(manifests_run1) == len(manifests_run2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for m1, m2 in zip(manifests_run1, manifests_run2):
            hash1 = m1.compute_sha256()
            hash2 = m2.compute_sha256()
            assert hash1 == hash2, f"Manifest hash mismatch: {hash1} vs {hash2}"

            # Save and reload
            file1 = tmp_path / f"fold_{m1.fold_id}"
            m1.save(file1)
            reloaded = FoldManifest.load(file1)
            assert reloaded.compute_sha256() == hash1
