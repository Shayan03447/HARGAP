"""Verification Suite Group 2: PR-2 Window Boundary Integrity Assertions."""

import pytest
import numpy as np
from hargap.data.transform import segment_stream, window_segment


def test_window_boundaries_within_segment():
    """
    Asserts no window ever crosses a segment or activity boundary.
    Windows must sit strictly inside continuous same-activity data.
    """
    # 500 samples of activity 1, followed by 500 samples of activity 2
    data = np.random.randn(1000, 3).astype(np.float32)
    labels = np.array([1] * 500 + [2] * 500, dtype=np.int64)

    segments = segment_stream(data, labels, min_length_samples=256)
    assert len(segments) == 2, f"Expected 2 segments, found {len(segments)}"

    for seg_idx, (seg_data, seg_label) in enumerate(segments):
        wins, metas = window_segment(seg_data, seg_label, subject_id=1, segment_id=seg_idx, window_length=128, stride=64)
        assert len(wins) > 0
        for m in metas:
            assert m["activity_code"] == seg_label
            # Start and end samples must be strictly within the segment length
            assert 0 <= m["start_sample"] < len(seg_data)
            assert 0 < m["end_sample"] <= len(seg_data)
            assert (m["end_sample"] - m["start_sample"]) == 128


def test_p2_sample_index_disjointness(synthetic_store, synthetic_p2_manifests):
    """
    Asserts that under P2, the window indices of the test set have zero overlap
    with training or validation window indices.
    """
    for manifest in synthetic_p2_manifests:
        t_indices = set(manifest.train_indices)
        v_indices = set(manifest.val_indices)
        te_indices = set(manifest.test_indices)

        assert len(t_indices.intersection(te_indices)) == 0, f"Fold {manifest.fold_id} has test/train overlap!"
        assert len(v_indices.intersection(te_indices)) == 0, f"Fold {manifest.fold_id} has test/val overlap!"
