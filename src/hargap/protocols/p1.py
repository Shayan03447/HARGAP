"""Protocol P1: Record-wise random splitting (stratified 70/15/15)."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from hargap.protocols.manifest import FoldManifest


def build_p1_folds(
    metadata: pd.DataFrame,
    dataset_name: str,
    cache_sha256: str,
    seeds: List[int] = [1, 2, 3],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15
) -> List[FoldManifest]:
    """
    Builds P1 record-wise stratified folds.
    Windows from all subjects are pooled and split across partitions.
    Overlapping windows from the same subject appear in train, val, and test.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5
    labels = metadata["activity_code"].values
    n_samples = len(metadata)
    all_indices = np.arange(n_samples)

    manifests = []

    for seed in seeds:
        # Step 1: Split into Train (70%) and Temp (30%)
        sss_train = StratifiedShuffleSplit(n_splits=1, test_size=(val_frac + test_frac), random_state=seed)
        train_idx, temp_idx = next(sss_train.split(all_indices, labels))

        # Step 2: Split Temp into Validation (15% of total) and Test (15% of total) -> 50% of temp
        temp_labels = labels[temp_idx]
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 100)
        sub_val_idx, sub_test_idx = next(sss_val.split(temp_idx, temp_labels))

        val_idx = temp_idx[sub_val_idx]
        test_idx = temp_idx[sub_test_idx]

        train_subjects = sorted(list(set(metadata.iloc[train_idx]["subject_id"].values)))
        val_subjects = sorted(list(set(metadata.iloc[val_idx]["subject_id"].values)))
        test_subjects = sorted(list(set(metadata.iloc[test_idx]["subject_id"].values)))

        manifest = FoldManifest(
            dataset=dataset_name,
            protocol="p1",
            fold_id=seed,
            seed=seed,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            train_indices=sorted(train_idx.tolist()),
            val_indices=sorted(val_idx.tolist()),
            test_indices=sorted(test_idx.tolist()),
            window_counts={
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
                "total": n_samples,
            },
            cache_sha256=cache_sha256,
        )
        manifests.append(manifest)

    return manifests
