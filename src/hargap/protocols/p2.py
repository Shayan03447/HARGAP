"""Protocol P2: Leave-One-Subject-Out (LOSO) with canonical 20% validation panel."""

import math
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from hargap.protocols.manifest import FoldManifest


def build_p2_folds(
    metadata: pd.DataFrame,
    dataset_name: str,
    cache_sha256: str,
    seed: int = 1
) -> List[FoldManifest]:
    """
    Constructs P2 Leave-One-Subject-Out (LOSO) cross-validation folds.
    In each fold:
      - Exactly one subject is held out as test.
      - From the remaining subjects, a deterministic validation panel is selected:
        first max(1, floor(0.2 * (n_subjects - 1))) ascending canonical subject IDs.
      - Remaining subjects form the training set.
      - Training, validation, and test subject sets are strictly disjoint.
    """
    unique_subjects = sorted(list(set(metadata["subject_id"].values)))
    n_subjects = len(unique_subjects)
    n_val = max(1, math.floor(0.2 * (n_subjects - 1)))

    manifests = []

    for fold_idx, test_subject in enumerate(unique_subjects):
        remaining = [s for s in unique_subjects if s != test_subject]
        val_subjects = remaining[:n_val]
        train_subjects = remaining[n_val:]

        train_mask = metadata["subject_id"].isin(train_subjects)
        val_mask = metadata["subject_id"].isin(val_subjects)
        test_mask = metadata["subject_id"] == test_subject

        train_indices = np.where(train_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        test_indices = np.where(test_mask)[0].tolist()

        manifest = FoldManifest(
            dataset=dataset_name,
            protocol="p2",
            fold_id=fold_idx + 1,  # 1-indexed fold ID
            seed=seed,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=[test_subject],
            train_indices=sorted(train_indices),
            val_indices=sorted(val_indices),
            test_indices=sorted(test_indices),
            window_counts={
                "train": len(train_indices),
                "val": len(val_indices),
                "test": len(test_indices),
                "total": len(metadata),
            },
            cache_sha256=cache_sha256,
        )
        manifests.append(manifest)

    return manifests
