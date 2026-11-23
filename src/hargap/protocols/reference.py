"""Official UCI-HAR split reference protocol for reproduction and benchmark auditing."""

import math
from typing import List
import numpy as np
import pandas as pd
from hargap.protocols.manifest import FoldManifest


def build_official_ucihar_manifest(
    metadata: pd.DataFrame,
    cache_sha256: str,
    seed: int = 1
) -> FoldManifest:
    """
    Constructs the official UCI-HAR partition (21 train / 9 test subjects).
    A validation sub-panel is carved from the 21 training volunteers using
    the standard ascending canonical ID 20% rule (first 4 subjects).
    """
    official_train = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    official_test = [2, 4, 9, 10, 12, 13, 18, 20, 24]

    n_val = max(1, math.floor(0.2 * len(official_train)))  # 4
    val_subjects = sorted(official_train)[:n_val]
    train_subjects = sorted(official_train)[n_val:]

    train_mask = metadata["subject_id"].isin(train_subjects)
    val_mask = metadata["subject_id"].isin(val_subjects)
    test_mask = metadata["subject_id"].isin(official_test)

    train_indices = np.where(train_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    return FoldManifest(
        dataset="ucihar",
        protocol="official",
        fold_id=0,
        seed=seed,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=sorted(official_test),
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
