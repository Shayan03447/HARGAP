"""Window caching and WindowStore abstraction for label-neutral raw arrays."""

import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class WindowStore:
    """Read-only container for cached window tensors and accompanying metadata."""

    def __init__(self, windows: np.ndarray, metadata: pd.DataFrame, cache_path: Optional[str] = None):
        self.windows = windows.astype(np.float32)
        self.metadata = metadata
        self.cache_path = cache_path
        self._sha256 = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.windows.shape

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]

    @property
    def n_channels(self) -> int:
        return self.windows.shape[1]

    @property
    def window_length(self) -> int:
        return self.windows.shape[2]

    @property
    def sha256(self) -> str:
        if self._sha256 is None:
            # Hash array buffer and metadata
            h = hashlib.sha256()
            h.update(self.windows.tobytes())
            h.update(self.metadata.to_json().encode("utf-8"))
            self._sha256 = h.hexdigest()
        return self._sha256

    def get_subject_mask(self, subject_id: int) -> np.ndarray:
        return (self.metadata["subject_id"] == subject_id).values

    def get_subject_indices(self, subject_id: int) -> np.ndarray:
        return np.where(self.metadata["subject_id"] == subject_id)[0]

    def get_subset(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns windows (M, C, 128), labels (M,), subjects (M,)."""
        sub_windows = self.windows[indices]
        sub_labels = self.metadata.iloc[indices]["activity_code"].values.astype(np.int64)
        sub_subjects = self.metadata.iloc[indices]["subject_id"].values.astype(np.int64)
        return sub_windows, sub_labels, sub_subjects

    def save(self, cache_file: str | Path):
        path = Path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        npz_path = path.with_suffix(".npz")
        meta_path = path.with_suffix(".parquet")
        csv_fallback_path = path.with_suffix(".meta.csv")

        np.savez_compressed(npz_path, windows=self.windows)
        try:
            self.metadata.to_parquet(meta_path, index=False)
        except Exception:
            self.metadata.to_csv(csv_fallback_path, index=False)

    @classmethod
    def load(cls, cache_file: str | Path) -> "WindowStore":
        path = Path(cache_file)
        npz_path = path.with_suffix(".npz")
        meta_path = path.with_suffix(".parquet")
        csv_fallback_path = path.with_suffix(".meta.csv")

        if not npz_path.exists():
            raise FileNotFoundError(f"Cache file {npz_path} does not exist.")

        with np.load(npz_path) as data:
            windows = data["windows"]

        if meta_path.exists():
            metadata = pd.read_parquet(meta_path)
        elif csv_fallback_path.exists():
            metadata = pd.read_csv(csv_fallback_path)
        else:
            raise FileNotFoundError(f"Metadata file not found alongside {npz_path}")

        store = cls(windows, metadata, cache_path=str(path))
        return store


def build_cache(
    windows: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str | Path
) -> WindowStore:
    """Builds, hashes, and persists a WindowStore to disk."""
    store = WindowStore(windows, metadata, cache_path=str(output_path))
    store.save(output_path)
    return store
