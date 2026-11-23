"""Data normalizers: baseline global z-score and M1 per-subject z-score."""

from typing import Dict, Tuple, Optional
import numpy as np


class GlobalNormalizer:
    """
    Baseline global z-score normalization.
    Mean and standard deviation are fitted per channel strictly across training subjects' windows.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None  # (C, 1)
        self.std: Optional[np.ndarray] = None   # (C, 1)

    def fit(self, train_windows: np.ndarray):
        # train_windows: (N_train, C, 128)
        # Compute mean and std per channel over all N and timesteps
        self.mean = np.mean(train_windows, axis=(0, 2), keepdims=True)  # (1, C, 1)
        self.std = np.std(train_windows, axis=(0, 2), keepdims=True)    # (1, C, 1)
        self.std = np.where(self.std < self.eps, 1.0, self.std)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("GlobalNormalizer must be fitted before transform.")
        return (windows - self.mean) / self.std


class SubjectNormalizer:
    """
    M1: Per-subject z-score normalization.
    Calculates mean and standard deviation per channel for each subject from unlabelled signal.
    For training subjects: computed from their training windows.
    For test subjects: computed from the unlabelled test stream (or first 60s calibration).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        # Maps subject_id -> (mean: np.ndarray (1, C, 1), std: np.ndarray (1, C, 1))
        self.stats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def fit_subject(self, subject_id: int, windows: np.ndarray):
        # windows: (N_sub, C, 128)
        if len(windows) == 0:
            return
        mean = np.mean(windows, axis=(0, 2), keepdims=True)  # (1, C, 1)
        std = np.std(windows, axis=(0, 2), keepdims=True)    # (1, C, 1)
        std = np.where(std < self.eps, 1.0, std)
        self.stats[int(subject_id)] = (mean, std)

    def fit_subjects_batch(self, windows: np.ndarray, subjects: np.ndarray):
        unique_subs = np.unique(subjects)
        for s in unique_subs:
            mask = (subjects == s)
            self.fit_subject(int(s), windows[mask])

    def transform(self, windows: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(windows, dtype=np.float32)
        unique_subs = np.unique(subjects)

        for s in unique_subs:
            s_int = int(s)
            if s_int not in self.stats:
                raise KeyError(f"Subject {s_int} was not fitted in SubjectNormalizer.")
            mean, std = self.stats[s_int]
            mask = (subjects == s)
            normalized[mask] = (windows[mask] - mean) / std

        return normalized
