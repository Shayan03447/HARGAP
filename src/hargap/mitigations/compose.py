"""Mitigation pipeline composition (M3: M1 then M2)."""

from typing import Optional
import numpy as np
from hargap.mitigations.normalizer import GlobalNormalizer, SubjectNormalizer
from hargap.mitigations.augmentor import SensorAugmentor


class MitigationPipeline:
    """
    Composed mitigation pipeline supporting baseline, M1, M2, and M3.
    Guarantees validation and test data remain completely unaltered by augmentations.
    """

    def __init__(
        self,
        name: str = "baseline",
        global_normalizer: Optional[GlobalNormalizer] = None,
        subject_normalizer: Optional[SubjectNormalizer] = None,
        augmentor: Optional[SensorAugmentor] = None
    ):
        self.name = name.lower()
        self.global_normalizer = global_normalizer
        self.subject_normalizer = subject_normalizer
        self.augmentor = augmentor

    def process_train(self, windows: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        """Processes training windows with appropriate normalization and on-the-fly augmentation."""
        if self.name == "baseline":
            norm = self.global_normalizer.transform(windows)
            return norm

        elif self.name == "m1":
            norm = self.subject_normalizer.transform(windows, subjects)
            return norm

        elif self.name == "m2":
            norm = self.global_normalizer.transform(windows)
            aug = self.augmentor.transform_batch(norm)
            return aug

        elif self.name == "m3":
            norm = self.subject_normalizer.transform(windows, subjects)
            aug = self.augmentor.transform_batch(norm)
            return aug

        else:
            raise ValueError(f"Unknown mitigation: {self.name}")

    def process_eval(self, windows: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        """
        Processes validation or test windows.
        NO AUGMENTATION is ever applied to evaluation windows.
        """
        if self.name in ["baseline", "m2"]:
            # Baseline and M2 use global training-derived statistics
            return self.global_normalizer.transform(windows)

        elif self.name in ["m1", "m3"]:
            # M1 and M3 use subject-specific statistics
            return self.subject_normalizer.transform(windows, subjects)

        else:
            raise ValueError(f"Unknown mitigation: {self.name}")
