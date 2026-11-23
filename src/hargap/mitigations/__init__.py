"""Mitigation techniques: M1 per-subject normalization, M2 sensor augmentation, M3 composition."""

from hargap.mitigations.normalizer import GlobalNormalizer, SubjectNormalizer
from hargap.mitigations.augmentor import SensorAugmentor, build_3d_rotation_matrix
from hargap.mitigations.compose import MitigationPipeline

__all__ = [
    "GlobalNormalizer",
    "SubjectNormalizer",
    "SensorAugmentor",
    "build_3d_rotation_matrix",
    "MitigationPipeline",
]
