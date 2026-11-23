"""Evaluation protocols: P1 record-wise, P2 LOSO, and official benchmark split."""

from hargap.protocols.manifest import FoldManifest
from hargap.protocols.p1 import build_p1_folds
from hargap.protocols.p2 import build_p2_folds
from hargap.protocols.reference import build_official_ucihar_manifest

# Spec-compatible aliases
build_p1 = build_p1_folds
build_p2 = build_p2_folds
official_ucihar_split = build_official_ucihar_manifest
manifest = FoldManifest

__all__ = [
    "FoldManifest",
    "build_p1_folds",
    "build_p2_folds",
    "build_official_ucihar_manifest",
    "build_p1",
    "build_p2",
    "official_ucihar_split",
    "manifest",
]
