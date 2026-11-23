"""Evaluation metrics and result database management."""

from hargap.eval.metrics import compute_metrics, aggregate_per_subject_scores
from hargap.eval.database import ResultDB

# Spec-compatible alias
score_subjects = aggregate_per_subject_scores

__all__ = [
    "compute_metrics",
    "aggregate_per_subject_scores",
    "score_subjects",
    "ResultDB",
]
