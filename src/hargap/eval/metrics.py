"""Evaluation metrics: per-subject macro-F1, balanced accuracy, recall, and confusion matrix."""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Computes full suite of classification metrics for an evaluation partition."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    
    # Per-class recall
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recalls = {int(i): float(recalls[i]) for i in range(len(recalls))}
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "per_class_recall": per_class_recalls,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def aggregate_per_subject_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray
) -> Dict[int, float]:
    """
    Computes macro-F1 partitioned by subject ID.
    Used under P1 to measure per-subject performance on test windows,
    enabling paired statistical testing with P2 LOSO scores.
    """
    unique_subs = np.unique(subjects)
    scores = {}
    for s in unique_subs:
        mask = (subjects == s)
        s_true = y_true[mask]
        s_pred = y_pred[mask]
        if len(s_true) > 0:
            score = float(f1_score(s_true, s_pred, average="macro", zero_division=0))
            scores[int(s)] = score
    return scores
