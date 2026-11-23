"""Architecture ranking analysis: Friedman test, Kendall's W, and Holm post-hoc contrasts."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from hargap.stats.tests import paired_wilcoxon_test, holm_adjust_pvalues


def compute_kendalls_w(rankings_matrix: np.ndarray) -> float:
    """
    Computes Kendall's coefficient of concordance W for m subjects (judges/blocks)
    ranking k architectures.
    Input matrix: shape (m, k) where each row represents ranks (1 to k) assigned by subject.
    Formula: W = 12 * S / (m^2 * (k^3 - k))
    """
    m, k = rankings_matrix.shape
    if m <= 1 or k <= 1:
        return 0.0

    rank_sums = np.sum(rankings_matrix, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    w = 12.0 * S / (m ** 2 * (k ** 3 - k))
    return float(np.clip(w, 0.0, 1.0))


def friedman_ranking_test(
    scores_by_arch: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Executes Friedman test with subjects as blocks across architectures.
    scores_by_arch: dict of architecture_name -> (n_subjects,) array
    """
    arch_names = list(scores_by_arch.keys())
    k = len(arch_names)
    matrix = np.column_stack([scores_by_arch[name] for name in arch_names])
    m = matrix.shape[0]

    # Compute ranks per subject (row-wise), rankdata ranks smallest to largest
    # High macro-F1 is best, so rank descending (-matrix)
    ranks = np.zeros_like(matrix)
    for i in range(m):
        ranks[i] = stats.rankdata(-matrix[i])

    mean_ranks = {name: float(np.mean(ranks[:, idx])) for idx, name in enumerate(arch_names)}

    # Friedman test
    res = stats.friedmanchisquare(*[matrix[:, i] for i in range(k)])
    stat, p_val = float(res.statistic), float(res.pvalue)

    kendalls_w = compute_kendalls_w(ranks)

    # Post-hoc pairwise comparisons (two-sided Wilcoxon with Holm correction)
    pairs = []
    pair_pvals = []
    for i in range(k):
        for j in range(i + 1, k):
            name_a, name_b = arch_names[i], arch_names[j]
            test_res = paired_wilcoxon_test(matrix[:, i], matrix[:, j], alternative="two-sided")
            pairs.append({
                "arch_a": name_a,
                "arch_b": name_b,
                "mean_diff": test_res["mean_diff"],
                "p_raw": test_res["p_value"],
                "rank_biserial": test_res["rank_biserial"],
            })
            pair_pvals.append(test_res["p_value"])

    rejects, pvals_holm = holm_adjust_pvalues(pair_pvals)
    for idx, p in enumerate(pairs):
        p["p_holm"] = pvals_holm[idx]
        p["significant"] = rejects[idx]

    return {
        "m_subjects": m,
        "k_architectures": k,
        "friedman_stat": stat,
        "friedman_p": p_val,
        "kendalls_w": kendalls_w,
        "mean_ranks": mean_ranks,
        "pairwise_contrasts": pairs,
    }
