"""Statistical inference primitives: Wilcoxon signed-rank, rank-biserial, bootstrap CI, Holm."""

from typing import Tuple, Dict, Any, List
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def compute_rank_biserial(diffs: np.ndarray) -> float:
    """
    Computes matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-).
    Equivalent to r = 1 - 2*W / (n*(n+1)/2) for non-zero differences.
    """
    diffs = np.asarray(diffs)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(diffs))
    w_plus = np.sum(ranks[diffs > 0])
    w_minus = np.sum(ranks[diffs < 0])
    total = w_plus + w_minus
    if total == 0:
        return 0.0
    return float((w_plus - w_minus) / total)


def bootstrap_ci_mean_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes bootstrap percentile confidence interval on the mean difference (A - B).
    Returns (mean_diff, ci_lower, ci_upper).
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    if n <= 1:
        return mean_diff, mean_diff, mean_diff

    rng = np.random.RandomState(seed)
    boot_indices = rng.choice(n, size=(n_resamples, n), replace=True)
    boot_means = np.mean(diffs[boot_indices], axis=1)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return mean_diff, ci_lower, ci_upper


def paired_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "greater"
) -> Dict[str, Any]:
    """
    Executes one-sided Wilcoxon signed-rank test on paired per-subject scores.
    Null hypothesis: distribution of (A - B) is symmetric about 0.
    Alternative: A > B (or 'two-sided', 'less').
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    non_zero = diffs[diffs != 0]
    n = len(diffs)

    if len(non_zero) < 5:
        # Too few non-zero differences for asymptotic/exact Wilcoxon
        w_stat, p_val = 0.0, 1.0
    else:
        res = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
        w_stat, p_val = float(res.statistic), float(res.pvalue)

    r_rb = compute_rank_biserial(diffs)
    mean_diff, ci_low, ci_high = bootstrap_ci_mean_difference(scores_a, scores_b)

    return {
        "n_subjects": n,
        "mean_diff": mean_diff,
        "mean_diff_pp": mean_diff * 100.0,
        "ci_95": [ci_low, ci_high],
        "ci_95_pp": [ci_low * 100.0, ci_high * 100.0],
        "statistic": w_stat,
        "p_value": p_val,
        "rank_biserial": r_rb,
    }


def holm_adjust_pvalues(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Applies the Holm sequentially rejective multiple test procedure.
    Returns (reject_list, adjusted_pvalues).
    """
    if len(p_values) == 0:
        return [], []
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return reject.tolist(), pvals_corrected.tolist()
