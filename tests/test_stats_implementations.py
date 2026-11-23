"""Verification Suite: statistical implementation correctness against reference formulas/libraries."""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from hargap.stats.tests import (
    bootstrap_ci_mean_difference,
    compute_rank_biserial,
    holm_adjust_pvalues,
    paired_wilcoxon_test,
)
from hargap.stats.rankings import compute_kendalls_w


def test_rank_biserial_hand_computed():
    # Example: paired differences [2, -1, 3, 4] -> abs ranks [2,1,3,4], signs +,-,+,+ => W+=9, W-=1, r=(9-1)/10=0.8
    diffs = np.array([2.0, -1.0, 3.0, 4.0])
    r = compute_rank_biserial(diffs)
    assert np.isclose(r, 0.8)

    # All positive: r = 1.0
    assert np.isclose(compute_rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)
    # All zero: defined as 0.0
    assert compute_rank_biserial(np.array([0.0, 0.0, 0.0])) == 0.0


def test_holm_matches_statsmodels():
    pvals = [0.01, 0.04, 0.03, 0.20, 0.50]
    reject_ref, p_corr_ref, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    reject, p_corr = holm_adjust_pvalues(pvals, alpha=0.05)
    assert np.allclose(np.array(p_corr), np.array(p_corr_ref))
    assert list(reject) == reject_ref.tolist()


def test_bootstrap_ci_reproducible_with_seed():
    a = np.array([0.70, 0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.74])
    b = np.array([0.55, 0.50, 0.52, 0.60, 0.49, 0.53, 0.57, 0.58])
    mean1, lo1, hi1 = bootstrap_ci_mean_difference(a, b, n_resamples=2000, seed=123)
    mean2, lo2, hi2 = bootstrap_ci_mean_difference(a, b, n_resamples=2000, seed=123)
    assert np.isclose(mean1, np.mean(a - b))
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= mean1 <= hi1


def test_kendalls_w_hand_computed():
    # 3 judges, 3 items with identical rankings 1,2,3 => perfect concordance W=1
    ranks = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)
    assert np.isclose(compute_kendalls_w(ranks), 1.0)

    # Complete disagreement pattern gives small W; compute expected
    # ranks: [[1,2,3],[2,3,1],[3,1,2]] sums [6,6,6] -> S=0 -> W=0
    ranks2 = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=float)
    assert np.isclose(compute_kendalls_w(ranks2), 0.0)


def test_paired_wilcoxon_smoke():
    rng = np.random.RandomState(0)
    a = rng.uniform(0.6, 0.9, size=12)
    b = a - rng.uniform(0.05, 0.2, size=12)
    res = paired_wilcoxon_test(a, b, alternative="greater")
    assert res["n_subjects"] == 12
    assert res["mean_diff"] > 0
    assert 0.0 <= res["p_value"] <= 1.0
    assert -1.0 <= res["rank_biserial"] <= 1.0
    assert len(res["ci_95"]) == 2
