"""Statistical testing for RQ1, RQ2, and RQ3."""

from hargap.stats.tests import (
    compute_rank_biserial,
    bootstrap_ci_mean_difference,
    paired_wilcoxon_test,
    holm_adjust_pvalues,
)
from hargap.stats.rankings import (
    compute_kendalls_w,
    friedman_ranking_test,
)
from hargap.stats.mitigations import (
    evaluate_mitigation_condition,
    evaluate_rq3_family,
    verify_rq3_affirmative,
)

# Spec-compatible aliases
gap_tests = paired_wilcoxon_test
ranking_tests = friedman_ranking_test
mitigation_tests = evaluate_rq3_family

__all__ = [
    "compute_rank_biserial",
    "bootstrap_ci_mean_difference",
    "paired_wilcoxon_test",
    "holm_adjust_pvalues",
    "compute_kendalls_w",
    "friedman_ranking_test",
    "evaluate_mitigation_condition",
    "evaluate_rq3_family",
    "verify_rq3_affirmative",
    "gap_tests",
    "ranking_tests",
    "mitigation_tests",
]
