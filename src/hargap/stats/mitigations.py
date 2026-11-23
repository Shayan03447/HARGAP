"""Mitigation hypothesis evaluation against the pre-specified triple criterion."""

from typing import Dict, Any, List
import numpy as np
from hargap.stats.tests import paired_wilcoxon_test, holm_adjust_pvalues


def evaluate_mitigation_condition(
    baseline_scores: np.ndarray,
    mitigation_scores: np.ndarray,
    threshold_pp: float = 2.0,
    min_rank_biserial: float = 0.3
) -> Dict[str, Any]:
    """
    Evaluates a single (baseline vs mitigation) condition pair.
    Checks:
      1. Mean gain >= threshold_pp (e.g. 2.0 percentage points)
      2. Rank-biserial effect size >= min_rank_biserial (e.g. 0.3)
      3. One-sided Wilcoxon p-value (to be Holm-adjusted across families)
    """
    test_res = paired_wilcoxon_test(mitigation_scores, baseline_scores, alternative="greater")
    gain_pp = test_res["mean_diff_pp"]
    r_rb = test_res["rank_biserial"]
    p_raw = test_res["p_value"]

    criterion_gain = (gain_pp >= threshold_pp)
    criterion_effect = (r_rb >= min_rank_biserial)

    return {
        "n_subjects": len(baseline_scores),
        "mean_gain_pp": gain_pp,
        "ci_95_pp": test_res["ci_95_pp"],
        "rank_biserial": r_rb,
        "p_raw": p_raw,
        "criterion_gain_met": criterion_gain,
        "criterion_effect_met": criterion_effect,
    }


def evaluate_rq3_family(
    comparisons: List[Dict[str, Any]],
    threshold_pp: float = 2.0,
    min_rank_biserial: float = 0.3,
    alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Applies Holm correction across the 12 architecture-dataset comparisons of an M condition,
    then evaluates full triple criterion.
    """
    raw_pvals = [c["eval"]["p_raw"] for c in comparisons]
    rejects, pvals_holm = holm_adjust_pvalues(raw_pvals, alpha=alpha)

    evaluated = []
    for idx, comp in enumerate(comparisons):
        ev = dict(comp["eval"])
        ev["p_holm"] = pvals_holm[idx]
        ev["criterion_p_met"] = (pvals_holm[idx] < alpha)
        
        # Triple criterion check
        effective = (
            ev["criterion_gain_met"] and
            ev["criterion_effect_met"] and
            ev["criterion_p_met"]
        )
        ev["effective"] = effective

        entry = {
            "dataset": comp["dataset"],
            "architecture": comp["architecture"],
            "mitigation": comp["mitigation"],
            **ev
        }
        evaluated.append(entry)

    return evaluated


def verify_rq3_affirmative(
    evaluated_family: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    RQ3 affirmative condition: effective on >= 2 datasets for >= 2 architectures.
    """
    effective_entries = [e for e in evaluated_family if e["effective"]]
    effective_datasets = set(e["dataset"] for e in effective_entries)
    effective_archs = set(e["architecture"] for e in effective_entries)

    passed = (len(effective_datasets) >= 2) and (len(effective_archs) >= 2)

    return {
        "rq3_supported": passed,
        "effective_dataset_count": len(effective_datasets),
        "effective_architecture_count": len(effective_archs),
        "effective_pairs": [(e["dataset"], e["architecture"]) for e in effective_entries],
    }
