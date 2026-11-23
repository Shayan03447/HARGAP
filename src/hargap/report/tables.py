"""Automated dissertation table generation for Chapters 5 and 6."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from hargap.stats.tests import paired_wilcoxon_test, holm_adjust_pvalues
from hargap.stats.rankings import friedman_ranking_test
from hargap.stats.mitigations import evaluate_mitigation_condition, evaluate_rq3_family


def generate_table_rq1_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Table for RQ1: P1 record-wise vs P2 LOSO per-subject generalisation gap.
    Evaluates 12 architecture-dataset pairs with one-sided Wilcoxon, rank-biserial,
    10,000-resample bootstrap 95% CI, and Holm correction across the 12 pairs.
    """
    rows = []
    p_raw_list = []

    datasets = sorted(df["dataset"].unique())
    architectures = sorted(df["architecture"].unique())

    for d in datasets:
        for a in architectures:
            # Baseline unmitigated runs
            sub_df = df[(df["dataset"] == d) & (df["architecture"] == a) & (df["mitigation"] == "baseline")]
            p1_runs = sub_df[sub_df["protocol"] == "p1"]
            p2_runs = sub_df[sub_df["protocol"] == "p2"]

            if len(p1_runs) == 0 or len(p2_runs) == 0:
                continue

            # Identify common subject columns or per-subject scores
            sub_cols = [c for c in sub_df.columns if c.startswith("sub_") and c.endswith("_f1")]
            if sub_cols:
                # Average over seeds for P1
                p1_scores = p1_runs[sub_cols].mean(axis=0).values
                # P2 single fold per subject
                p2_scores = p2_runs[sub_cols].max(axis=0).values  # each fold has 1 subject
                
                # Filter valid subjects
                valid = ~np.isnan(p1_scores) & ~np.isnan(p2_scores)
                s_p1 = p1_scores[valid]
                s_p2 = p2_scores[valid]
            else:
                s_p1 = p1_runs["macro_f1"].values
                s_p2 = p2_runs["macro_f1"].values
                min_len = min(len(s_p1), len(s_p2))
                s_p1, s_p2 = s_p1[:min_len], s_p2[:min_len]

            if len(s_p1) == 0 or len(s_p2) == 0:
                continue

            res = paired_wilcoxon_test(s_p1, s_p2, alternative="greater")
            p_raw_list.append(res["p_value"])

            rows.append({
                "Dataset": d.upper(),
                "Architecture": a.upper(),
                "n": res["n_subjects"],
                "Mean P1 F1": float(np.mean(s_p1)),
                "Mean P2 F1": float(np.mean(s_p2)),
                "Gap (pp)": res["mean_diff_pp"],
                "95% CI (pp)": f"[{res['ci_95_pp'][0]:.2f}, {res['ci_95_pp'][1]:.2f}]",
                "r_rb": res["rank_biserial"],
                "p_raw": res["p_value"],
            })

    # Apply Holm correction across the 12 pairs
    rejects, p_holm = holm_adjust_pvalues(p_raw_list, alpha=0.05)
    for idx, r in enumerate(rows):
        r["p_Holm"] = p_holm[idx]
        r["Sig (alpha=0.05)"] = "Yes" if rejects[idx] else "No"

    table_df = pd.DataFrame(rows)
    return table_df


def generate_table_rq2_rankings(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Constructs Table for RQ2: Architecture rankings under P2 LOSO per dataset.
    Computes Friedman test, Kendall's W, and post-hoc pairwise contrasts.
    """
    tables_by_dataset = {}
    datasets = sorted(df["dataset"].unique())
    architectures = sorted(df["architecture"].unique())

    for d in datasets:
        sub_df = df[(df["dataset"] == d) & (df["protocol"] == "p2") & (df["mitigation"] == "baseline")]
        sub_cols = [c for c in sub_df.columns if c.startswith("sub_") and c.endswith("_f1")]

        scores_by_arch = {}
        for a in architectures:
            a_df = sub_df[sub_df["architecture"] == a]
            if sub_cols and len(a_df) > 0:
                vals = a_df[sub_cols].max(axis=0).dropna().values
                if len(vals) > 0:
                    scores_by_arch[a] = vals
            elif len(a_df) > 0:
                scores_by_arch[a] = a_df["macro_f1"].values

        if len(scores_by_arch) == len(architectures):
            min_n = min(len(v) for v in scores_by_arch.values())
            trimmed = {k: v[:min_n] for k, v in scores_by_arch.items()}
            f_res = friedman_ranking_test(trimmed)
            tables_by_dataset[d] = f_res

    return tables_by_dataset


def generate_table_rq3_mitigations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Table for RQ3: Mitigation effects (M1, M2, M3 vs baseline) under P2.
    Applies pre-specified triple criterion (>= 2 pp gain, Holm p < 0.05, r >= 0.3).
    """
    datasets = sorted(df["dataset"].unique())
    architectures = sorted(df["architecture"].unique())
    mitigations = ["m1", "m2", "m3"]

    all_comparisons = []

    for m in mitigations:
        m_comparisons = []
        for d in datasets:
            for a in architectures:
                base_runs = df[(df["dataset"] == d) & (df["architecture"] == a) & (df["protocol"] == "p2") & (df["mitigation"] == "baseline")]
                mit_runs = df[(df["dataset"] == d) & (df["architecture"] == a) & (df["protocol"] == "p2") & (df["mitigation"] == m)]

                if len(base_runs) == 0 or len(mit_runs) == 0:
                    continue

                sub_cols = [c for c in df.columns if c.startswith("sub_") and c.endswith("_f1")]
                if sub_cols:
                    s_base = base_runs[sub_cols].max(axis=0).dropna().values
                    s_mit = mit_runs[sub_cols].max(axis=0).dropna().values
                else:
                    s_base = base_runs["macro_f1"].values
                    s_mit = mit_runs["macro_f1"].values

                min_l = min(len(s_base), len(s_mit))
                if min_l < 3:
                    continue

                ev = evaluate_mitigation_condition(s_base[:min_l], s_mit[:min_l])
                m_comparisons.append({
                    "dataset": d,
                    "architecture": a,
                    "mitigation": m,
                    "eval": ev,
                })

        if m_comparisons:
            eval_family = evaluate_rq3_family(m_comparisons)
            all_comparisons.extend(eval_family)

    table_df = pd.DataFrame(all_comparisons)
    return table_df
