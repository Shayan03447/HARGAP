"""Reporting tables and figures for Chapters 5 and 6."""

from hargap.report.tables import (
    generate_table_rq1_gap,
    generate_table_rq2_rankings,
    generate_table_rq3_mitigations,
)
from hargap.report.figures import (
    set_matplotlib_style,
    plot_gap_bars,
    plot_subject_boxplots,
    plot_confusion_heatmap,
)

# Spec-compatible aliases
tables_for_ch5 = generate_table_rq1_gap
figures_for_ch56 = None  # convenience: use plot_* and scripts/make_report_tables.py


def generate_all_reports(results_path="RESULTS/runs.jsonl", out_dir="RESULTS/tables", fig_dir="RESULTS/figures"):
    """Generate RQ1-RQ3 tables/figures from a results JSONL path."""
    from pathlib import Path
    import json
    from hargap.eval.database import ResultDB
    from hargap.report.tables import generate_table_rq2_rankings, generate_table_rq3_mitigations
    from hargap.report.figures import plot_gap_bars

    db = ResultDB(results_path)
    df = db.to_dataframe()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fig = Path(fig_dir); fig.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if df.empty:
        return outputs
    rq1 = generate_table_rq1_gap(df)
    if not rq1.empty:
        rq1.to_csv(out / "rq1_gap.csv", index=False)
        outputs["rq1"] = str(out / "rq1_gap.csv")
        try:
            plot_gap_bars(rq1, fig / "rq1_gap_bars.png")
        except Exception:
            pass
    rq2 = generate_table_rq2_rankings(df)
    with open(out / "rq2_rankings.json", "w", encoding="utf-8") as f:
        json.dump(rq2, f, indent=2, default=str)
    outputs["rq2"] = str(out / "rq2_rankings.json")
    rq3 = generate_table_rq3_mitigations(df)
    if not rq3.empty:
        rq3.to_csv(out / "rq3_mitigations.csv", index=False)
        outputs["rq3"] = str(out / "rq3_mitigations.csv")
    return outputs

__all__ = [
    "generate_table_rq1_gap",
    "generate_table_rq2_rankings",
    "generate_table_rq3_mitigations",
    "set_matplotlib_style",
    "plot_gap_bars",
    "plot_subject_boxplots",
    "plot_confusion_heatmap",
    "tables_for_ch5",
    "generate_all_reports",
]
