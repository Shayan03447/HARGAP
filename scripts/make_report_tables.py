"""Generate dissertation tables and figures from RESULTS/runs.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hargap.eval.database import ResultDB
from hargap.report.figures import plot_confusion_heatmap, plot_gap_bars, plot_subject_boxplots
from hargap.report.tables import generate_table_rq1_gap, generate_table_rq2_rankings, generate_table_rq3_mitigations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS/runs.jsonl")
    ap.add_argument("--out-dir", default="RESULTS/tables")
    ap.add_argument("--fig-dir", default="RESULTS/figures")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = ROOT / fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    db = ResultDB(results_path)
    df = db.to_dataframe()
    records = db.load_all()

    outputs = {"results_path": str(results_path), "tables": [], "figures": []}

    if df.empty:
        print(json.dumps({"status": "empty", "results_path": str(results_path)}))
        return 0

    # RQ1
    try:
        rq1 = generate_table_rq1_gap(df)
        if not rq1.empty:
            rq1.to_csv(out_dir / "rq1_gap.csv", index=False)
            rq1.to_json(out_dir / "rq1_gap.json", orient="records", indent=2)
            outputs["tables"].append("rq1_gap")
            try:
                plot_gap_bars(rq1, fig_dir / "rq1_gap_bars.png")
                outputs["figures"].append("rq1_gap_bars.png")
            except Exception as e:
                outputs.setdefault("figure_errors", {})["rq1_gap_bars"] = str(e)
    except Exception as e:
        outputs.setdefault("table_errors", {})["rq1"] = str(e)

    # RQ2
    try:
        rq2 = generate_table_rq2_rankings(df)
        with open(out_dir / "rq2_rankings.json", "w", encoding="utf-8") as f:
            json.dump(rq2, f, indent=2, default=str)
        outputs["tables"].append("rq2_rankings")
    except Exception as e:
        outputs.setdefault("table_errors", {})["rq2"] = str(e)

    # RQ3
    try:
        rq3 = generate_table_rq3_mitigations(df)
        if not rq3.empty:
            rq3.to_csv(out_dir / "rq3_mitigations.csv", index=False)
            rq3.to_json(out_dir / "rq3_mitigations.json", orient="records", indent=2)
            outputs["tables"].append("rq3_mitigations")
    except Exception as e:
        outputs.setdefault("table_errors", {})["rq3"] = str(e)

    # Subject dispersion and confusion heatmap examples from available records
    try:
        # Per-subject boxplots: compare P1 vs P2 baseline macro_f1 distributions if available
        p1 = df[(df.get("protocol") == "p1") & (df.get("mitigation") == "baseline")]["macro_f1"].dropna().tolist() if "protocol" in df else []
        p2 = df[(df.get("protocol") == "p2") & (df.get("mitigation") == "baseline")]["macro_f1"].dropna().tolist() if "protocol" in df else []
        if p1 and p2:
            plot_subject_boxplots({"P1": [float(x) for x in p1], "P2": [float(x) for x in p2]}, fig_dir / "subject_dispersion_p1_vs_p2.png")
            outputs["figures"].append("subject_dispersion_p1_vs_p2.png")
    except Exception as e:
        outputs.setdefault("figure_errors", {})["subject_dispersion"] = str(e)

    try:
        # Confusion matrix heatmap from first record that has one
        for r in records:
            cm = r.get("metrics", {}).get("confusion_matrix")
            if cm:
                cm_arr = np.array(cm)
                k = cm_arr.shape[0]
                class_names = [str(i) for i in range(k)]
                title = f"{r.get('dataset','dataset')} {r.get('architecture','arch')} {r.get('protocol','')} fold{r.get('fold_id','')}"
                plot_confusion_heatmap(cm_arr, class_names, fig_dir / "example_confusion_matrix.png", title=title)
                outputs["figures"].append("example_confusion_matrix.png")
                break
    except Exception as e:
        outputs.setdefault("figure_errors", {})["confusion_matrix"] = str(e)

    with open(out_dir / "report_outputs_index.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, default=str)

    print(json.dumps(outputs, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
