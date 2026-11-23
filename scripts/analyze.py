"""Analyse RESULTS/runs.jsonl and emit summary statistics and budget report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hargap.eval.database import ResultDB
from hargap.report.tables import generate_table_rq1_gap, generate_table_rq2_rankings, generate_table_rq3_mitigations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS/runs.jsonl")
    ap.add_argument("--out", default="RESULTS/analysis_summary.json")
    ap.add_argument("--budget-hours", type=float, default=40.0, help="Budget cap for reporting")
    ap.add_argument("--emit-tables", action="store_true", help="Also generate RQ1-RQ3 table CSV/JSON outputs")
    ap.add_argument("--tables-dir", default="RESULTS/tables")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    db = ResultDB(results_path)
    records = db.load_all()
    df = db.to_dataframe()

    summary: dict = {
        "results_path": str(results_path),
        "n_runs": len(records),
        "datasets": sorted(df["dataset"].dropna().unique().tolist()) if not df.empty and "dataset" in df else [],
        "architectures": sorted(df["architecture"].dropna().unique().tolist()) if not df.empty and "architecture" in df else [],
        "protocols": sorted(df["protocol"].dropna().unique().tolist()) if not df.empty and "protocol" in df else [],
        "mitigations": sorted(df["mitigation"].dropna().unique().tolist()) if not df.empty and "mitigation" in df else [],
    }

    if not df.empty and "wall_clock_sec" in df:
        total_sec = float(pd.to_numeric(df["wall_clock_sec"], errors="coerce").fillna(0).sum())
        summary["budget"] = {
            "total_wall_clock_seconds": total_sec,
            "total_wall_clock_hours": total_sec / 3600.0,
            "budget_hours": args.budget_hours,
            "within_budget": (total_sec / 3600.0) <= args.budget_hours if args.budget_hours > 0 else None,
        }
        if "macro_f1" in df:
            summary["macro_f1_overall"] = {
                "mean": float(pd.to_numeric(df["macro_f1"], errors="coerce").mean()),
                "std": float(pd.to_numeric(df["macro_f1"], errors="coerce").std(ddof=0)),
                "min": float(pd.to_numeric(df["macro_f1"], errors="coerce").min()),
                "max": float(pd.to_numeric(df["macro_f1"], errors="coerce").max()),
            }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.emit_tables and not df.empty:
        tables_dir = Path(args.tables_dir)
        if not tables_dir.is_absolute():
            tables_dir = ROOT / tables_dir
        tables_dir.mkdir(parents=True, exist_ok=True)
        try:
            rq1 = generate_table_rq1_gap(df)
            if not rq1.empty:
                rq1.to_csv(tables_dir / "rq1_gap.csv", index=False)
                rq1.to_json(tables_dir / "rq1_gap.json", orient="records", indent=2)
                summary["rq1_rows"] = int(len(rq1))
        except Exception as e:
            summary["rq1_error"] = str(e)
        try:
            rq2 = generate_table_rq2_rankings(df)
            with open(tables_dir / "rq2_rankings.json", "w", encoding="utf-8") as f:
                json.dump(rq2, f, indent=2, default=str)
            summary["rq2_datasets"] = list(rq2.keys()) if isinstance(rq2, dict) else []
        except Exception as e:
            summary["rq2_error"] = str(e)
        try:
            rq3 = generate_table_rq3_mitigations(df)
            if not rq3.empty:
                rq3.to_csv(tables_dir / "rq3_mitigations.csv", index=False)
                rq3.to_json(tables_dir / "rq3_mitigations.json", orient="records", indent=2)
                summary["rq3_rows"] = int(len(rq3))
        except Exception as e:
            summary["rq3_error"] = str(e)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
