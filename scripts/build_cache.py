"""Build cached window tensors and metadata for a dataset."""

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

from hargap.data.cache import build_cache
from hargap.data.loaders import generate_synthetic_benchmark, load_mhealth, load_pamap2, load_ucihar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ucihar", "pamap2", "mhealth", "synthetic"])
    ap.add_argument("--raw-dir", help="Raw dataset directory (not used for synthetic)")
    ap.add_argument("--out", required=True, help="Output cache base path (.npz written)")
    ap.add_argument("--min-windows-per-activity", type=int, default=5, help="PAMAP2 subject screen threshold")
    ap.add_argument("--synthetic-subjects", type=int, default=10)
    ap.add_argument("--synthetic-classes", type=int, default=6)
    ap.add_argument("--synthetic-channels", type=int, default=9)
    ap.add_argument("--synthetic-windows-per-class", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    audit = {"dataset": args.dataset}
    if args.dataset == "synthetic":
        windows, meta = generate_synthetic_benchmark(
            n_subjects=args.synthetic_subjects,
            n_classes=args.synthetic_classes,
            n_channels=args.synthetic_channels,
            windows_per_subject_class=args.synthetic_windows_per_class,
            seed=args.seed,
        )
    elif args.dataset == "ucihar":
        if not args.raw_dir:
            ap.error("--raw-dir is required for ucihar")
        windows, meta = load_ucihar(args.raw_dir)
    elif args.dataset == "pamap2":
        if not args.raw_dir:
            ap.error("--raw-dir is required for pamap2")
        windows, meta, subject_audit = load_pamap2(args.raw_dir, min_windows_per_activity=args.min_windows_per_activity)
        audit["subject_audit"] = subject_audit
    else:  # mhealth
        if not args.raw_dir:
            ap.error("--raw-dir is required for mhealth")
        windows, meta = load_mhealth(args.raw_dir)

    if len(windows) == 0:
        raise SystemExit("No windows produced. Check raw data path and dataset contents.")

    out = Path(args.out)
    store = build_cache(windows, meta, out)
    audit.update(
        {
            "cache_base": str(out),
            "n_windows": int(store.n_windows),
            "n_channels": int(store.n_channels),
            "window_length": int(store.window_length),
            "subjects": sorted([int(x) for x in pd.Series(meta["subject_id"]).unique().tolist()]),
            "activities": sorted([int(x) for x in pd.Series(meta["activity_code"]).unique().tolist()]),
            "cache_sha256": store.sha256,
            "class_counts": meta.groupby(["subject_id", "activity_code"]).size().reset_index(name="n").to_dict(orient="records"),
        }
    )
    audit_path = out.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"Dataset: {args.dataset}")
    print(f"Windows: {store.n_windows} | Shape: {store.shape}")
    print(f"Subjects: {len(audit['subjects'])} {audit['subjects']}")
    print(f"Cache sha256: {store.sha256}")
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
