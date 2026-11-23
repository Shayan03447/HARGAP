"""Build P1/P2 (and UCI official reference) fold manifests from a cached dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hargap.data.cache import WindowStore
from hargap.protocols.p1 import build_p1_folds
from hargap.protocols.p2 import build_p2_folds
from hargap.protocols.reference import build_official_ucihar_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ucihar", "pamap2", "mhealth", "synthetic"])
    ap.add_argument("--cache", required=True, help="Cache base path (e.g. data/cache/ucihar_windows.npz)")
    ap.add_argument("--out-dir", default="MANIFESTS", help="Output directory for manifests")
    ap.add_argument("--p1-seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--p2-seed", type=int, default=1)
    ap.add_argument("--skip-p1", action="store_true")
    ap.add_argument("--skip-p2", action="store_true")
    ap.add_argument("--skip-official", action="store_true")
    args = ap.parse_args()

    store = WindowStore.load(args.cache)
    meta = store.metadata
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    summary = {"dataset": args.dataset, "cache": str(args.cache), "cache_sha256": store.sha256, "manifests": []}

    if not args.skip_p2:
        for m in build_p2_folds(meta, dataset_name=args.dataset, cache_sha256=store.sha256, seed=args.p2_seed):
            base = out_dir / f"{args.dataset}_p2_fold{m.fold_id:02d}_seed{m.seed}"
            digest = m.save(base)
            written.append(str(base.with_suffix('.json')))
            summary["manifests"].append({"file": base.name + ".json", "sha256": digest, "protocol": "p2", "fold_id": m.fold_id})

    if not args.skip_p1:
        for m in build_p1_folds(meta, dataset_name=args.dataset, cache_sha256=store.sha256, seeds=args.p1_seeds):
            base = out_dir / f"{args.dataset}_p1_seed{m.seed}"
            digest = m.save(base)
            written.append(str(base.with_suffix('.json')))
            summary["manifests"].append({"file": base.name + ".json", "sha256": digest, "protocol": "p1", "fold_id": m.fold_id, "seed": m.seed})

    if args.dataset == "ucihar" and not args.skip_official:
        m = build_official_ucihar_manifest(meta, cache_sha256=store.sha256, seed=1)
        base = out_dir / "ucihar_official_fold00_seed1"
        digest = m.save(base)
        written.append(str(base.with_suffix('.json')))
        summary["manifests"].append({"file": base.name + ".json", "sha256": digest, "protocol": "official", "fold_id": m.fold_id})

    index_path = out_dir / f"{args.dataset}_manifests_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(written)} manifests to {out_dir}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
