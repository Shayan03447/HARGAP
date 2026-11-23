"""Download/verify dataset provenance and record file hashes.

Raw datasets are not redistributed. This script helps locate expected files,
compute SHA256 hashes for provenance, and emit a manifest for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


DATASET_INFO = {
    "ucihar": {
        "expected": [
            "train/subject_train.txt",
            "train/y_train.txt",
            "train/Inertial Signals/body_acc_x_train.txt",
            "test/subject_test.txt",
            "test/y_test.txt",
            "test/Inertial Signals/body_acc_x_test.txt",
        ],
        "source": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        "citation": "Anguita et al., ESANN 2013",
    },
    "pamap2": {
        "expected": [f"subject10{i}.dat" for i in range(1, 10)],
        "source": "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring",
        "citation": "Reiss & Stricker, ISWC 2012 (doi:10.1109/ISWC.2012.13)",
    },
    "mhealth": {
        "expected": [f"mHealth_subject{i}.log" for i in range(1, 11)],
        "source": "https://archive.ics.uci.edu/dataset/319/mhealth+dataset",
        "citation": "Banos et al., 2014 (doi:10.1007/978-3-319-13105-4_14)",
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes(raw_dir: Path) -> List[Dict[str, str]]:
    entries = []
    if not raw_dir.exists():
        return entries
    for p in sorted([x for x in raw_dir.rglob("*") if x.is_file()]):
        entries.append(
            {
                "path": str(p.relative_to(raw_dir)).replace("\\", "/"),
                "sha256": sha256_file(p),
                "bytes": str(p.stat().st_size),
            }
        )
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="List datasets and expected locations")
    ap.add_argument("--dataset", choices=sorted(DATASET_INFO.keys()))
    ap.add_argument("--raw-dir", help="Path to raw dataset directory")
    ap.add_argument("--record-hashes", action="store_true", help="Compute and save hashes if files exist")
    ap.add_argument("--out", default="RESULTS/logs/data_provenance.json", help="Output provenance JSON path")
    args = ap.parse_args()

    if args.list or not args.dataset:
        for name, info in DATASET_INFO.items():
            print(f"[{name}]")
            print(f"  source: {info['source']}")
            print(f"  citation: {info['citation']}")
            print(f"  expected (sample): {info['expected'][:3]}{' ...' if len(info['expected'])>3 else ''}")
        return 0

    info = DATASET_INFO[args.dataset]
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(f"data/raw/{args.dataset}")
    print(f"Dataset: {args.dataset}")
    print(f"Source:  {info['source']}")
    print(f"Citation:{info['citation']}")
    print(f"Raw dir: {raw_dir}")
    exists = raw_dir.exists()
    print(f"Exists:  {exists}")
    missing = []
    for rel in info["expected"]:
        if not (raw_dir / rel).exists():
            missing.append(rel)
    if missing:
        print(f"Missing expected files ({len(missing)}):")
        for m in missing[:20]:
            print(f"  - {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")
    else:
        print("All expected sentinel files found.")

    provenance = {
        "dataset": args.dataset,
        "raw_dir": str(raw_dir),
        "source": info["source"],
        "citation": info["citation"],
        "expected_sentinels": info["expected"],
        "missing_sentinels": missing,
        "files": [],
    }
    if args.record_hashes and exists:
        print("Computing file hashes...")
        provenance["files"] = collect_hashes(raw_dir)
        print(f"Hashed {len(provenance['files'])} files.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Provenance written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
