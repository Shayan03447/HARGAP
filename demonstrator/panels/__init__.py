"""Shared helpers for demonstrator panels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_runs(root: Path, rel: str = "RESULTS/runs.jsonl") -> List[Dict[str, Any]]:
    path = root / rel
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def runs_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    flat = []
    for r in records:
        m = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
        flat.append(
            {
                "run_id": r.get("run_id"),
                "dataset": r.get("dataset"),
                "architecture": r.get("architecture"),
                "protocol": r.get("protocol"),
                "fold_id": r.get("fold_id"),
                "seed": r.get("seed"),
                "mitigation": r.get("mitigation"),
                "macro_f1": m.get("macro_f1", r.get("macro_f1")),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "accuracy": m.get("accuracy"),
                "manifest_sha256": r.get("manifest_sha256"),
                "checkpoint_path": r.get("checkpoint_path"),
                "timestamp": r.get("timestamp"),
            }
        )
    return pd.DataFrame(flat)
