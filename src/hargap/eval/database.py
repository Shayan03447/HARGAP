"""Result database logging to RESULTS/runs.jsonl and query abstractions."""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import pandas as pd


class ResultDB:
    """Thread-safe append-only result logger and query store."""

    def __init__(self, jsonl_path: str | Path = "RESULTS/runs.jsonl"):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, run_record: Dict[str, Any]):
        """Appends a completed run entry to the jsonl log."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        """Loads all recorded runs into memory."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def to_dataframe(self) -> pd.DataFrame:
        """Converts runs to a flattened pandas DataFrame for tabular analysis."""
        records = self.load_all()
        if not records:
            return pd.DataFrame()

        flattened = []
        for r in records:
            row = {
                "run_id": r.get("run_id"),
                "dataset": r.get("dataset"),
                "architecture": r.get("architecture"),
                "protocol": r.get("protocol"),
                "fold_id": r.get("fold_id"),
                "seed": r.get("seed"),
                "mitigation": r.get("mitigation"),
                "macro_f1": r.get("metrics", {}).get("macro_f1"),
                "balanced_accuracy": r.get("metrics", {}).get("balanced_accuracy"),
                "accuracy": r.get("metrics", {}).get("accuracy"),
                "best_epoch": r.get("best_epoch"),
                "total_epochs": r.get("total_epochs"),
                "wall_clock_sec": r.get("budget", {}).get("wall_clock_seconds", 0.0),
                "manifest_sha256": r.get("manifest_sha256"),
                "config_sha256": r.get("config_sha256"),
            }
            # Flatten per-subject scores if present
            per_sub = r.get("per_subject_macro_f1", {})
            for sub_id, score in per_sub.items():
                row[f"sub_{sub_id}_f1"] = score
            flattened.append(row)

        return pd.DataFrame(flattened)
