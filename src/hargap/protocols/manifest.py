"""Fold manifest schema, serialization, and cryptographic hashing."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import hashlib
import json
from pathlib import Path


@dataclass
class FoldManifest:
    dataset: str
    protocol: str  # "p1", "p2", "official"
    fold_id: int
    seed: int
    train_subjects: List[int]
    val_subjects: List[int]
    test_subjects: List[int]
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    window_counts: Dict[str, int]
    cache_sha256: str
    config_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "fold_id": int(self.fold_id),
            "seed": int(self.seed),
            "train_subjects": [int(s) for s in sorted(self.train_subjects)],
            "val_subjects": [int(s) for s in sorted(self.val_subjects)],
            "test_subjects": [int(s) for s in sorted(self.test_subjects)],
            "train_indices": [int(i) for i in self.train_indices],
            "val_indices": [int(i) for i in self.val_indices],
            "test_indices": [int(i) for i in self.test_indices],
            "window_counts": {k: int(v) for k, v in self.window_counts.items()},
            "cache_sha256": self.cache_sha256,
            "config_sha256": self.config_sha256,
        }

    def compute_sha256(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def save(self, filepath: str | Path) -> str:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        digest = self.compute_sha256()

        json_path = path.with_suffix(".json")
        sha_path = path.with_suffix(".sha256")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        with open(sha_path, "w", encoding="utf-8") as f:
            f.write(f"{digest}  {json_path.name}\n")

        return digest

    @classmethod
    def load(cls, filepath: str | Path) -> "FoldManifest":
        json_path = Path(filepath).with_suffix(".json")
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)
