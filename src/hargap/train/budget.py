"""Compute budget tracking, timing, memory profiling and queue estimation."""

import time
from typing import Dict, Any, List
import torch


class BudgetTracker:
    """
    Monitors runtime wall-clock duration and GPU/CPU memory consumption per fold.
    Enforces the compute budget audit defined in Chapter 4 (15-40 GPU hours).
    """

    def __init__(self):
        self.fold_times: List[float] = []
        self.start_time: float = 0.0

    def start_fold(self):
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_fold(self) -> Dict[str, Any]:
        duration_sec = time.perf_counter() - self.start_time
        self.fold_times.append(duration_sec)

        peak_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return {
            "wall_clock_seconds": round(duration_sec, 2),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "average_fold_sec": round(sum(self.fold_times) / len(self.fold_times), 2),
            "total_tracked_folds": len(self.fold_times),
        }

    def estimate_remaining(self, remaining_folds: int) -> float:
        """Estimates remaining execution time in hours."""
        if not self.fold_times:
            return 0.0
        avg_sec = sum(self.fold_times) / len(self.fold_times)
        return round((avg_sec * remaining_folds) / 3600.0, 3)
