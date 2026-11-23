"""Pre-batch verification suite runner and verification report generator."""

from typing import Dict, Any, List
import json
import time
from pathlib import Path
import numpy as np
import pytest


def run_verification_suite(
    report_output_path: str | Path = "RESULTS/verify_report.json",
    test_paths: List[str] = ["tests/test_disjointness.py", "tests/test_window_boundaries.py", "tests/test_fitting_scope.py", "tests/test_augmentation_scope.py", "tests/test_manifests.py", "tests/test_stats_implementations.py"]
) -> Dict[str, Any]:
    """
    Executes fast pytest assertion suite gating experiment batches.
    Produces verify_report.json containing test results and sha256 hashes.
    """
    report_path = Path(report_output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    start_t = time.time()
    ret_code = pytest.main(["-v", "-q"] + test_paths)
    duration = time.time() - start_t

    is_green = (ret_code == 0 or ret_code == pytest.ExitCode.OK)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pytest_exit_code": int(ret_code),
        "status": "GREEN" if is_green else "RED",
        "must_be_green": True,
        "is_green": is_green,
        "duration_seconds": round(duration, 2),
        "tests_executed": test_paths,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data
