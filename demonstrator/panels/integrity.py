"""DR-3: Integrity panel — run verification suite and display manifest hashes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from demonstrator.panels import load_runs, runs_df


def render(root: Path):
    st.header("Integrity panel (DR-3)")
    st.write("Runs the fast verification suite and displays manifest/run linkage for leakage-aware auditing.")

    report_path = root / "RESULTS" / "verify_report.json"

    c1, c2 = st.columns([1, 2])
    with c1:
        run_btn = st.button("Run verification suite", type="primary")
        show_existing = st.button("Load latest report")
    with c2:
        st.caption(f"Report path: `{report_path}`")

    if run_btn:
        with st.spinner("Running verification suite (pytest)..."):
            cmd = [sys.executable, "-m", "pytest", "-q",
                   "tests/test_disjointness.py", "tests/test_window_boundaries.py",
                   "tests/test_fitting_scope.py", "tests/test_augmentation_scope.py",
                   "tests/test_manifests.py", "tests/test_stats_implementations.py"]
            env = dict(**__import__('os').environ)
            env["PYTHONPATH"] = str(root / "src")
            proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env)
            status = "GREEN" if proc.returncode == 0 else "RED"
            report = {
                "status": status,
                "is_green": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-4000:],
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if proc.returncode == 0:
            st.success("Verification suite: GREEN")
        else:
            st.error("Verification suite: RED")
        st.code((proc.stdout + "\n" + proc.stderr).strip() or "(no output)")
    elif show_existing or report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            ok = report.get("is_green") or report.get("status") == "GREEN"
            (st.success if ok else st.error)(f"Verification report status: {report.get('status', 'unknown')}")
            st.json(report)
        except Exception as e:
            st.warning(f"Could not read report: {e}")

    st.subheader("Manifest index and hashes")
    manifests_dir = root / "MANIFESTS"
    manifest_files = sorted(manifests_dir.glob("*.json")) if manifests_dir.exists() else []
    if not manifest_files:
        st.info("No manifests found in MANIFESTS/. Build folds first.")
    else:
        rows = []
        for p in manifest_files[:200]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                sha_path = p.with_suffix(".sha256")
                sha = sha_path.read_text(encoding="utf-8").strip().split()[0] if sha_path.exists() else ""
                rows.append({
                    "file": p.name,
                    "dataset": d.get("dataset"),
                    "protocol": d.get("protocol"),
                    "fold_id": d.get("fold_id"),
                    "seed": d.get("seed"),
                    "sha256": sha or d.get("sha256", ""),
                    "cache_sha256": d.get("cache_sha256", ""),
                })
            except Exception:
                rows.append({"file": p.name, "error": "unreadable"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Runs linked to manifests/checkpoints")
    records = load_runs(root)
    df = runs_df(records)
    if df.empty:
        st.caption("No runs logged yet.")
    else:
        cols = [c for c in ["run_id", "dataset", "architecture", "protocol", "fold_id", "seed", "mitigation", "macro_f1", "manifest_sha256", "checkpoint_path", "timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)

import pandas as pd  # noqa: E402
