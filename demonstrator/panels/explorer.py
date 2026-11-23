"""DR-2: Results explorer — distributions, confusion matrices, mitigation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from demonstrator.panels import load_runs, runs_df


def render(root: Path):
    st.header("Results explorer (DR-2)")
    records = load_runs(root)
    df = runs_df(records)
    if df.empty:
        st.info("No RESULTS/runs.jsonl records found yet. Run experiments to populate results.")
        st.code("python scripts/run_experiment.py --config configs/runs/example_synthetic_cnn_p2_baseline.yaml --smoke --allow-unverified")
        return

    st.subheader("Runs overview")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Runs", len(df))
    f2.metric("Datasets", df["dataset"].nunique() if "dataset" in df else 0)
    f3.metric("Architectures", df["architecture"].nunique() if "architecture" in df else 0)
    f4.metric("Mean macro-F1", f"{pd.to_numeric(df['macro_f1'], errors='coerce').mean():.3f}" if "macro_f1" in df else "n/a")

    with st.expander("Filter and table", expanded=True):
        datasets = ["(all)"] + sorted([x for x in df.get("dataset", pd.Series(dtype=str)).dropna().unique().tolist()])
        protocols = ["(all)"] + sorted([x for x in df.get("protocol", pd.Series(dtype=str)).dropna().unique().tolist()])
        archs = ["(all)"] + sorted([x for x in df.get("architecture", pd.Series(dtype=str)).dropna().unique().tolist()])
        mits = ["(all)"] + sorted([x for x in df.get("mitigation", pd.Series(dtype=str)).dropna().unique().tolist()])
        c1, c2, c3, c4 = st.columns(4)
        ds = c1.selectbox("Dataset", datasets)
        pr = c2.selectbox("Protocol", protocols)
        ar = c3.selectbox("Architecture", archs)
        mi = c4.selectbox("Mitigation", mits)
        view = df.copy()
        if ds != "(all)": view = view[view["dataset"] == ds]
        if pr != "(all)": view = view[view["protocol"] == pr]
        if ar != "(all)": view = view[view["architecture"] == ar]
        if mi != "(all)": view = view[view["mitigation"] == mi]
        st.dataframe(view, use_container_width=True)

    # Per-subject/score distribution boxplots by protocol (DR-2)
    st.subheader("Score distribution (P1 vs P2)")
    if "protocol" in df.columns and "macro_f1" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        groups = []
        labels = []
        for proto in ["p1", "p2", "official"]:
            vals = pd.to_numeric(df[df["protocol"] == proto]["macro_f1"], errors="coerce").dropna().values
            if len(vals):
                groups.append(vals)
                labels.append(proto.upper())
        if groups:
            ax.boxplot(groups, tick_labels=labels, patch_artist=True)
            ax.set_ylabel("Macro-F1")
            ax.set_ylim(0, 1.05)
            ax.set_title("Macro-F1 distribution by protocol")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("No protocol-grouped scores available.")
    else:
        st.caption("Missing protocol/macro_f1 fields.")

    # Mitigation effect bars (baseline vs M1-M3) from available records
    st.subheader("Mitigation effect summary (available runs)")
    if {"mitigation", "macro_f1"}.issubset(df.columns):
        agg = df.copy()
        agg["macro_f1"] = pd.to_numeric(agg["macro_f1"], errors="coerce")
        by_m = agg.groupby("mitigation", dropna=False)["macro_f1"].mean().sort_index()
        if not by_m.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(by_m.index.astype(str), by_m.values)
            ax.set_ylabel("Mean macro-F1")
            ax.set_ylim(0, 1.05)
            ax.set_title("Mean macro-F1 by mitigation (as logged)")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    # Confusion matrix heatmap for selected run
    st.subheader("Confusion matrix (selected run)")
    with_cm = [r for r in records if (r.get("metrics") or {}).get("confusion_matrix")]
    if not with_cm:
        st.caption("No confusion matrices recorded in results.")
        return
    options = [f"{r.get('run_id','?')} | {r.get('dataset')} {r.get('architecture')} {r.get('protocol')} f{r.get('fold_id')} {r.get('mitigation')}" for r in with_cm]
    idx = st.selectbox("Run", list(range(len(options))), format_func=lambda i: options[i])
    rec = with_cm[idx]
    cm = np.array(rec["metrics"]["confusion_matrix"], dtype=float)
    if cm.size == 0:
        st.caption("Empty confusion matrix.")
        return
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-8)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    k = cm_norm.shape[0]
    ticks = list(range(k))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in ticks])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    st.json({"run_id": rec.get("run_id"), "macro_f1": (rec.get("metrics") or {}).get("macro_f1"), "manifest_sha256": rec.get("manifest_sha256")})
