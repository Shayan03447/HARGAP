"""DR-1: Replay cached held-out windows with P1 vs LOSO predictions (when artifacts exist)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

from hargap.data.cache import WindowStore
from hargap.mitigations.normalizer import GlobalNormalizer
from hargap.models import build_model
from hargap.protocols.manifest import FoldManifest
from demonstrator.panels import load_runs


def _list_caches(root: Path):
    d = root / "data" / "cache"
    return sorted(d.glob("*.npz")) if d.exists() else []


def _load_checkpoint(path: Path, n_channels: int, n_classes: int):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None, "unreadable"
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    arch = ((cfg.get("model") or {}).get("name")) or "cnn"
    params = ((cfg.get("model") or {}).get("params")) or {}
    try:
        model = build_model(arch, n_channels=n_channels, n_classes=n_classes, **params)
        if state:
            model.load_state_dict(state, strict=False)
        model.eval()
        return model, arch
    except Exception as e:
        return None, str(e)


def render(root: Path):
    st.header("Replay mode (DR-1)")
    st.write(
        "Replay held-out cached windows and compare predictions from available P1/LOSO checkpoints. "
        "If checkpoints/manifests are not present, this panel provides an artifact-guided replay view."
    )

    caches = _list_caches(root)
    manifests = sorted((root / "MANIFESTS").glob("*.json")) if (root / "MANIFESTS").exists() else []
    runs = load_runs(root)

    c1, c2 = st.columns(2)
    with c1:
        cache_path = st.selectbox("Cached windows", ["(none)"] + [str(p.relative_to(root)) for p in caches])
    with c2:
        manifest_path = st.selectbox("Fold manifest", ["(none)"] + [str(p.relative_to(root)) for p in manifests])

    if cache_path == "(none)" or manifest_path == "(none)":
        st.info("Select a cache and a manifest to replay held-out windows. Build them with scripts/build_cache.py and scripts/build_folds.py.")
        return

    try:
        store = WindowStore.load(root / cache_path)
        manifest = FoldManifest.load(root / manifest_path)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return

    test_idxs = np.array(manifest.test_indices, dtype=np.int64)
    if len(test_idxs) == 0:
        st.warning("Manifest has no test indices.")
        return
    test_x, test_y, test_subs = store.get_subset(test_idxs)
    n_classes = int(max(int(test_y.max()) + 1 if len(test_y) else 1, (manifest.window_counts or {}).get("n_classes", 0) or 0, 2))

    # Find candidate checkpoints for same dataset/fold/protocol context
    def match_runs(proto: str):
        out = []
        for r in runs:
            if r.get("dataset") != manifest.dataset:
                continue
            if r.get("protocol") != proto:
                continue
            cp = r.get("checkpoint_path")
            if not cp:
                continue
            p = Path(cp)
            if not p.exists():
                # Results may have been recorded on another machine: fall back to local RESULTS/checkpoints/<name>.
                p = root / "RESULTS" / "checkpoints" / Path(str(cp).replace("\\", "/")).name
            if p.exists():
                out.append({**r, "checkpoint_path": str(p)})
        return out

    p1_runs = [r for r in match_runs("p1") if r.get("mitigation", "baseline") == "baseline"]
    p2_runs = [r for r in (match_runs("p2") or match_runs(manifest.protocol)) if r.get("mitigation", "baseline") == "baseline"]

    # Models were trained on globally z-scored windows (fit on THEIR OWN training split only),
    # so replay must apply the same transform or predictions are meaningless.
    def norm_for(proto: str):
        if manifest.protocol == proto:
            m = manifest
        else:
            name = (f"{manifest.dataset}_p1_seed1.json" if proto == "p1"
                    else f"{manifest.dataset}_p2_fold{int(manifest.fold_id):02d}_seed1.json")
            f = root / "MANIFESTS" / name
            if not f.exists():
                return None
            m = FoldManifest.load(f)
        xtr, _, _ = store.get_subset(np.array(m.train_indices, dtype=np.int64))
        n = GlobalNormalizer()
        n.fit(xtr)
        return n

    norm_p1 = norm_for("p1") if p1_runs else None
    norm_p2 = norm_for("p2") if p2_runs else None

    st.caption(f"Dataset: {manifest.dataset} | fold: {manifest.fold_id} | protocol: {manifest.protocol} | test windows: {len(test_idxs)} | subject(s): {manifest.test_subjects}")

    # Load models if available
    model_p1 = model_p2 = None
    if p1_runs:
        model_p1, _ = _load_checkpoint(Path(p1_runs[0]["checkpoint_path"]), store.n_channels, n_classes)
    if p2_runs:
        # prefer matching fold
        chosen = None
        for r in p2_runs:
            if int(r.get("fold_id", -1)) == int(manifest.fold_id):
                chosen = r; break
        chosen = chosen or p2_runs[0]
        model_p2, _ = _load_checkpoint(Path(chosen["checkpoint_path"]), store.n_channels, n_classes)

    idx = st.slider("Window index (held-out stream)", 0, len(test_x) - 1, 0)
    win = test_x[idx]
    true_y = int(test_y[idx]) if len(test_y) else -1
    true_sub = int(test_subs[idx]) if len(test_subs) else (manifest.test_subjects[0] if manifest.test_subjects else -1)

    st.subheader("Window signal (first 3 channels)")
    show_t = np.arange(win.shape[1]) / 50.0
    chart_df = pd.DataFrame({f"ch{i}": win[i] for i in range(min(3, win.shape[0]))}, index=show_t)
    st.line_chart(chart_df)

    def predict(model, norm):
        if model is None:
            return None, None
        with torch.no_grad():
            w = norm.transform(win[None, ...]) if norm is not None else win[None, ...]
            x = torch.from_numpy(w).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred = int(np.argmax(probs))
        return pred, probs

    pred_p1, prob_p1 = predict(model_p1, norm_p1)
    pred_p2, prob_p2 = predict(model_p2, norm_p2)

    m1, m2, m3 = st.columns(3)
    m1.metric("True label", str(true_y))
    m2.metric("P1 prediction", str(pred_p1) if pred_p1 is not None else "n/a")
    m3.metric("LOSO/P2 prediction", str(pred_p2) if pred_p2 is not None else "n/a")

    st.write({"test_subject": true_sub, "p1_correct": (pred_p1 == true_y) if pred_p1 is not None else None, "p2_correct": (pred_p2 == true_y) if pred_p2 is not None else None})

    # Running macro-F1 over played prefix (when predictions available)
    if pred_p1 is not None or pred_p2 is not None:
        from sklearn.metrics import f1_score
        prefix = st.slider("Replay prefix length for running macro-F1", 1, len(test_x), min(50, len(test_x)))
        y_true = test_y[:prefix]
        rows = {"windows": list(range(1, prefix + 1))}
        with torch.no_grad():
            def _X(norm):
                xs = test_x[:prefix]
                return torch.from_numpy(norm.transform(xs) if norm is not None else xs).float()
            if model_p1 is not None:
                p = torch.argmax(model_p1(_X(norm_p1)), dim=1).numpy()
                rows["P1_running_macroF1"] = [f1_score(y_true[:i], p[:i], average="macro", zero_division=0) for i in range(1, prefix + 1)]
            if model_p2 is not None:
                p = torch.argmax(model_p2(_X(norm_p2)), dim=1).numpy()
                rows["LOSO_running_macroF1"] = [f1_score(y_true[:i], p[:i], average="macro", zero_division=0) for i in range(1, prefix + 1)]
        chart = pd.DataFrame({k: v for k, v in rows.items() if k != "windows"}, index=rows["windows"])
        if not chart.empty:
            st.line_chart(chart)
        else:
            st.caption("Load matching checkpoints to compute running macro-F1 curves.")
    else:
        st.caption("No matching checkpoints found for this dataset/protocol. The panel shows the held-out stream; train runs to enable side-by-side predictions.")
