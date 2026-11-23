"""Streamlit demonstrator app: replay, results explorer, integrity, and live stretch panel."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEMO = ROOT / "demonstrator"
for p in (str(SRC), str(ROOT), str(DEMO)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from demonstrator.panels import explorer, integrity, live, replay  # type: ignore
except Exception:
    from panels import explorer, integrity, live, replay  # type: ignore


st.set_page_config(page_title="HARGAP Demonstrator", layout="wide")
st.title("HARGAP Demonstrator")
st.caption("Cross-subject generalisation gap (P1 vs P2) — replay, results, integrity, and live stretch demo")

panel = st.sidebar.radio(
    "Panel",
    [
        "Replay (DR-1)",
        "Results explorer (DR-2)",
        "Integrity (DR-3)",
        "Live phone stream (DR-4, stretch)",
    ],
)

if panel == "Replay (DR-1)":
    replay.render(ROOT)
elif panel == "Results explorer (DR-2)":
    explorer.render(ROOT)
elif panel == "Integrity (DR-3)":
    integrity.render(ROOT)
else:
    live.render(ROOT)
