"""DR-4 (stretch): Live phone stream via Phyphox websocket with sliding-window inference."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def render(root: Path):
    st.header("Live phone stream (DR-4, stretch)")
    st.warning(
        "Stretch feature (off by default). This panel provides the integration scaffold for Phyphox websocket streaming. "
        "Enable and configure only after core P1/P2 experiments and verification are complete."
    )

    st.markdown(
        """
**Intended flow (per outline):**
1. Connect to Phyphox websocket (default `ws://<phone-ip>:8080` or configured port) streaming accelerometer/gyroscope.
2. Buffer samples and form 128-sample windows at 50 Hz (resample if device rate differs).
3. Apply declared calibration (e.g. first 10 s for M1 subject normalisation) and run the selected LOSO checkpoint.
4. Display live predicted activity; presenter is an unseen subject by construction.

**Notes:** Requires network access to the phone, compatible sensor channels, and a trained checkpoint. Any deviation (windowing/calibration) must be declared in captions/logs.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        ws_url = st.text_input("Phyphox websocket URL", value="ws://192.168.0.100:8080")
        checkpoint = st.text_input("Checkpoint path (relative to repo)", value="RESULTS/checkpoints/best_loso.pt")
    with col2:
        sample_rate = st.number_input("Incoming sample rate (Hz)", min_value=10, max_value=200, value=100)
        calib_sec = st.number_input("Calibration seconds (M1, declared)", min_value=0, max_value=120, value=10)

    if st.button("Check prerequisites"):
        ckpt = root / checkpoint
        status = {
            "websocket_url": ws_url,
            "checkpoint_exists": ckpt.exists(),
            "checkpoint_path": str(ckpt),
            "incoming_hz": sample_rate,
            "calibration_sec": calib_sec,
            "enabled": False,
            "reason": "Stretch scaffold only; live inference loop requires Phyphox client runtime and sensor mapping configuration.",
        }
        st.json(status)

    st.caption(
        "To implement full live inference, install/add a websocket client, map Phyphox payload fields to model channels, "
        "maintain a sliding buffer (128 samples, stride configurable), and feed windows through the same preprocessing/checkpoint path as offline replay."
    )
