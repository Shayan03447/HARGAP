"""Raw dataset loaders for UCI-HAR, PAMAP2, MHEALTH and synthetic testing benchmarks."""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from hargap.data.transform import (
    heal_pamap2_gaps,
    resample_50hz,
    segment_stream,
    window_segment,
)


def load_ucihar(raw_dir: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Loads UCI-HAR dataset from raw release files or supplied windows.
    Returns:
        windows: (N, 9, 128) float32 array
        meta_df: pandas DataFrame with subject_id, activity_code, segment_id, window_index
    """
    raw_path = Path(raw_dir)
    # Check if raw files exist
    train_signals = raw_path / "train" / "Inertial Signals"
    test_signals = raw_path / "test" / "Inertial Signals"

    if not train_signals.exists():
        # Fallback to generating or checking cached representation
        raise FileNotFoundError(f"UCI-HAR raw directory not found at {raw_path}")

    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    def _read_partition(split: str):
        split_dir = raw_path / split
        sub_file = split_dir / f"subject_{split}.txt"
        act_file = split_dir / f"y_{split}.txt"
        subjects = np.loadtxt(sub_file, dtype=int)
        activities = np.loadtxt(act_file, dtype=int) - 1  # 0-indexed (0 to 5)

        sig_arrays = []
        for sname in signal_names:
            sig_file = split_dir / "Inertial Signals" / f"{sname}_{split}.txt"
            arr = np.loadtxt(sig_file, dtype=np.float32)  # (N, 128)
            sig_arrays.append(arr)

        # Stack into (N, 9, 128)
        stacked = np.stack(sig_arrays, axis=1)
        return stacked, subjects, activities

    train_w, train_s, train_y = _read_partition("train")
    test_w, test_s, test_y = _read_partition("test")

    windows = np.concatenate([train_w, test_w], axis=0).astype(np.float32)
    all_subjects = np.concatenate([train_s, test_s])
    all_activities = np.concatenate([train_y, test_y])

    meta_records = []
    for i in range(len(windows)):
        meta_records.append({
            "subject_id": int(all_subjects[i]),
            "activity_code": int(all_activities[i]),
            "segment_id": int(all_subjects[i] * 100 + all_activities[i]),
            "window_index": i,
            "dataset": "ucihar"
        })

    meta_df = pd.DataFrame(meta_records)
    return windows, meta_df


def load_pamap2(
    raw_dir: str | Path,
    min_windows_per_activity: int = 5
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Parses PAMAP2 per-subject files (subject101.dat to subject109.dat).
    Retains hand, chest, ankle IMU accelerometer and gyroscope (18 channels).
    Applies decimation 100->50Hz, dropout gap healing rule <=128 samples,
    and screens subjects requiring >= 5 windows per protocol activity.
    """
    raw_path = Path(raw_dir)
    protocol_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    activity_map = {act: idx for idx, act in enumerate(protocol_activities)}

    subject_files = sorted(list(raw_path.glob("subject10*.dat")))
    if not subject_files:
        raise FileNotFoundError(f"No PAMAP2 subject files found at {raw_path}")

    all_windows = []
    all_meta = []
    subject_audit = {}

    # PAMAP2 columns:
    # 0: timestamp, 1: activity_id, 2: heart_rate
    # Hand IMU: 3: temp, 4-6: acc1, 7-9: acc2, 10-12: gyro, 13-15: mag, 16-19: orientation
    # Chest IMU: 20: temp, 21-23: acc1, 24-26: acc2, 27-29: gyro, 30-32: mag, 33-36: orientation
    # Ankle IMU: 37: temp, 38-40: acc1, 41-43: acc2, 44-46: gyro, 47-49: mag, 50-53: orientation
    # 18 channels = Hand (acc1 4..6, gyro 10..12), Chest (acc1 21..23, gyro 27..29), Ankle (acc1 38..40, gyro 44..46)
    channel_indices = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46]

    for sf in subject_files:
        sub_id = int(sf.stem.replace("subject", ""))
        df_raw = pd.read_csv(sf, sep=r"\s+", header=None)
        
        # Filter protocol activities
        df_sub = df_raw[df_raw[1].isin(protocol_activities)].copy()
        if len(df_sub) == 0:
            continue

        timestamps = df_sub[0].values
        labels = df_sub[1].values
        inertial_data = df_sub[channel_indices].values.astype(np.float32)

        # Heal gaps
        data_segs, label_segs, gap_stats = heal_pamap2_gaps(
            timestamps, inertial_data, labels, max_gap_samples=128
        )

        sub_windows = []
        sub_meta = []
        seg_id_counter = 0

        for d_seg, l_seg in zip(data_segs, label_segs):
            # Resample 100Hz -> 50Hz
            if len(d_seg) >= 2:
                d_seg_50hz = resample_50hz(d_seg, current_rate=100)
                l_seg_50hz = l_seg[::2][:len(d_seg_50hz)]
                # Continuous segment extraction
                continuous = segment_stream(d_seg_50hz, l_seg_50hz, min_length_samples=256)
                for c_data, c_label in continuous:
                    mapped_label = activity_map[c_label]
                    wins, metas = window_segment(
                        c_data, mapped_label, sub_id, seg_id_counter, window_length=128, stride=64
                    )
                    if len(wins) > 0:
                        sub_windows.append(wins)
                        sub_meta.extend(metas)
                        seg_id_counter += 1

        if sub_windows:
            sub_w_stacked = np.concatenate(sub_windows, axis=0)
            sub_df = pd.DataFrame(sub_meta)
            # Check subject screen: >= 5 windows per protocol activity
            act_counts = sub_df["activity_code"].value_counts()
            retained = len(act_counts) == len(protocol_activities) and (act_counts >= min_windows_per_activity).all()
            subject_audit[sub_id] = {
                "total_windows": len(sub_w_stacked),
                "retained": retained,
                "activity_distribution": act_counts.to_dict(),
                "gap_stats": gap_stats,
            }
            if retained:
                all_windows.append(sub_w_stacked)
                all_meta.extend(sub_meta)

    final_windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    final_meta_df = pd.DataFrame(all_meta)
    return final_windows, final_meta_df, subject_audit


def load_mhealth(raw_dir: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Parses MHEALTH per-subject log files (mHealth_subject1.log to mHealth_subject10.log).
    Retains chest, wrist, ankle acc + wrist, ankle gyro (15 channels).
    Drops the null class (0).
    """
    raw_path = Path(raw_dir)
    log_files = sorted(list(raw_path.glob("mHealth_subject*.log")))
    if not log_files:
        raise FileNotFoundError(f"No MHEALTH log files found at {raw_path}")

    # Channels:
    # 0-2: chest acc, 3-4: ECG, 5-7: left ankle acc, 8-10: left ankle gyro,
    # 11-13: left ankle mag, 14-16: right lower arm acc, 17-19: right lower arm gyro,
    # 20-22: right lower arm mag, 23: label (0-12)
    # 15 channels = Chest acc (0..2), Wrist acc (14..16), Wrist gyro (17..19), Ankle acc (5..7), Ankle gyro (8..10)
    channel_indices = [0, 1, 2, 14, 15, 16, 17, 18, 19, 5, 6, 7, 8, 9, 10]

    all_windows = []
    all_meta = []

    for lf in log_files:
        sub_id = int(lf.stem.replace("mHealth_subject", ""))
        df_raw = pd.read_csv(lf, sep=r"\s+", header=None)
        
        # Drop null class (0)
        df_active = df_raw[df_raw[23] > 0].copy()
        if len(df_active) == 0:
            continue

        inertial_data = df_active[channel_indices].values.astype(np.float32)
        labels = (df_active[23].values - 1).astype(np.int64)  # 0-indexed (0 to 11)

        continuous = segment_stream(inertial_data, labels, min_length_samples=256)
        seg_id = 0
        for c_data, c_label in continuous:
            wins, metas = window_segment(
                c_data, c_label, sub_id, seg_id, window_length=128, stride=64
            )
            if len(wins) > 0:
                all_windows.append(wins)
                all_meta.extend(metas)
                seg_id += 1

    final_windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    final_meta_df = pd.DataFrame(all_meta)
    return final_windows, final_meta_df


def generate_synthetic_benchmark(
    n_subjects: int = 10,
    n_classes: int = 6,
    n_channels: int = 9,
    windows_per_subject_class: int = 15,
    seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generates deterministic synthetic wearable inertial sensor dataset
    simulating subject-specific signatures, frequency harmonics, and sensor noise.
    Matches exact (N, C, 128) format for offline testing and fast verification.
    """
    rng = np.random.RandomState(seed)
    total_windows = n_subjects * n_classes * windows_per_subject_class
    windows = np.zeros((total_windows, n_channels, 128), dtype=np.float32)
    meta_records = []

    t = np.linspace(0, 2.56, 128, endpoint=False)
    idx = 0

    # Subject physical characteristics (biomechanical offsets and scale factors)
    subject_scales = rng.uniform(0.8, 1.25, size=(n_subjects, n_channels))
    subject_biases = rng.normal(0.0, 0.15, size=(n_subjects, n_channels))

    for s_idx in range(1, n_subjects + 1):
        s_scale = subject_scales[s_idx - 1]
        s_bias = subject_biases[s_idx - 1]

        for c_idx in range(n_classes):
            base_freq = 0.8 + 0.3 * c_idx
            harmonic_weight = 0.2 + 0.1 * (c_idx % 3)

            for w_idx in range(windows_per_subject_class):
                phase = rng.uniform(0, 2 * np.pi)
                noise = rng.normal(0, 0.05, size=(n_channels, 128))

                # Synthesize tri-axial pattern with activity signature
                carrier = np.sin(2 * np.pi * base_freq * t + phase)
                harmonic = np.cos(4 * np.pi * base_freq * t + phase)
                base_signal = carrier + harmonic_weight * harmonic

                for ch in range(n_channels):
                    ch_signal = (base_signal * (1.0 + 0.1 * ch) + s_bias[ch]) * s_scale[ch] + noise[ch]
                    windows[idx, ch, :] = ch_signal.astype(np.float32)

                meta_records.append({
                    "subject_id": int(s_idx),
                    "activity_code": int(c_idx),
                    "segment_id": int(s_idx * 100 + c_idx),
                    "window_index": idx,
                    "dataset": "synthetic",
                })
                idx += 1

    meta_df = pd.DataFrame(meta_records)
    return windows, meta_df
