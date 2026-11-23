"""Signal transformation, resampling, dropout healing, segmentation and windowing."""

import numpy as np
from scipy.signal import decimate
from typing import List, Tuple, Dict, Any


def resample_50hz(signal: np.ndarray, current_rate: int = 100) -> np.ndarray:
    """
    Decimates signal from 100 Hz to 50 Hz using anti-alias filtering.
    Signal shape: (T, C) or (C, T)
    """
    if current_rate == 50:
        return signal
    if current_rate == 100:
        factor = 2
        # Apply along time axis (axis 0 if T > C, else axis 1)
        if signal.ndim == 2:
            if signal.shape[0] > signal.shape[1]:
                # Shape (T, C)
                return decimate(signal, factor, axis=0, zero_phase=True)
            else:
                # Shape (C, T)
                return decimate(signal, factor, axis=1, zero_phase=True)
        elif signal.ndim == 1:
            return decimate(signal, factor, zero_phase=True)
    raise ValueError(f"Unsupported decimation rate from {current_rate} Hz to 50 Hz")


def heal_pamap2_gaps(
    timestamps: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    max_gap_samples: int = 128,
    nominal_dt: float = 0.01  # 100 Hz = 0.01s
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    """
    Fills dropout gaps <= max_gap_samples by linear interpolation between valid samples.
    Gaps > max_gap_samples close the current segment and start a new one.
    Never interpolates across an activity label change.
    Returns:
        continuous_data_segments: list of (T_i, C) float32 arrays
        continuous_label_segments: list of (T_i,) int arrays
        audit_stats: dict of interpolated and discarded gap counts
    """
    n_samples = len(timestamps)
    if n_samples == 0:
        return [], [], {"interpolated_gaps": 0, "discarded_gaps": 0}

    stats = {"interpolated_gaps": 0, "discarded_gaps": 0}
    data_segments = []
    label_segments = []

    current_data = [data[0]]
    current_label = [labels[0]]

    for i in range(1, n_samples):
        dt = timestamps[i] - timestamps[i - 1]
        missing_samples = int(round(dt / nominal_dt)) - 1
        label_changed = (labels[i] != labels[i - 1])

        if missing_samples <= 0 and not label_changed:
            current_data.append(data[i])
            current_label.append(labels[i])
        elif 0 < missing_samples <= max_gap_samples and not label_changed:
            # Linear interpolation for gap
            stats["interpolated_gaps"] += 1
            start_val = data[i - 1]
            end_val = data[i]
            for step in range(1, missing_samples + 1):
                alpha = step / (missing_samples + 1)
                interp_val = (1.0 - alpha) * start_val + alpha * end_val
                current_data.append(interp_val)
                current_label.append(labels[i - 1])
            current_data.append(data[i])
            current_label.append(labels[i])
        else:
            # Discard gap or break across activity change -> close segment
            if missing_samples > max_gap_samples:
                stats["discarded_gaps"] += 1
            if len(current_data) > 0:
                data_segments.append(np.array(current_data, dtype=np.float32))
                label_segments.append(np.array(current_label, dtype=np.int64))
            current_data = [data[i]]
            current_label = [labels[i]]

    if len(current_data) > 0:
        data_segments.append(np.array(current_data, dtype=np.float32))
        label_segments.append(np.array(current_label, dtype=np.int64))

    return data_segments, label_segments, stats


def segment_stream(
    data: np.ndarray,
    labels: np.ndarray,
    min_length_samples: int = 256
) -> List[Tuple[np.ndarray, int]]:
    """
    Extracts maximal runs of consecutive samples carrying the same activity code.
    Segments shorter than min_length_samples are excluded.
    """
    if len(data) == 0:
        return []

    segments_list = []
    start_idx = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            seg_len = i - start_idx
            if seg_len >= min_length_samples:
                segments_list.append((data[start_idx:i], int(current_label)))
            start_idx = i
            current_label = labels[i]

    # Final segment
    seg_len = len(labels) - start_idx
    if seg_len >= min_length_samples:
        segments_list.append((data[start_idx:], int(current_label)))

    return segments_list


# Spec-compatible aliases
segment = segment_stream


def window_segment(
    segment_data: np.ndarray,
    label: int,
    subject_id: int,
    segment_id: int,
    window_length: int = 128,
    stride: int = 64
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Slices a continuous segment into windows strictly inside its boundaries.
    Input segment_data shape: (T, C)
    Output windows shape: (N, C, window_length) float32
    """
    T, C = segment_data.shape
    if T < window_length:
        return np.zeros((0, C, window_length), dtype=np.float32), []

    windows = []
    metadata = []
    window_idx = 0

    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        # (window_length, C) -> transpose to (C, window_length)
        win = segment_data[start:end, :].T.astype(np.float32)
        windows.append(win)

        metadata.append({
            "subject_id": int(subject_id),
            "activity_code": int(label),
            "segment_id": int(segment_id),
            "window_index": window_idx,
            "start_sample": start,
            "end_sample": end,
        })
        window_idx += 1

    return np.array(windows, dtype=np.float32), metadata


# Spec-compatible alias
window = window_segment
