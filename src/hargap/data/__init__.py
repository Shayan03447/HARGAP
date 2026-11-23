"""Data ingestion, transformation, windowing, and cache management."""

from hargap.data.transform import (
    resample_50hz,
    heal_pamap2_gaps,
    segment_stream,
    window_segment,
    segment,
    window,
)
from hargap.data.loaders import (
    load_ucihar,
    load_pamap2,
    load_mhealth,
    generate_synthetic_benchmark,
)
from hargap.data.cache import WindowStore, build_cache

__all__ = [
    "resample_50hz",
    "heal_pamap2_gaps",
    "segment_stream",
    "window_segment",
    "segment",
    "window",
    "load_ucihar",
    "load_pamap2",
    "load_mhealth",
    "generate_synthetic_benchmark",
    "WindowStore",
    "build_cache",
]
