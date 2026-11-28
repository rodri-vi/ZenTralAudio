"""Helpers for preparing playback gain outside real-time audio callbacks."""

from __future__ import annotations

import numpy as np


def calculate_loudness_compensation(
    original: np.ndarray,
    source: np.ndarray,
    selection_frames: tuple[int, int] | None = None,
) -> float:
    """Return the clamped RMS-matching gain for a buffer or selected fragment."""
    if original is source:
        return 1.0
    if original.ndim != 2 or source.ndim != 2:
        return 1.0

    channel_count = min(original.shape[0], source.shape[0])
    frame_count = min(original.shape[1], source.shape[1])
    if channel_count <= 0 or frame_count <= 0:
        return 1.0

    if selection_frames is None:
        start, end = 0, frame_count
    else:
        start, end = selection_frames
        start = max(0, min(frame_count, int(start)))
        end = max(start, min(frame_count, int(end)))

    if end <= start:
        return 1.0

    original_slice = original[:channel_count, start:end]
    source_slice = source[:channel_count, start:end]
    rms_original = float(np.sqrt(np.mean(np.square(original_slice), dtype=np.float64)) + 1e-6)
    rms_source = float(np.sqrt(np.mean(np.square(source_slice), dtype=np.float64)) + 1e-6)
    return float(max(0.35, min(2.0, rms_original / rms_source)))
