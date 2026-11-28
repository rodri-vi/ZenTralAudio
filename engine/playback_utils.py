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


def splice_with_crossfade(base: np.ndarray, window: np.ndarray, start: int, fade: int) -> np.ndarray:
    """Copia ``base`` sustituyendo ``[start, start+len(window))`` por ``window``.

    Los bordes se funden linealmente durante ``fade`` muestras para no producir
    clics al pasar de la parte antigua a la nueva.
    """
    out = np.array(base, dtype=np.float32, copy=True)
    channels = min(out.shape[0], window.shape[0])
    length = min(window.shape[1], out.shape[1] - start)
    if length <= 0:
        return out
    end = start + length
    new = window[:channels, :length]
    fade = max(0, min(int(fade), length // 2))
    if fade == 0:
        out[:channels, start:end] = new
        return out
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:channels, start + fade : end - fade] = new[:, fade : length - fade]
    if start > 0:
        out[:channels, start : start + fade] = (
            out[:channels, start : start + fade] * (1.0 - ramp) + new[:, :fade] * ramp
        )
    else:
        out[:channels, :fade] = new[:, :fade]
    if end < out.shape[1]:
        out[:channels, end - fade : end] = (
            new[:, length - fade :] * (1.0 - ramp) + out[:channels, end - fade : end] * ramp
        )
    else:
        out[:channels, end - fade : end] = new[:, length - fade :]
    return out
