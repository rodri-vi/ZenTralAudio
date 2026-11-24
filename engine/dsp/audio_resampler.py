"""Conversión controlada de frecuencia de muestreo preservando alineación temporal."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

from engine.audio_types import AudioBuffer


def resample_audio(
    buffer: AudioBuffer,
    target_sample_rate: int,
    *,
    target_length: int | None = None,
) -> AudioBuffer:
    """Convierte la frecuencia de muestreo usando soxr (HQ) o resample_poly como respaldo."""
    src_sr = buffer.sample_rate
    dst_sr = int(target_sample_rate)
    if dst_sr <= 0:
        raise ValueError(f"Frecuencia destino inválida: {dst_sr}")

    if src_sr == dst_sr:
        out = buffer.samples.copy()
    else:
        try:
            import soxr

            resampled = soxr.resample(buffer.samples.T, src_sr, dst_sr, quality="HQ")
            out = np.asarray(resampled.T, dtype=np.float32)
        except ImportError:
            gcd = math.gcd(src_sr, dst_sr)
            up = dst_sr // gcd
            down = src_sr // gcd
            out = np.asarray(resample_poly(buffer.samples, up=up, down=down, axis=1), dtype=np.float32)

    if out.ndim == 1:
        out = out[np.newaxis, :]

    if target_length is not None and target_length > 0:
        cur_len = out.shape[1]
        if cur_len > target_length:
            out = out[:, :target_length]
        elif cur_len < target_length:
            out = np.pad(out, ((0, 0), (0, target_length - cur_len)), mode="constant")

    return AudioBuffer(samples=out, sample_rate=dst_sr)
