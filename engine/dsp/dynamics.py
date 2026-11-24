"""Compresor dinámico vectorizado por bloques con soft-knee, attack/release y makeup."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from engine.audio_types import AudioBuffer


def apply_compressor(
    buffer: AudioBuffer,
    *,
    enabled: bool = True,
    threshold_db: float = -18.0,
    ratio: float = 2.5,
    attack_ms: float = 12.0,
    release_ms: float = 140.0,
    knee_db: float = 4.0,
    makeup_gain_db: float = 0.0,
) -> AudioBuffer:
    """Comprime picos vocales controlando transitorios y suavizando recuperación."""
    if not enabled or ratio <= 1.01 or buffer.num_samples < 32:
        return buffer.copy()

    sr = buffer.sample_rate
    x = buffer.samples
    # Detección enlazada estéreo (máximo abs entre canales) en bloques de 1 ms para alta velocidad
    hop = max(1, int(round(sr * 0.001)))
    n = x.shape[1]
    num_blocks = math.ceil(n / hop)
    padded_len = num_blocks * hop

    mono_env = np.max(np.abs(x), axis=0)
    if padded_len > n:
        mono_env = np.pad(mono_env, (0, padded_len - n), mode="edge")
    block_peak = mono_env.reshape(num_blocks, hop).max(axis=1)
    env_db = 20.0 * np.log10(np.maximum(block_peak, 1e-6))

    # Curva de ganancia estática con soft-knee
    half_knee = max(0.0, float(knee_db)) * 0.5
    slope = 1.0 - (1.0 / max(1.01, float(ratio)))
    overshoot = env_db - float(threshold_db)

    gain_red_db = np.zeros_like(env_db, dtype=np.float64)
    if half_knee > 0.01:
        in_knee = (overshoot > -half_knee) & (overshoot < half_knee)
        above = overshoot >= half_knee
        gain_red_db[in_knee] = -slope * ((overshoot[in_knee] + half_knee) ** 2) / (4.0 * half_knee)
        gain_red_db[above] = -slope * overshoot[above]
    else:
        above = overshoot > 0.0
        gain_red_db[above] = -slope * overshoot[above]

    # Suavizado asimétrico attack/release sobre bloques de 1 ms
    block_sr = sr / float(hop)
    att_coeff = math.exp(-1.0 / max(1e-4, (attack_ms / 1000.0) * block_sr))
    rel_coeff = math.exp(-1.0 / max(1e-4, (release_ms / 1000.0) * block_sr))

    smooth_gr_db = np.empty_like(gain_red_db)
    cur = 0.0
    for idx in range(num_blocks):
        target = float(gain_red_db[idx])
        coeff = att_coeff if target < cur else rel_coeff
        cur = coeff * cur + (1.0 - coeff) * target
        smooth_gr_db[idx] = cur

    # Interpolación suave al número exacto de muestras
    block_centers = (np.arange(num_blocks) + 0.5) * hop
    sample_idx = np.arange(n, dtype=np.float64)
    sample_gr_db = np.interp(sample_idx, block_centers, smooth_gr_db)
    linear_gain = (10.0 ** ((sample_gr_db + float(makeup_gain_db)) / 20.0)).astype(np.float32)

    out = x * linear_gain[np.newaxis, :]
    return AudioBuffer(samples=out, sample_rate=sr)


def apply_noise_gate(
    buffer: AudioBuffer,
    *,
    enabled: bool = False,
    threshold_db: float = -48.0,
    reduction_db: float = 18.0,
) -> AudioBuffer:
    """Expansor descendente suave para reducir ruido residual en pausas entre frases."""
    if not enabled or buffer.num_samples < 64:
        return buffer.copy()

    sr = buffer.sample_rate
    env = np.max(np.abs(buffer.samples), axis=0).astype(np.float64)
    alpha = math.exp(-1.0 / max(1.0, sr * 0.015))
    smoothed = lfilter([1.0 - alpha], [1.0, -alpha], env)
    db = 20.0 * np.log10(np.maximum(smoothed, 1e-6))

    below = db < float(threshold_db)
    att = np.zeros_like(db)
    att[below] = -min(float(reduction_db), (float(threshold_db) - db[below]) * 1.5)
    gain = (10.0 ** (att / 20.0)).astype(np.float32)
    return AudioBuffer(samples=buffer.samples * gain[np.newaxis, :], sample_rate=sr)
