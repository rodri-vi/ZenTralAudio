"""Limitador de seguridad lookahead para evitar clipping digital en salida."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter1d

from engine.audio_types import AudioBuffer


def apply_safety_limiter(buffer: AudioBuffer, *, ceiling_db: float = -1.0) -> AudioBuffer:
    """Garantiza que ninguna muestra supere `ceiling_db` suavizando la ganancia sin cortes bruscos."""
    ceiling_lin = 10.0 ** (min(-0.1, float(ceiling_db)) / 20.0)
    x = buffer.samples
    if x.size == 0:
        return buffer.copy()

    peak_global = float(np.max(np.abs(x)))
    if peak_global <= ceiling_lin:
        return buffer.copy()

    sr = buffer.sample_rate
    lookahead = max(3, int(round(sr * 0.002)))
    link_env = np.max(np.abs(x), axis=0).astype(np.float64)
    held_env = maximum_filter1d(link_env, size=lookahead * 2 + 1, mode="nearest")

    desired_gain = np.where(held_env > ceiling_lin, ceiling_lin / np.maximum(held_env, 1e-8), 1.0)

    # Suavizado rápido de ataque y liberación moderada
    rel_alpha = math.exp(-1.0 / max(1.0, sr * 0.04))
    smooth_gain = np.empty_like(desired_gain)
    cur = 1.0
    for idx in range(desired_gain.shape[0]):
        g = float(desired_gain[idx])
        if g < cur:
            cur = 0.4 * cur + 0.6 * g
        else:
            cur = rel_alpha * cur + (1.0 - rel_alpha) * g
        smooth_gain[idx] = min(cur, g)

    limited = x * smooth_gain[np.newaxis, :].astype(np.float32)
    # Resguardo final absoluto ante cualquier mínimo sobrepaso numérico
    np.clip(limited, -ceiling_lin, ceiling_lin, out=limited)
    return AudioBuffer(samples=limited, sample_rate=sr)
