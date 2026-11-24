"""Tratamiento vocal: presencia, claridad/aire, control dinámico de sibilancia y saturación suave."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from engine.audio_types import AudioBuffer
from engine.dsp.equalizer import _highshelf_sos, _peaking_sos, apply_sos_safe


def apply_vocal_treatment(
    buffer: AudioBuffer,
    *,
    presence_db: float = 0.0,
    clarity_db: float = 0.0,
    deesser_amount: float = 0.0,
    saturation: float = 0.0,
) -> AudioBuffer:
    """Mejora inteligibilidad y presencia sin estridencia ni sibilancias agresivas."""
    sr = buffer.sample_rate
    x = buffer.samples.astype(np.float64, copy=True)

    # 1. Presencia (3.5 kHz) y claridad (9.5 kHz)
    sos_list: list[np.ndarray] = []
    if abs(presence_db) >= 0.05 and sr > 8000:
        sos_list.append(_peaking_sos(3500.0, presence_db, 1.0, sr))
    if abs(clarity_db) >= 0.05 and sr > 22000:
        sos_list.append(_highshelf_sos(9500.0, clarity_db, 0.707, sr))

    if sos_list:
        x = apply_sos_safe(x, np.vstack(sos_list), zero_phase=True).astype(np.float64)

    # 2. De-Esser split-band real (5 kHz - 9.5 kHz)
    deess = max(0.0, min(1.0, float(deesser_amount)))
    if deess > 0.01 and sr >= 22050 and x.shape[1] >= 64:
        nyq = sr * 0.5
        low_c = 5000.0 / nyq
        high_c = min(0.94, 9500.0 / nyq)
        if high_c > low_c + 0.05:
            bp_sos = butter(2, [low_c, high_c], btype="bandpass", output="sos")
            sibilance_band = sosfilt(bp_sos, x, axis=1)
            sib_env = np.max(np.abs(sibilance_band), axis=0)
            alpha = math.exp(-1.0 / max(1.0, sr * 0.005))
            smooth_env = lfilter([1.0 - alpha], [1.0, -alpha], sib_env)

            # Umbral adaptativo según intensidad de de-esser
            thresh = 0.035 * (1.15 - 0.85 * deess)
            excess = np.maximum(smooth_env - thresh, 0.0)
            sib_gain = 1.0 / (1.0 + (excess / max(1e-4, thresh)) * (2.5 * deess))
            reduction = (1.0 - sib_gain)[np.newaxis, :]
            x = x - sibilance_band * reduction

    # 3. Saturación armónica suave opcional
    sat = max(0.0, min(1.0, float(saturation)))
    if sat > 0.01:
        drive = 1.0 + sat * 2.5
        x = np.tanh(x * drive) / math.tanh(drive)

    return AudioBuffer(samples=x.astype(np.float32), sample_rate=sr)
