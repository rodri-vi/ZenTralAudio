"""Ecualización real con secciones de segundo orden (SOS) y fase cero controlada."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from engine.audio_types import AudioBuffer


def _peaking_sos(freq: float, gain_db: float, q: float, sample_rate: int) -> np.ndarray:
    """Calcula coeficientes biquad (RBJ) de filtro paramétrico peaking en formato SOS."""
    nyquist = sample_rate * 0.5
    f0 = max(20.0, min(nyquist * 0.92, float(freq)))
    a = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * f0 / float(sample_rate)
    alpha = math.sin(w0) / (2.0 * max(0.2, float(q)))
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _highshelf_sos(freq: float, gain_db: float, q: float, sample_rate: int) -> np.ndarray:
    """Calcula filtro shelf de agudos en formato SOS."""
    nyquist = sample_rate * 0.5
    f0 = max(200.0, min(nyquist * 0.92, float(freq)))
    a = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * f0 / float(sample_rate)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * max(0.3, float(q)))
    sqrt_a_2alpha = 2.0 * math.sqrt(a) * alpha

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_2alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_2alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_2alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_2alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def apply_sos_safe(samples: np.ndarray, sos: np.ndarray, *, zero_phase: bool = True) -> np.ndarray:
    """Aplica filtro SOS evitando errores de padding en señales muy cortas."""
    n = samples.shape[1]
    min_padlen = 3 * (2 * sos.shape[0] + 1)
    if zero_phase and n > min_padlen + 4:
        return np.asarray(sosfiltfilt(sos, samples, axis=1), dtype=np.float32)
    return np.asarray(sosfilt(sos, samples, axis=1), dtype=np.float32)


def apply_equalizer(
    buffer: AudioBuffer,
    *,
    hp_freq: float = 80.0,
    lp_freq: float = 18000.0,
    mid_gain_db: float = 0.0,
    mid_freq: float = 2500.0,
    mid_q: float = 1.1,
    extra_bands: list[dict[str, float | bool]] | None = None,
) -> AudioBuffer:
    """Aplica paso alto, paso bajo y bandas paramétricas activas."""
    sr = buffer.sample_rate
    nyquist = sr * 0.5
    x = buffer.samples.astype(np.float64, copy=True)

    sos_sections: list[np.ndarray] = []

    if hp_freq > 22.0 and hp_freq < nyquist * 0.85:
        sos_sections.append(butter(2, hp_freq / nyquist, btype="highpass", output="sos"))

    if lp_freq < nyquist * 0.96 and lp_freq > max(150.0, hp_freq * 1.5):
        sos_sections.append(butter(2, lp_freq / nyquist, btype="lowpass", output="sos"))

    if abs(mid_gain_db) >= 0.05:
        sos_sections.append(_peaking_sos(mid_freq, mid_gain_db, mid_q, sr))

    if extra_bands:
        for band in extra_bands:
            if not band.get("enabled", True):
                continue
            gain = float(band.get("gain_db", 0.0))
            if abs(gain) >= 0.05:
                sos_sections.append(
                    _peaking_sos(
                        float(band.get("freq", 1000.0)),
                        gain,
                        float(band.get("q", 1.0)),
                        sr,
                    )
                )

    if not sos_sections:
        return buffer.copy()

    combined_sos = np.vstack(sos_sections)
    filtered = apply_sos_safe(x, combined_sos, zero_phase=True)
    return AudioBuffer(samples=filtered, sample_rate=sr)
