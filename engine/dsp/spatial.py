"""Tratamiento espacial: dereverb DSP por envolvente, anchura M/S y balance."""

from __future__ import annotations

import numpy as np

from engine.audio_mixer import apply_pan
from engine.audio_types import AudioBuffer


def apply_dsp_dereverb(buffer: AudioBuffer, *, amount: float = 0.0) -> AudioBuffer:
    """Atenúa colas reverberantes tardías en frecuencias medias-altas sin artefactos bruscos."""
    amt = max(0.0, min(1.0, float(amount)))
    if amt <= 0.01 or buffer.num_samples < 256:
        return buffer.copy()

    n_fft = 1024
    hop = 256
    win = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    out = np.zeros_like(buffer.samples, dtype=np.float32)
    decay_alpha = 0.72 + 0.16 * amt

    for ch in range(buffer.channels):
        x = buffer.samples[ch]
        n = x.shape[0]
        pad_tail = (hop - ((n - n_fft) % hop)) % hop if n >= n_fft else (n_fft - n)
        x_pad = np.pad(x, (0, pad_tail), mode="constant")
        total = x_pad.shape[0]
        starts = range(0, total - n_fft + 1, hop)

        frames = np.stack([x_pad[s : s + n_fft] * win for s in starts], axis=0)
        spec = np.fft.rfft(frames, axis=1)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # Estima energía reverberante retrasada por banda y la atenúa proporcionalmente
        reverb_est = np.zeros_like(mag)
        for t in range(1, mag.shape[0]):
            reverb_est[t] = decay_alpha * reverb_est[t - 1] + (1.0 - decay_alpha) * mag[t - 1]

        clean_mag = np.maximum(mag - (0.65 * amt) * reverb_est, mag * (1.0 - 0.75 * amt))
        recon = np.fft.irfft(clean_mag * np.exp(1j * phase), n=n_fft, axis=1).astype(np.float32)

        y = np.zeros(total, dtype=np.float32)
        norm = np.zeros(total, dtype=np.float32)
        win2 = win * win
        for idx, s in enumerate(starts):
            y[s : s + n_fft] += recon[idx] * win
            norm[s : s + n_fft] += win2
        y /= np.maximum(norm, 1e-6)
        out[ch] = y[:n]

    return AudioBuffer(samples=out, sample_rate=buffer.sample_rate)


def apply_stereo_width(buffer: AudioBuffer, *, width: float = 1.0) -> AudioBuffer:
    """Ajusta anchura estéreo mediante matriz Mid/Side (0=mono, 1=original, >1=amplio)."""
    w = max(0.0, min(2.0, float(width)))
    if buffer.channels < 2 or abs(w - 1.0) < 1e-4:
        return buffer.copy()

    left = buffer.samples[0]
    right = buffer.samples[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right) * w
    out = np.stack([mid + side, mid - side], axis=0).astype(np.float32)
    return AudioBuffer(samples=out, sample_rate=buffer.sample_rate)


def apply_spatial_stage(
    buffer: AudioBuffer,
    *,
    dereverb_amount: float = 0.0,
    stereo_width: float = 1.0,
    pan: float = 0.0,
) -> AudioBuffer:
    """Ejecuta las operaciones espaciales seleccionadas en orden seguro."""
    current = apply_dsp_dereverb(buffer, amount=dereverb_amount)
    if current.channels >= 2:
        current = apply_stereo_width(current, width=stereo_width)
    if abs(pan) > 1e-4:
        current = apply_pan(current, pan)
    return current
