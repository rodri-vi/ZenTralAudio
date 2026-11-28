"""Tratamiento espacial: dereverb DSP por envolvente, anchura M/S y balance."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from engine.audio_mixer import apply_pan
from engine.audio_types import AudioBuffer
from engine.dsp.stft import (
    frame_count,
    inverse_chunk,
    iter_spectra,
    overlap_add_chunk,
    overlap_add_norm,
    pad_for_frames,
)


def apply_dsp_dereverb(buffer: AudioBuffer, *, amount: float = 0.0) -> AudioBuffer:
    """Atenúa colas reverberantes tardías en frecuencias medias-altas sin artefactos bruscos."""
    amt = max(0.0, min(1.0, float(amount)))
    if amt <= 0.01 or buffer.num_samples < 256:
        return buffer.copy()

    n_fft = 1024
    hop = 256
    win = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    win_sq = win * win
    out = np.zeros_like(buffer.samples, dtype=np.float32)
    decay_alpha = 0.72 + 0.16 * amt
    sub_amount = 0.65 * amt
    floor_ratio = 1.0 - 0.75 * amt
    bins = n_fft // 2 + 1

    for ch in range(buffer.channels):
        x = buffer.samples[ch]
        n = x.shape[0]
        x_pad, lead = pad_for_frames(x, n_fft, hop)
        total = x_pad.shape[0]
        frames = frame_count(total, n_fft, hop)

        # Estima energía reverberante retrasada por banda (filtro recursivo sobre la
        # magnitud de la trama anterior) y la atenúa proporcionalmente.
        previous_mag = np.zeros((1, bins), dtype=np.float32)
        state = np.zeros((1, bins), dtype=np.float64)
        y = np.zeros(total, dtype=np.float32)
        for first, spec in iter_spectra(x_pad, win, hop):
            mag = np.abs(spec)
            delayed = np.concatenate([previous_mag, mag[:-1]], axis=0)
            previous_mag = mag[-1:]
            reverb_est, state = lfilter(
                [1.0 - decay_alpha], [1.0, -decay_alpha], delayed, axis=0, zi=state
            )
            clean_mag = np.maximum(mag - sub_amount * reverb_est, mag * floor_ratio)
            ratio = (clean_mag / np.maximum(mag, 1e-30)).astype(np.float32)
            ratio[mag <= 0.0] = 0.0
            recon = inverse_chunk(spec * ratio, n_fft)
            recon *= win
            overlap_add_chunk(y, recon, first, hop)

        y /= np.maximum(overlap_add_norm(total, win_sq, hop, frames), 1e-6)
        out[ch] = y[lead : lead + n]

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
