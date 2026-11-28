"""Reducción de ruido espectral mediante STFT con seguimiento de suelo de ruido y suavizado."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from engine.audio_types import AudioBuffer
from engine.dsp.stft import (
    frame_count,
    inverse_chunk,
    iter_spectra,
    overlap_add_chunk,
    overlap_add_norm,
    pad_for_frames,
)


def apply_spectral_denoise(
    buffer: AudioBuffer,
    *,
    amount: float = 0.5,
    smoothing: float = 0.4,
    noise_percentile: float = 12.0,
) -> AudioBuffer:
    """Atenúa ruido estacionario/cuasi-estacionario preservando transitorios vocales.

    Se procesa por bloques de tramas para acotar la memoria en grabaciones largas:
    1) energía por trama, 2) perfil de ruido con las tramas más silenciosas y
    3) aplicación de la ganancia tipo Wiener con suavizado temporal continuo.
    """
    amt = max(0.0, min(1.0, float(amount)))
    if amt <= 0.01 or buffer.num_samples < 128:
        return buffer.copy()

    smooth = max(0.0, min(0.95, float(smoothing)))
    n_fft = 1024
    hop = 256
    win = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    win_sq = win * win

    out_channels = np.zeros_like(buffer.samples, dtype=np.float32)
    min_floor = max(0.03, 1.0 - 0.94 * amt)
    alpha = 0.8 + 2.2 * amt
    percentile = max(5.0, min(40.0, noise_percentile))

    for ch in range(buffer.channels):
        x = buffer.samples[ch]
        n = x.shape[0]
        x_pad, lead = pad_for_frames(x, n_fft, hop)
        total = x_pad.shape[0]
        frames = frame_count(total, n_fft, hop)

        # 1. Energía media por trama.
        frame_energy = np.empty(frames, dtype=np.float64)
        for first, spec in iter_spectra(x_pad, win, hop):
            mag = np.abs(spec)
            frame_energy[first : first + spec.shape[0]] = np.mean(
                np.square(mag, dtype=np.float64), axis=1
            )

        # 2. Perfil de ruido a partir de las tramas de menor energía.
        thresh = np.percentile(frame_energy, percentile)
        quiet_mask = frame_energy <= thresh
        if np.any(quiet_mask):
            noise_sum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
            for first, spec in iter_spectra(x_pad, win, hop):
                local = quiet_mask[first : first + spec.shape[0]]
                if np.any(local):
                    noise_sum += np.abs(spec[local]).sum(axis=0, dtype=np.float64)
            noise_profile = (noise_sum / float(np.count_nonzero(quiet_mask)))[np.newaxis, :]
        else:
            mags = np.concatenate([np.abs(spec) for _, spec in iter_spectra(x_pad, win, hop)])
            noise_profile = np.percentile(mags, noise_percentile, axis=0, keepdims=True)
        noise_profile = (alpha * noise_profile).astype(np.float32)

        # 3. Ganancia espectral, suavizado temporal y reconstrucción overlap-add.
        use_smoothing = smooth > 0.01 and frames > 1
        smoothing_state: np.ndarray | None = None
        y = np.zeros(total, dtype=np.float32)
        for first, spec in iter_spectra(x_pad, win, hop):
            mag = np.abs(spec)
            gains = np.maximum(mag - noise_profile, 0.0) / np.maximum(mag, 1e-8)
            np.maximum(gains, min_floor, out=gains)
            if use_smoothing:
                if smoothing_state is None:
                    smoothing_state = (smooth * gains[0])[np.newaxis, :]
                gains, smoothing_state = lfilter(
                    [1.0 - smooth], [1.0, -smooth], gains, axis=0, zi=smoothing_state
                )
                gains = gains.astype(np.float32, copy=False)
            recon = inverse_chunk(spec * gains, n_fft)
            recon *= win
            overlap_add_chunk(y, recon, first, hop)

        y /= np.maximum(overlap_add_norm(total, win_sq, hop, frames), 1e-6)
        out_channels[ch] = y[lead : lead + n]

    return AudioBuffer(samples=out_channels, sample_rate=buffer.sample_rate)
