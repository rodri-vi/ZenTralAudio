"""Reducción de ruido espectral mediante STFT con seguimiento de suelo de ruido y suavizado."""

from __future__ import annotations

import numpy as np

from engine.audio_types import AudioBuffer


def apply_spectral_denoise(
    buffer: AudioBuffer,
    *,
    amount: float = 0.5,
    smoothing: float = 0.4,
    noise_percentile: float = 12.0,
) -> AudioBuffer:
    """Atenúa ruido estacionario/cuasi-estacionario preservando transitorios vocales."""
    amt = max(0.0, min(1.0, float(amount)))
    if amt <= 0.01 or buffer.num_samples < 128:
        return buffer.copy()

    smooth = max(0.0, min(0.95, float(smoothing)))
    n_fft = 1024
    hop = 256
    win = np.sqrt(np.hanning(n_fft)).astype(np.float32)

    out_channels = np.zeros_like(buffer.samples, dtype=np.float32)
    min_floor = max(0.03, 1.0 - 0.94 * amt)

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

        # Estimación robusta del espectro de ruido a partir de tramas de menor energía
        frame_energy = np.mean(mag ** 2, axis=1)
        thresh = np.percentile(frame_energy, max(5.0, min(40.0, noise_percentile)))
        quiet_mask = frame_energy <= thresh
        if np.any(quiet_mask):
            noise_profile = np.mean(mag[quiet_mask], axis=0, keepdims=True)
        else:
            noise_profile = np.percentile(mag, noise_percentile, axis=0, keepdims=True)

        # Regla espectral tipo Wiener con sobre-sustracción moderada
        alpha = 0.8 + 2.2 * amt
        snr_like = np.maximum(mag - alpha * noise_profile, 0.0) / np.maximum(mag, 1e-8)
        gains = np.maximum(snr_like, min_floor)

        # Suavizado temporal de ganancias para reducir ruido musical
        if smooth > 0.01 and gains.shape[0] > 1:
            smoothed = gains.copy()
            for t in range(1, gains.shape[0]):
                smoothed[t] = smooth * smoothed[t - 1] + (1.0 - smooth) * gains[t]
            gains = smoothed

        enhanced_spec = (mag * gains) * np.exp(1j * phase)
        recon_frames = np.fft.irfft(enhanced_spec, n=n_fft, axis=1).astype(np.float32)

        y = np.zeros(total, dtype=np.float32)
        norm = np.zeros(total, dtype=np.float32)
        win2 = win * win
        for idx, s in enumerate(starts):
            y[s : s + n_fft] += recon_frames[idx] * win
            norm[s : s + n_fft] += win2

        y /= np.maximum(norm, 1e-6)
        out_channels[ch] = y[:n]

    return AudioBuffer(samples=out_channels, sample_rate=buffer.sample_rate)
