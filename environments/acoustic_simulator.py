"""Simulacion acustica mediante convolucion IR real y sintesis geometrica algoritmica."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from engine.audio_mixer import apply_pan, dry_wet_mix
from engine.audio_types import AudioBuffer
from engine.dsp.spatial import apply_stereo_width


def _apply_air_damping(samples: np.ndarray, sample_rate: int, damping_pct: float) -> np.ndarray:
    """Atenúa progresivamente agudos en la cola reflejada según el parámetro damping (0..100)."""
    d = max(0.0, min(100.0, float(damping_pct))) / 100.0
    if d <= 0.02 or samples.shape[1] < 16:
        return samples
    cutoff = max(1200.0, (sample_rate * 0.45) * (1.0 - 0.82 * d))
    nyq = sample_rate * 0.5
    if cutoff >= nyq * 0.94:
        return samples
    sos = butter(2, cutoff / nyq, btype="lowpass", output="sos")
    return np.asarray(sosfilt(sos, samples, axis=1), dtype=np.float32)


def _apply_predelay(samples: np.ndarray, sample_rate: int, predelay_ms: float) -> np.ndarray:
    delay_frames = max(0, int(round((max(0.0, float(predelay_ms)) / 1000.0) * sample_rate)))
    if delay_frames == 0:
        return samples
    n = samples.shape[1]
    if delay_frames >= n:
        return np.zeros_like(samples, dtype=np.float32)
    out = np.zeros_like(samples, dtype=np.float32)
    out[:, delay_frames:] = samples[:, : n - delay_frames]
    return out


def build_algorithmic_room_ir(
    sample_rate: int,
    *,
    room_size: float = 45.0,
    decay_s: float = 1.2,
    damping: float = 45.0,
) -> AudioBuffer:
    """Genera una respuesta acústica determinista de sala (reflexiones tempranas + cola difusa).

    No pretende sustituir una IR real medida, sino ofrecer una simulación geométrica
    paramétrica real y reproducible cuando no se ha cargado un archivo IR externo.
    """
    rt60 = max(0.10, min(6.0, float(decay_s)))
    size_norm = max(0.05, min(1.0, float(room_size) / 100.0))
    ir_len = max(256, min(sample_rate * 5, int(round(rt60 * sample_rate * 1.15))))
    ir = np.zeros((2, ir_len), dtype=np.float32)

    # 1. Reflexiones tempranas con tiempos primos escalados por tamaño aparente del espacio
    early_base_ms = np.array([7.3, 11.7, 17.9, 23.3, 31.1, 41.9, 53.7, 67.3], dtype=np.float64)
    early_times_ms = early_base_ms * (0.4 + 1.1 * size_norm)
    for idx, t_ms in enumerate(early_times_ms):
        pos = int(round((t_ms / 1000.0) * sample_rate))
        if 0 < pos < ir_len:
            amp = 0.55 * (0.78 ** idx)
            # Decorrelación izquierda/derecha realista
            ir[0, pos] += amp * (1.0 if idx % 2 == 0 else 0.65)
            ir[1, pos] += amp * (0.65 if idx % 2 == 0 else 1.0)

    # 2. Cola difusa densa determinista (suma de pulsos escalonados + envolvente RT60)
    rng = np.random.default_rng(seed=int(round(size_norm * 1000 + rt60 * 100)))
    t_sec = np.arange(ir_len, dtype=np.float64) / float(sample_rate)
    decay_env = np.exp(-6.907755 * t_sec / rt60)
    tail_onset = int(round((early_times_ms[1] / 1000.0) * sample_rate))
    tail = rng.standard_normal((2, ir_len)).astype(np.float64) * decay_env[np.newaxis, :] * 0.25
    if tail_onset > 0:
        ramp = np.minimum(1.0, np.arange(ir_len, dtype=np.float64) / float(max(1, tail_onset)))
        tail *= ramp[np.newaxis, :]

    ir_raw = (ir.astype(np.float64) + tail).astype(np.float32)
    ir_damped = _apply_air_damping(ir_raw, sample_rate, damping)

    # Normalizar energía total para mantener coherencia de nivel
    energy = float(np.sqrt(np.mean(np.sum(ir_damped ** 2, axis=1))))
    if energy > 1e-6:
        ir_damped /= energy
    return AudioBuffer(samples=ir_damped, sample_rate=sample_rate)


def simulate_acoustic_space(
    voice: AudioBuffer,
    *,
    ir_buffer: AudioBuffer | None,
    acoustics: dict[str, float],
) -> AudioBuffer:
    """Aplica convolución IR o simulación algorítmica más distancia aparente, anchura y pan."""
    sr = voice.sample_rate
    room_size = float(acoustics.get("room_size", 45.0))
    reverb_mix = max(0.0, min(1.0, float(acoustics.get("reverb_mix", 28.0)) / 100.0))
    decay_s = float(acoustics.get("decay_s", 1.2))
    predelay_ms = float(acoustics.get("predelay_ms", 14.0))
    damping = float(acoustics.get("damping", 45.0))
    direct_level = max(0.0, min(1.5, float(acoustics.get("direct_level", 100.0)) / 100.0))
    reflected_level = max(0.0, min(1.5, float(acoustics.get("reflected_level", 40.0)) / 100.0))
    stereo_width = max(0.0, min(1.5, float(acoustics.get("stereo_width", 100.0)) / 100.0))
    distance = max(0.0, min(1.0, float(acoustics.get("apparent_distance", 20.0)) / 100.0))
    pan = max(-1.0, min(1.0, float(acoustics.get("pan", 0.0)) / 100.0))

    # Convertir voz a estéreo para imagen espacial consistente
    voice_st = voice.to_stereo()
    n = voice_st.num_samples

    # Simulación de absorción aérea en voz directa según distancia aparente
    direct_sig = voice_st.samples.copy()
    if distance > 0.05:
        direct_sig = _apply_air_damping(direct_sig, sr, distance * 65.0)

    # Nivel efectivo directo vs reflejado ajustado por distancia aparente
    eff_direct = direct_level * (1.0 - 0.45 * distance)
    eff_reflected = reflected_level * (1.0 + 0.65 * distance)

    if reverb_mix <= 0.005 or eff_reflected <= 0.005:
        out = AudioBuffer(samples=direct_sig * eff_direct, sample_rate=sr)
        out = apply_stereo_width(out, width=stereo_width)
        return apply_pan(out, pan) if abs(pan) > 1e-4 else out

    if ir_buffer is not None:
        ir = ir_buffer.to_stereo()
        ir_samples = _apply_air_damping(ir.samples, sr, damping * 0.6)
    else:
        ir = build_algorithmic_room_ir(sr, room_size=room_size, decay_s=decay_s, damping=damping)
        ir_samples = ir.samples

    wet = np.zeros_like(voice_st.samples, dtype=np.float32)
    for ch in range(2):
        conv = fftconvolve(voice_st.samples[ch], ir_samples[ch], mode="full")[:n]
        wet[ch] = conv.astype(np.float32)

    wet = _apply_predelay(wet, sr, predelay_ms)
    wet_buf = AudioBuffer(samples=wet * eff_reflected, sample_rate=sr)
    dry_buf = AudioBuffer(samples=direct_sig * eff_direct, sample_rate=sr)

    mixed = dry_wet_mix(dry_buf, wet_buf, reverb_mix)
    # Compensación suave para preservar volumen global equilibrado
    mixed = AudioBuffer(samples=mixed.samples / max(0.8, math.sqrt(1.0 - reverb_mix * 0.25)), sample_rate=sr)
    mixed = apply_stereo_width(mixed, width=stereo_width)
    if abs(pan) > 1e-4:
        mixed = apply_pan(mixed, pan)
    return mixed
