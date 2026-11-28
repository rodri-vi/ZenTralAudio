"""Regresiones de las optimizaciones DSP/acústicas frente a implementaciones de referencia.

Las referencias reproducen los algoritmos originales (bucles Python, FFT global y
``dry_wet_mix``) para comprobar que la versión optimizada conserva el resultado.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import fftconvolve

from engine.audio_mixer import apply_pan, dry_wet_mix
from engine.audio_types import AudioBuffer
from engine.dsp import kernels
from engine.dsp.dynamics import apply_compressor
from engine.dsp.safety_limiter import apply_safety_limiter
from engine.dsp.spatial import apply_dsp_dereverb, apply_stereo_width
from engine.dsp.spectral_denoise import apply_spectral_denoise
from environments import acoustic_simulator
from environments.acoustic_simulator import (
    _apply_air_damping,
    _apply_predelay,
    build_algorithmic_room_ir,
    clear_acoustic_caches,
    simulate_acoustic_space,
)


def _voice(seconds: float = 2.0, sr: int = 48000, channels: int = 2, seed: int = 7) -> AudioBuffer:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * t) ** 2
    base = 0.25 * env * np.sin(2 * np.pi * 220 * t) + 0.08 * np.sin(2 * np.pi * 1870 * t)
    data = np.stack([base * (1.0 - 0.2 * c) for c in range(channels)])
    data += 0.01 * rng.standard_normal(data.shape)
    return AudioBuffer(samples=data.astype(np.float32), sample_rate=sr)


def _reference_simulate(voice: AudioBuffer, ir_buffer: AudioBuffer | None, acoustics: dict) -> np.ndarray:
    """Algoritmo original de simulate_acoustic_space (FFT global + dry_wet_mix)."""
    sr = voice.sample_rate
    reverb_mix = max(0.0, min(1.0, float(acoustics.get("reverb_mix", 28.0)) / 100.0))
    decay_s = float(acoustics.get("decay_s", 1.2))
    damping = float(acoustics.get("damping", 45.0))
    direct_level = max(0.0, min(1.5, float(acoustics.get("direct_level", 100.0)) / 100.0))
    reflected_level = max(0.0, min(1.5, float(acoustics.get("reflected_level", 40.0)) / 100.0))
    stereo_width = max(0.0, min(1.5, float(acoustics.get("stereo_width", 100.0)) / 100.0))
    distance = max(0.0, min(1.0, float(acoustics.get("apparent_distance", 20.0)) / 100.0))
    pan = max(-1.0, min(1.0, float(acoustics.get("pan", 0.0)) / 100.0))
    voice_st = voice.to_stereo()
    n = voice_st.num_samples
    direct = voice_st.samples.copy()
    if distance > 0.05:
        direct = _apply_air_damping(direct, sr, distance * 65.0)
    eff_direct = direct_level * (1.0 - 0.45 * distance)
    eff_reflected = reflected_level * (1.0 + 0.65 * distance)
    if ir_buffer is not None:
        ir = _apply_air_damping(ir_buffer.to_stereo().samples, sr, damping * 0.6)
    else:
        ir = build_algorithmic_room_ir(
            sr,
            room_size=float(acoustics.get("room_size", 45.0)),
            decay_s=decay_s,
            damping=damping,
        ).samples
    wet = np.stack([fftconvolve(voice_st.samples[c], ir[c])[:n] for c in range(2)]).astype(np.float32)
    wet = _apply_predelay(wet, sr, float(acoustics.get("predelay_ms", 14.0)))
    mixed = dry_wet_mix(
        AudioBuffer(samples=direct * eff_direct, sample_rate=sr),
        AudioBuffer(samples=wet * eff_reflected, sample_rate=sr),
        reverb_mix,
    )
    mixed = AudioBuffer(
        samples=mixed.samples / max(0.8, math.sqrt(1.0 - reverb_mix * 0.25)), sample_rate=sr
    )
    mixed = apply_stereo_width(mixed, width=stereo_width)
    if abs(pan) > 1e-4:
        mixed = apply_pan(mixed, pan)
    return mixed.samples


ACOUSTIC_CASES = [
    {"reverb_mix": 62.0, "decay_s": 1.8, "predelay_ms": 38.0, "damping": 38.0, "room_size": 95.0,
     "direct_level": 82.0, "reflected_level": 85.0, "stereo_width": 130.0, "apparent_distance": 45.0, "pan": 0.0},
    {"reverb_mix": 100.0, "decay_s": 0.6, "predelay_ms": 0.0, "damping": 10.0,
     "direct_level": 100.0, "reflected_level": 40.0, "stereo_width": 100.0, "apparent_distance": 0.0, "pan": -35.0},
    {"reverb_mix": 20.0, "decay_s": 3.0, "predelay_ms": 5000.0, "damping": 70.0,
     "direct_level": 120.0, "reflected_level": 60.0, "stereo_width": 0.0, "apparent_distance": 90.0, "pan": 60.0},
]


@pytest.mark.unit
@pytest.mark.parametrize("acoustics", ACOUSTIC_CASES)
@pytest.mark.parametrize("channels", [1, 2])
def test_simulate_acoustic_space_matches_reference(acoustics: dict, channels: int) -> None:
    clear_acoustic_caches()
    voice = _voice(seconds=1.5, channels=channels)
    expected = _reference_simulate(voice, None, acoustics)
    out = simulate_acoustic_space(voice, ir_buffer=None, acoustics=acoustics)
    assert out.samples.shape == expected.shape == (2, voice.num_samples)
    assert np.isfinite(out.samples).all()
    np.testing.assert_allclose(out.samples, expected, rtol=1e-4, atol=2e-5)
    corr = np.corrcoef(out.samples.ravel(), expected.ravel())[0, 1]
    assert corr > 0.99999


@pytest.mark.unit
def test_simulate_with_loaded_ir_matches_reference() -> None:
    clear_acoustic_caches()
    voice = _voice(seconds=1.0)
    rng = np.random.default_rng(3)
    ir = (rng.standard_normal((2, 9000)) * np.exp(-np.arange(9000) / 2000.0)).astype(np.float32)
    ir_buf = AudioBuffer(samples=ir / np.sqrt(np.mean(np.sum(ir**2, axis=1))), sample_rate=48000)
    acoustics = ACOUSTIC_CASES[0]
    expected = _reference_simulate(voice, ir_buf, acoustics)
    out = simulate_acoustic_space(voice, ir_buffer=ir_buf, acoustics=acoustics)
    np.testing.assert_allclose(out.samples, expected, rtol=1e-4, atol=2e-5)


@pytest.mark.unit
def test_wet_cache_reused_for_mix_changes_and_invalidated_for_ir_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_acoustic_caches()
    calls = {"n": 0}
    real = acoustic_simulator.oaconvolve

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(acoustic_simulator, "oaconvolve", counting)
    voice = _voice(seconds=1.0)
    base = dict(ACOUSTIC_CASES[0])
    first = simulate_acoustic_space(voice, ir_buffer=None, acoustics=base, cache_token="voice-a")
    assert calls["n"] == 2  # un canal cada uno

    for key, value in (("reverb_mix", 30.0), ("stereo_width", 80.0), ("pan", 20.0),
                       ("predelay_ms", 5.0), ("direct_level", 50.0), ("apparent_distance", 10.0)):
        changed = dict(base, **{key: value})
        out = simulate_acoustic_space(voice, ir_buffer=None, acoustics=changed, cache_token="voice-a")
        np.testing.assert_allclose(out.samples, _reference_simulate(voice, None, changed), rtol=1e-4, atol=2e-5)
    assert calls["n"] == 2, "Los parámetros de mezcla no deben recalcular la convolución"

    # Cambiar la IR o la voz invalida la convolución.
    simulate_acoustic_space(voice, ir_buffer=None, acoustics=dict(base, decay_s=0.9), cache_token="voice-a")
    assert calls["n"] == 4
    other_voice = _voice(seconds=1.0, seed=99)
    simulate_acoustic_space(other_voice, ir_buffer=None, acoustics=dict(base, decay_s=0.9), cache_token="voice-b")
    assert calls["n"] == 6

    # El resultado devuelto es independiente de la caché (escribible y no compartido).
    assert first.samples.flags.writeable
    first.samples[:] = 0.0
    again = simulate_acoustic_space(voice, ir_buffer=None, acoustics=base, cache_token="voice-a")
    assert np.abs(again.samples).max() > 0.01


@pytest.mark.unit
def test_cached_ir_is_not_mutable_through_returned_buffer() -> None:
    ir1 = build_algorithmic_room_ir(48000, room_size=40.0, decay_s=0.5, damping=30.0)
    ir1.samples[:] = 0.0
    ir2 = build_algorithmic_room_ir(48000, room_size=40.0, decay_s=0.5, damping=30.0)
    assert np.abs(ir2.samples).max() > 0.0


def _reference_denoise(buffer: AudioBuffer, amount: float, smoothing: float) -> np.ndarray:
    """Algoritmo original trama a trama de apply_spectral_denoise."""
    amt = amount
    smooth = smoothing
    n_fft, hop = 1024, 256
    win = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    out = np.zeros_like(buffer.samples)
    min_floor = max(0.03, 1.0 - 0.94 * amt)
    for ch in range(buffer.channels):
        x = buffer.samples[ch]
        n = x.shape[0]
        pad_tail = (hop - ((n - n_fft) % hop)) % hop
        x_pad = np.pad(x, (0, pad_tail))
        starts = range(0, x_pad.shape[0] - n_fft + 1, hop)
        frames = np.stack([x_pad[s : s + n_fft] * win for s in starts])
        spec = np.fft.rfft(frames, axis=1)
        mag = np.abs(spec)
        energy = np.mean(mag**2, axis=1)
        quiet = energy <= np.percentile(energy, 12.0)
        noise = np.mean(mag[quiet], axis=0, keepdims=True)
        gains = np.maximum(np.maximum(mag - (0.8 + 2.2 * amt) * noise, 0.0) / np.maximum(mag, 1e-8), min_floor)
        for t in range(1, gains.shape[0]):
            gains[t] = smooth * gains[t - 1] + (1.0 - smooth) * gains[t]
        recon = np.fft.irfft(spec * gains, n=n_fft, axis=1).astype(np.float32)
        y = np.zeros(x_pad.shape[0], dtype=np.float32)
        norm = np.zeros_like(y)
        for idx, s in enumerate(starts):
            y[s : s + n_fft] += recon[idx] * win
            norm[s : s + n_fft] += win * win
        out[ch] = (y / np.maximum(norm, 1e-6))[:n]
    return out


@pytest.mark.unit
def test_spectral_denoise_matches_reference_interior_and_has_no_edge_click() -> None:
    voice = _voice(seconds=3.0)
    expected = _reference_denoise(voice, 0.6, 0.4)
    out = apply_spectral_denoise(voice, amount=0.6, smoothing=0.4).samples
    assert out.shape == voice.samples.shape
    assert np.isfinite(out).all()
    interior = slice(4096, -4096)
    err = out[:, interior] - expected[:, interior]
    snr_db = 10 * np.log10(np.sum(expected[:, interior] ** 2) / np.sum(err**2))
    assert snr_db > 30.0
    # La normalización overlap-add ya no amplifica las primeras muestras.
    assert np.abs(out[:, :1024]).max() <= np.abs(voice.samples[:, :2048]).max() * 1.5


@pytest.mark.unit
@pytest.mark.parametrize("length", [128, 300, 1024, 1500, 48000 + 17])
def test_stft_stages_handle_short_and_odd_lengths(length: int) -> None:
    rng = np.random.default_rng(length)
    buf = AudioBuffer(samples=(rng.standard_normal((2, length)) * 0.1).astype(np.float32), sample_rate=48000)
    for out in (apply_spectral_denoise(buf, amount=0.7), apply_dsp_dereverb(buf, amount=0.6)):
        assert out.samples.shape == buf.samples.shape
        assert np.isfinite(out.samples).all()
        assert np.abs(out.samples).max() < 1.0


@pytest.mark.unit
def test_compiled_kernels_match_python_reference() -> None:
    rng = np.random.default_rng(11)
    desired = np.clip(1.4 - np.abs(rng.standard_normal(20000)), 0.05, 1.0)
    np.testing.assert_array_equal(
        kernels.limiter_gain_smoothing(desired, 0.9995),
        kernels._python_limiter_smoothing(desired.copy(), 0.9995),
    )
    target = -np.abs(rng.standard_normal(5000)) * 6.0
    np.testing.assert_array_equal(
        kernels.attack_release_smoothing(target, 0.8, 0.99),
        kernels._python_attack_release(target.copy(), 0.8, 0.99),
    )


@pytest.mark.unit
def test_limiter_and_compressor_outputs_are_bounded_and_finite() -> None:
    loud = AudioBuffer(samples=_voice(seconds=2.0).samples * 4.0, sample_rate=48000)
    limited = apply_safety_limiter(loud, ceiling_db=-1.0).samples
    ceiling = 10 ** (-1.0 / 20.0)
    assert np.isfinite(limited).all()
    assert np.abs(limited).max() <= ceiling + 1e-6
    compressed = apply_compressor(loud, threshold_db=-20.0, ratio=4.0).samples
    assert np.isfinite(compressed).all()
    assert np.abs(compressed).max() < np.abs(loud.samples).max()
