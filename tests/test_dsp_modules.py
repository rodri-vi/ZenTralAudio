"""Pruebas unitarias de los módulos DSP con señales sintéticas controladas."""

from __future__ import annotations

import numpy as np
import pytest

from engine.audio_types import AudioBuffer
from engine.dsp.dynamics import apply_compressor
from engine.dsp.equalizer import apply_equalizer
from engine.dsp.safety_limiter import apply_safety_limiter
from engine.dsp.spectral_denoise import apply_spectral_denoise
from engine.dsp.vocal_treatment import apply_vocal_treatment
from tests.conftest import make_noisy_speech_like_buffer, make_sine_buffer


@pytest.mark.unit
def test_spectral_denoise_reduces_noise_floor() -> None:
    noisy = make_noisy_speech_like_buffer(48000, 1.0)
    cleaned = apply_spectral_denoise(noisy, amount=0.85, smoothing=0.5)
    # Medir energía en la zona silenciosa inicial (primeros 15% del audio)
    end_quiet = int(noisy.num_samples * 0.15)
    rms_before = float(np.sqrt(np.mean(noisy.samples[:, :end_quiet] ** 2)))
    rms_after = float(np.sqrt(np.mean(cleaned.samples[:, :end_quiet] ** 2)))
    assert rms_after < rms_before * 0.55


@pytest.mark.unit
def test_equalizer_highpass_attenuates_subbass() -> None:
    rumble = make_sine_buffer(freq_hz=35.0, duration_s=0.5, sample_rate=48000, amplitude=0.6)
    filtered = apply_equalizer(rumble, hp_freq=180.0, lp_freq=18000.0)
    rms_in = float(np.sqrt(np.mean(rumble.samples ** 2)))
    rms_out = float(np.sqrt(np.mean(filtered.samples[:, 2000:-2000] ** 2)))
    assert rms_out < rms_in * 0.2


@pytest.mark.unit
def test_compressor_reduces_peaks_above_threshold() -> None:
    hot = make_sine_buffer(freq_hz=300.0, duration_s=0.5, sample_rate=48000, amplitude=0.9)
    comp = apply_compressor(hot, enabled=True, threshold_db=-24.0, ratio=6.0, attack_ms=2.0, release_ms=50.0)
    rms_in = float(np.sqrt(np.mean(hot.samples[:, 4800:] ** 2)))
    rms_out = float(np.sqrt(np.mean(comp.samples[:, 4800:] ** 2)))
    assert rms_out < rms_in * 0.6


@pytest.mark.unit
def test_deesser_attenuates_sibilance_band() -> None:
    sibilant = make_sine_buffer(freq_hz=7000.0, duration_s=0.4, sample_rate=48000, amplitude=0.5)
    treated = apply_vocal_treatment(sibilant, deesser_amount=0.9)
    rms_in = float(np.sqrt(np.mean(sibilant.samples[:, 4800:] ** 2)))
    rms_out = float(np.sqrt(np.mean(treated.samples[:, 4800:] ** 2)))
    assert rms_out < rms_in * 0.65


@pytest.mark.unit
def test_safety_limiter_prevents_clipping() -> None:
    loud = AudioBuffer(samples=np.ones((2, 4800), dtype=np.float32) * 2.5, sample_rate=48000)
    limited = apply_safety_limiter(loud, ceiling_db=-1.0)
    ceiling_lin = 10.0 ** (-1.0 / 20.0)
    assert float(np.max(np.abs(limited.samples))) <= ceiling_lin + 1e-5
    assert np.all(np.isfinite(limited.samples))
