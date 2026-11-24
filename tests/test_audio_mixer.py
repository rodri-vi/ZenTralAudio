"""Pruebas unitarias de mezcla dry/wet, panorámica y capas ambientales."""

from __future__ import annotations

import numpy as np
import pytest

from engine.audio_mixer import apply_pan, dry_wet_mix, mix_ambience_layers
from tests.conftest import make_sine_buffer


@pytest.mark.unit
def test_dry_wet_extremes() -> None:
    dry = make_sine_buffer(freq_hz=220.0, duration_s=0.2, amplitude=0.4)
    wet = make_sine_buffer(freq_hz=440.0, duration_s=0.2, amplitude=0.8)

    out_0 = dry_wet_mix(dry, wet, 0.0)
    out_100 = dry_wet_mix(dry, wet, 1.0)
    out_50 = dry_wet_mix(dry, wet, 0.5)

    assert np.allclose(out_0.samples, dry.samples)
    assert np.allclose(out_100.samples, wet.samples)
    assert np.allclose(out_50.samples, 0.5 * dry.samples + 0.5 * wet.samples, atol=1e-6)


@pytest.mark.unit
def test_pan_and_ambience_solo() -> None:
    mono = make_sine_buffer(freq_hz=300.0, duration_s=0.2, channels=1, amplitude=0.3)
    panned_l = apply_pan(mono, -1.0)
    assert panned_l.channels == 2
    assert np.max(np.abs(panned_l.samples[0])) > 0.3
    assert np.max(np.abs(panned_l.samples[1])) < 1e-5

    voice = make_sine_buffer(freq_hz=200.0, duration_s=0.4, channels=1, amplitude=0.2)
    layer1 = make_sine_buffer(freq_hz=1000.0, duration_s=0.1, channels=1, amplitude=0.4)
    layer2 = make_sine_buffer(freq_hz=2500.0, duration_s=0.1, channels=1, amplitude=0.4)

    mixed = mix_ambience_layers(
        voice,
        [
            (layer1, {"enabled": True, "volume": 0.5, "pan": 0.0, "solo": False}),
            (layer2, {"enabled": True, "volume": 0.5, "pan": 0.0, "solo": True}),
        ],
        master_volume=1.0,
    )
    # Solo debe haberse mezclado layer2 (2500 Hz) además de la voz, no layer1 (1000 Hz)
    fft_mag = np.abs(np.fft.rfft(mixed.samples[0]))
    freqs = np.fft.rfftfreq(mixed.num_samples, d=1.0 / mixed.sample_rate)
    idx_1000 = int(np.argmin(np.abs(freqs - 1000.0)))
    idx_2500 = int(np.argmin(np.abs(freqs - 2500.0)))
    assert fft_mag[idx_2500] > fft_mag[idx_1000] * 5.0
