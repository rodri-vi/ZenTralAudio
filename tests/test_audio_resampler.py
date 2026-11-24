"""Pruebas unitarias del resampler y alineación exacta de longitud."""

from __future__ import annotations

import numpy as np
import pytest

from engine.audio_resampler import resample_audio
from tests.conftest import make_sine_buffer


@pytest.mark.unit
def test_roundtrip_resample_preserves_exact_length() -> None:
    orig = make_sine_buffer(freq_hz=1000.0, duration_s=0.73, sample_rate=48000, channels=2)
    down = resample_audio(orig, 16000)
    assert down.sample_rate == 16000
    assert down.channels == 2

    back = resample_audio(down, 48000, target_length=orig.num_samples)
    assert back.sample_rate == 48000
    assert back.num_samples == orig.num_samples
    # La correlación entre el tono de 1 kHz original y el re-muestreado debe ser muy alta
    corr = float(np.corrcoef(orig.samples[0, 500:-500], back.samples[0, 500:-500])[0, 1])
    assert corr > 0.98
