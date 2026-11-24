"""Fixtures y generadores de señales sintéticas controladas para pruebas sin modelos pesados."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.configuration import AppSettings
from core.paths import AppPaths
from engine.audio_types import AudioBuffer


@pytest.fixture()
def portable_paths(tmp_path: Path) -> AppPaths:
    """Crea un entorno de rutas portables aislado dentro de tmp_path."""
    fake_exe = tmp_path / "ZenTralAudio.exe"
    fake_exe.write_bytes(b"")
    paths = AppPaths.detect(frozen=True, executable_path=fake_exe, meipass_path=tmp_path / "_internal")
    paths.ensure_writable_directories()
    return paths


@pytest.fixture()
def default_settings() -> AppSettings:
    return AppSettings()


def make_sine_buffer(
    *,
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 48000,
    channels: int = 1,
    amplitude: float = 0.5,
) -> AudioBuffer:
    t = np.arange(int(round(duration_s * sample_rate)), dtype=np.float32) / float(sample_rate)
    wave = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    samples = np.tile(wave, (channels, 1))
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def make_noisy_speech_like_buffer(sample_rate: int = 48000, duration_s: float = 1.2) -> AudioBuffer:
    """Genera señal vocal sintética (armónicos + silencios) contaminada con ruido blanco estacionario."""
    rng = np.random.default_rng(42)
    n = int(round(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    voice = (
        0.35 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 360.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 1800.0 * t)
    ).astype(np.float32)
    # Dejar pausas silenciosas en los extremos para que el estimador espectral mida el ruido real
    env = np.zeros(n, dtype=np.float32)
    env[int(n * 0.25) : int(n * 0.75)] = 1.0
    noise = rng.normal(0.0, 0.04, size=n).astype(np.float32)
    return AudioBuffer(samples=(voice * env + noise)[np.newaxis, :], sample_rate=sample_rate)


@pytest.fixture()
def sample_wav_file(tmp_path: Path) -> Path:
    buf = make_sine_buffer(freq_hz=440.0, duration_s=0.5, sample_rate=48000, channels=2, amplitude=0.4)
    path = tmp_path / "input_stereo.wav"
    sf.write(str(path), buf.samples.T, buf.sample_rate, subtype="PCM_24")
    return path
