"""Pruebas unitarias e integración de lectura, validación y escritura de audio."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from core.configuration import AppSettings
from core.exceptions import JobCancelledError, ValidationError
from core.paths import AppPaths
from core.job_manager import CancellationToken
from engine.audio_io import _run_subprocess_cancellable, load_audio, write_audio
from tests.conftest import make_sine_buffer


@pytest.mark.unit
def test_load_wav_preserves_properties(sample_wav_file: Path, portable_paths: AppPaths, default_settings: AppSettings) -> None:
    buf, meta = load_audio(sample_wav_file, paths=portable_paths, config=default_settings)
    assert buf.sample_rate == 48000
    assert buf.channels == 2
    assert meta.bit_depth == 24
    assert pytest.approx(meta.duration_seconds, abs=1e-3) == 0.5


@pytest.mark.unit
def test_load_audio_stops_immediately_when_already_cancelled(
    sample_wav_file: Path,
) -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(JobCancelledError):
        load_audio(sample_wav_file, cancel_token=token)


@pytest.mark.unit
def test_cancellable_subprocess_terminates_running_ffmpeg_like_process() -> None:
    token = CancellationToken()
    timer = threading.Timer(0.2, token.cancel)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(JobCancelledError):
            _run_subprocess_cancellable(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cancel_token=token,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
    finally:
        timer.cancel()

    assert time.monotonic() - started < 3.0


@pytest.mark.unit
def test_load_mono_and_stereo_wav(tmp_path: Path) -> None:
    mono = make_sine_buffer(channels=1, sample_rate=44100, duration_s=0.25)
    mono_path = tmp_path / "mono.wav"
    sf.write(str(mono_path), mono.samples.T, mono.sample_rate, subtype="PCM_16")

    loaded, meta = load_audio(mono_path)
    assert loaded.channels == 1
    assert loaded.sample_rate == 44100
    assert meta.bit_depth == 16


@pytest.mark.unit
def test_reject_empty_or_invalid_extension(tmp_path: Path) -> None:
    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(ValidationError):
        load_audio(empty_wav)

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("hola", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_audio(bad_ext)


@pytest.mark.integration
def test_ffmpeg_mp3_roundtrip_if_available(tmp_path: Path, portable_paths: AppPaths, default_settings: AppSettings) -> None:
    if portable_paths.find_ffmpeg() is None:
        pytest.skip("FFmpeg no disponible en este entorno.")
    buf = make_sine_buffer(channels=2, sample_rate=44100, duration_s=0.4)
    mp3_path = tmp_path / "test_out.mp3"
    meta = write_audio(buf, mp3_path, export_format="mp3", mp3_bitrate_kbps=192, paths=portable_paths, config=default_settings)
    assert mp3_path.is_file()
    assert meta.sample_rate == 44100
    assert meta.channels == 2
    reloaded, _ = load_audio(mp3_path, paths=portable_paths, config=default_settings)
    assert reloaded.num_samples > 0


@pytest.mark.unit
def test_mp3_roundtrip_without_ffmpeg(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MP3 se exporta y se lee con libsndfile aunque no haya FFmpeg disponible."""
    import numpy as np

    from core.paths import AppPaths
    from engine import audio_io
    from engine.audio_types import AudioBuffer

    if not audio_io._LIBSNDFILE_MP3:
        pytest.skip("libsndfile sin soporte MP3 en este entorno")
    monkeypatch.setattr(AppPaths, "find_ffmpeg", lambda self, custom=None: None)
    monkeypatch.setattr(AppPaths, "find_ffprobe", lambda self, custom=None: None)

    t = np.arange(44100 * 2, dtype=np.float32) / 44100
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = AudioBuffer(samples=np.stack([tone, tone]), sample_rate=44100)
    target = tmp_path / "out.mp3"
    meta = audio_io.write_audio(buf, target, export_format="mp3", mp3_bitrate_kbps=320)
    assert target.is_file() and target.stat().st_size > 10_000
    assert meta.sample_rate == 44100

    loaded, info = audio_io.load_audio(target)
    assert loaded.channels == 2
    assert abs(loaded.duration_seconds - 2.0) < 0.1
    assert np.isfinite(loaded.samples).all()
    assert 0.15 < float(np.max(np.abs(loaded.samples))) < 0.45


@pytest.mark.unit
def test_ffmpeg_gatekeeper_kill_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from engine import audio_io

    monkeypatch.setattr(audio_io.sys, "platform", "darwin")
    message = audio_io._describe_ffmpeg_failure(Path("/tmp/ffmpeg"), -9, "")
    assert "macOS bloqueó" in message
