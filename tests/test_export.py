"""Pruebas de exportación WAV/FLAC, protección de sobrescritura y cancelación limpia."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.dialogs.export_dialog import ExportRequest
from core.configuration import AppSettings
from core.exceptions import JobCancelledError, ValidationError
from core.job_manager import JobManager
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.export_engine import export_with_pipeline
from tests.conftest import make_sine_buffer


@pytest.mark.unit
def test_export_wav_and_flac(tmp_path: Path, portable_paths: AppPaths, default_settings: AppSettings) -> None:
    buf = make_sine_buffer(freq_hz=520.0, duration_s=0.4, sample_rate=48000, channels=2, amplitude=0.5)
    state = {"intensity": 50, "advanced_params": {"nr_enabled": False}, "environment": {"enabled": False}}

    for fmt in ("wav", "flac"):
        out_path = tmp_path / f"result.{fmt}"
        req = ExportRequest(
            output_path=out_path,
            export_format=fmt,
            sample_rate=48000,
            bit_depth=24,
            mp3_bitrate_kbps=320,
            channels=2,
            include_environment=False,
            allow_overwrite=False,
        )
        export_with_pipeline(buf.samples, buf.sample_rate, state, req, portable_paths, default_settings)
        assert out_path.is_file()
        loaded, meta = load_audio(out_path, paths=portable_paths, config=default_settings)
        assert meta.sample_rate == 48000
        assert loaded.channels == 2


@pytest.mark.unit
def test_export_refuses_overwrite_source_file(
    sample_wav_file: Path,
    portable_paths: AppPaths,
    default_settings: AppSettings,
) -> None:
    buf, _ = load_audio(sample_wav_file, paths=portable_paths, config=default_settings)
    req = ExportRequest(
        output_path=sample_wav_file,
        export_format="wav",
        sample_rate=48000,
        bit_depth=24,
        mp3_bitrate_kbps=320,
        channels=2,
        include_environment=False,
        allow_overwrite=True,
    )
    with pytest.raises(ValidationError):
        export_with_pipeline(
            buf.samples,
            buf.sample_rate,
            {"intensity": 20},
            req,
            portable_paths,
            default_settings,
            source_path=sample_wav_file,
        )


@pytest.mark.unit
def test_export_cancellation_leaves_no_corrupt_file(
    tmp_path: Path,
    portable_paths: AppPaths,
    default_settings: AppSettings,
) -> None:
    buf = make_sine_buffer(duration_s=0.3, sample_rate=48000)
    out_path = tmp_path / "cancelled.wav"
    req = ExportRequest(
        output_path=out_path,
        export_format="wav",
        sample_rate=48000,
        bit_depth=24,
        mp3_bitrate_kbps=320,
        channels=1,
        include_environment=False,
        allow_overwrite=True,
    )
    jm = JobManager(max_workers=1)
    try:
        handle = jm.submit(
            "cancel_export",
            lambda ctx: (
                ctx.token.cancel(),
                export_with_pipeline(buf.samples, buf.sample_rate, {"intensity": 50}, req, portable_paths, default_settings, ctx),
            )[-1],
        )
        with pytest.raises(JobCancelledError):
            handle.wait(timeout=5.0)
        assert not out_path.exists()
    finally:
        jm.shutdown()
