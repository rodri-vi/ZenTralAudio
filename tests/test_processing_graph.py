"""Pruebas unitarias de serialización, validación y paridad Preview vs Export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.dialogs.export_dialog import ExportRequest
from core.configuration import AppSettings
from core.exceptions import ValidationError
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.export_engine import export_with_pipeline
from engine.preview_engine import render_preview_from_state
from engine.processing_graph import PipelineConfig
from tests.conftest import make_noisy_speech_like_buffer


@pytest.mark.unit
def test_pipeline_config_validation_rejects_invalid_bounds() -> None:
    cfg = PipelineConfig(intensity=150)
    with pytest.raises(ValidationError):
        cfg.validate()


@pytest.mark.unit
def test_intensity_zero_matches_original_and_preview_equals_export(
    tmp_path: Path,
    portable_paths: AppPaths,
    default_settings: AppSettings,
) -> None:
    buf = make_noisy_speech_like_buffer(48000, 0.5)
    state_zero = {
        "model_id": "deepfilternet3",
        "intensity": 0,
        "advanced_params": {"nr_enabled": True, "nr_amount": 0.8},
        "environment": {"enabled": False},
    }
    out_zero = render_preview_from_state(buf.samples, buf.sample_rate, state_zero, portable_paths)
    assert np.allclose(out_zero, buf.samples, atol=1e-6)

    state_active = {
        "model_id": "deepfilternet3",
        "intensity": 80,
        "advanced_params": {
            "nr_enabled": True,
            "nr_amount": 0.65,
            "eq_hp_freq": 90,
            "comp_enabled": True,
            "comp_threshold": -18.0,
            "comp_ratio": 3.0,
            "vocal_presence": 2.0,
        },
        "environment": {"enabled": False},
    }
    preview_samples = render_preview_from_state(buf.samples, buf.sample_rate, state_active, portable_paths)

    out_wav = tmp_path / "parity_export.wav"
    req = ExportRequest(
        output_path=out_wav,
        export_format="wav",
        sample_rate=48000,
        bit_depth=32,
        mp3_bitrate_kbps=320,
        channels=1,
        include_environment=False,
        allow_overwrite=True,
    )
    export_with_pipeline(buf.samples, buf.sample_rate, state_active, req, portable_paths, default_settings)
    exported_buf, _ = load_audio(out_wav, paths=portable_paths, config=default_settings)
    assert np.allclose(preview_samples, exported_buf.samples, atol=1e-5)
