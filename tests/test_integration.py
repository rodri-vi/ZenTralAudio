"""Pruebas de integración extremo a extremo: UI, caché de modelos, ambientes y paridad."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.dialogs.export_dialog import ExportRequest
from core.configuration import AppSettings, ConfigManager
from core.job_manager import CancellationToken, JobManager
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_types import AudioBuffer
from engine.export_engine import export_with_pipeline
from engine.preview_engine import get_shared_pipeline, render_preview_from_state
from engine.processing_graph import PipelineConfig
from models.base_adapter import BaseModelAdapter, ModelCapabilities, ModelStatus, ParameterSpec
from tests.conftest import make_noisy_speech_like_buffer, make_sine_buffer


class _SyntheticTestAdapter(BaseModelAdapter):
    """Adaptador sintético determinista para probar caché Etapa 2 y resampling sin pesos reales."""

    def __init__(self) -> None:
        self.calls = 0

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id="synthetic_16k",
            display_name="Synthetic16K",
            version="test",
            required_sample_rate=16000,
        )

    def get_parameter_schema(self) -> list[ParameterSpec]:
        return []

    def get_status(self) -> ModelStatus:
        return ModelStatus.AVAILABLE

    def load_model(self, device: str = "auto") -> None:
        return None

    def unload_model(self) -> None:
        return None

    def process(
        self,
        buffer: AudioBuffer,
        params: dict[str, Any],
        *,
        progress_cb: Any = None,
        cancel_token: CancellationToken | None = None,
    ) -> AudioBuffer:
        self.calls += 1
        return AudioBuffer(samples=buffer.samples * 0.85, sample_rate=16000)


@pytest.mark.integration
def test_end_to_end_pipeline_with_ai_cache_and_environment_parity(
    tmp_path: Path,
    portable_paths: AppPaths,
    default_settings: AppSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models import model_registry

    adapter = _SyntheticTestAdapter()
    reg = model_registry.get_global_registry(portable_paths)
    monkeypatch.setitem(reg._adapters, "synthetic_16k", adapter)

    pipeline = get_shared_pipeline()
    pipeline.clear_cache()

    voice = make_noisy_speech_like_buffer(48000, 0.5)
    amb_path = tmp_path / "amb_test.wav"
    amb = make_sine_buffer(freq_hz=800.0, duration_s=0.25, sample_rate=48000, amplitude=0.15)
    sf.write(str(amb_path), amb.samples.T, 48000, subtype="FLOAT")

    state_a = {
        "model_id": "synthetic_16k",
        "intensity": 80,
        "advanced_params": {"nr_enabled": True, "nr_amount": 0.6, "eq_hp_freq": 80},
        "environment": {
            "enabled": True,
            "environment_id": "bosque",
            "master_volume": 0.7,
            "layers": [
                {"layer_id": "l1", "file_path": str(amb_path), "enabled": True, "volume": 0.4, "pan": 0.1, "solo": False}
            ],
        },
    }

    preview_1 = render_preview_from_state(voice.samples, voice.sample_rate, state_a, portable_paths)
    assert adapter.calls == 1

    # Modificar solo EQ: debe reutilizar la caché Etapa 2 del modelo IA sin re-ejecutarlo
    state_b = dict(state_a)
    state_b["advanced_params"] = {"nr_enabled": True, "nr_amount": 0.6, "eq_hp_freq": 120}
    preview_2 = render_preview_from_state(voice.samples, voice.sample_rate, state_b, portable_paths)
    assert adapter.calls == 1

    out_wav = tmp_path / "e2e_export.wav"
    req = ExportRequest(
        output_path=out_wav,
        export_format="wav",
        sample_rate=48000,
        bit_depth=32,
        mp3_bitrate_kbps=320,
        channels=2,
        include_environment=True,
        allow_overwrite=True,
    )
    export_with_pipeline(voice.samples, voice.sample_rate, state_b, req, portable_paths, default_settings)
    exported, _ = load_audio(out_wav, paths=portable_paths, config=default_settings)
    assert np.allclose(preview_2, exported.samples, atol=1e-5)
    assert preview_1.shape == preview_2.shape


@pytest.mark.integration
def test_main_window_structure_two_tabs_and_permanent_player(
    portable_paths: AppPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cfg_mgr = ConfigManager(portable_paths.config_dir)
    cfg_mgr.load()
    jm = JobManager(max_workers=1)
    win = MainWindow(paths=portable_paths, config_manager=cfg_mgr, job_manager=jm)
    try:
        assert win.tabs.count() == 2
        assert win.tabs.tabText(0) == "RESTAURACIÓN IA"
        assert win.tabs.tabText(1) == "AMBIENTES ACÚSTICOS"
        assert not win.preview_player.isHidden()

        win.tabs.setCurrentIndex(1)
        assert not win.preview_player.isHidden()

        win.controller._on_env_selected("bosque")
        assert len(win.controller._layers) == 3

        spec = PipelineConfig.from_controller_state(win.controller.export_pipeline_dict())
        spec.validate()
    finally:
        win.close()
        jm.shutdown()
        _ = app
