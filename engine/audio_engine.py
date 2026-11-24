"""Fachada unificada de alto nivel sobre carga, procesamiento, previsualización y exportación."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.configuration import AppSettings
from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_types import AudioBuffer, AudioMetadata
from engine.preview_engine import get_shared_pipeline
from engine.processing_graph import PipelineConfig


class AudioEngine:
    """Coordina lectura, renderizado de cadena y limpieza de caché/recursos."""

    def __init__(self, paths: AppPaths, settings: AppSettings) -> None:
        self.paths = paths
        self.settings = settings
        self.pipeline = get_shared_pipeline()

    def load(self, file_path: str | Path) -> tuple[AudioBuffer, AudioMetadata]:
        return load_audio(file_path, paths=self.paths, config=self.settings)

    def process(
        self,
        buffer: AudioBuffer,
        state_dict: dict[str, Any],
        *,
        cancel_token: CancellationToken | None = None,
    ) -> AudioBuffer:
        config = PipelineConfig.from_controller_state(state_dict)
        return self.pipeline.render(
            buffer,
            config,
            paths=self.paths,
            cancel_token=cancel_token,
        )

    def clear_cache(self) -> None:
        self.pipeline.clear_cache()
