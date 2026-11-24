"""Motor de previsualización conectado a la instancia compartida de ProcessingPipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.job_manager import JobContext
from core.paths import AppPaths
from engine.audio_types import AudioBuffer
from engine.processing_graph import PipelineConfig
from engine.processing_pipeline import ProcessingPipeline

_SHARED_PIPELINE = ProcessingPipeline()


def get_shared_pipeline() -> ProcessingPipeline:
    """Devuelve la instancia única compartida entre previsualización y exportación."""
    return _SHARED_PIPELINE


def render_preview_from_state(
    samples: np.ndarray,
    sample_rate: int,
    state_dict: dict[str, Any],
    paths: AppPaths,
    ctx: JobContext | None = None,
) -> np.ndarray:
    """Renderiza el audio de previsualización usando exactamente la cadena común."""
    buf = AudioBuffer(samples=samples, sample_rate=sample_rate)
    config = PipelineConfig.from_controller_state(state_dict)
    token = ctx.token if ctx is not None else None
    progress_cb = ctx.report_progress if ctx is not None else None
    out = _SHARED_PIPELINE.render(
        buf,
        config,
        paths=paths,
        cancel_token=token,
        progress_cb=progress_cb,
    )
    return out.samples
