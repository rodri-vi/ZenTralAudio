"""Motor de exportación atómica que reutiliza exactamente la misma cadena de previsualización."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.configuration import AppSettings
from core.exceptions import ExportError
from core.job_manager import JobContext
from core.paths import AppPaths
from core.validation import validate_export_target
from engine.audio_io import write_audio
from engine.audio_resampler import resample_audio
from engine.audio_types import AudioBuffer
from engine.preview_engine import get_shared_pipeline
from engine.processing_graph import PipelineConfig


def export_with_pipeline(
    samples: np.ndarray,
    sample_rate: int,
    pipeline_dict: dict[str, Any],
    req: Any,
    paths: AppPaths,
    settings: AppSettings,
    ctx: JobContext | None = None,
    *,
    source_path: Path | None = None,
) -> Path:
    """Procesa y exporta el audio final garantizando paridad con la previsualización."""
    output_path = Path(req.output_path)
    validate_export_target(
        output_path,
        source_path=source_path,
        export_format=req.export_format,
        sample_rate=req.sample_rate,
        allow_overwrite=req.allow_overwrite,
    )

    token = ctx.token if ctx is not None else None
    progress_cb = ctx.report_progress if ctx is not None else None

    source_buf = AudioBuffer(samples=samples, sample_rate=sample_rate)
    config = PipelineConfig.from_controller_state(pipeline_dict)

    rendered = get_shared_pipeline().render(
        source_buf,
        config,
        paths=paths,
        cancel_token=token,
        progress_cb=progress_cb,
    )

    if token:
        token.raise_if_cancelled()

    if progress_cb:
        progress_cb(0.92, f"Formateando salida ({req.export_format.upper()})...")

    out_buf = rendered
    if out_buf.sample_rate != int(req.sample_rate):
        out_buf = resample_audio(out_buf, int(req.sample_rate))
    if out_buf.channels != int(req.channels):
        out_buf = out_buf.match_channels(int(req.channels))

    verified_meta = write_audio(
        out_buf,
        output_path,
        export_format=req.export_format,
        bit_depth=int(req.bit_depth),
        mp3_bitrate_kbps=int(req.mp3_bitrate_kbps),
        paths=paths,
        config=settings,
        cancel_token=token,
    )

    if verified_meta.sample_rate != int(req.sample_rate) or verified_meta.channels != int(req.channels):
        raise ExportError(
            "El archivo exportado no coincide con las propiedades seleccionadas.",
            details=(
                f"Esperado {req.sample_rate} Hz / {req.channels} ch, "
                f"obtenido {verified_meta.sample_rate} Hz / {verified_meta.channels} ch."
            ),
        )

    if progress_cb:
        progress_cb(1.0, f"Exportación verificada: {output_path.name}")

    return output_path
