"""Motor unificado de ambientes acusticos conectado al ProcessingPipeline."""

<<<<<<< HEAD
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_mixer import mix_ambience_layers
from engine.audio_types import AudioBuffer
from engine.processing_graph import PipelineConfig
from environments.acoustic_simulator import simulate_acoustic_space
from environments.ambience_manager import AmbienceManager
from environments.environment_catalog import EnvironmentCatalog
from environments.impulse_response_manager import ImpulseResponseManager

logger = logging.getLogger("zentralaudio.environments.engine")


def apply_environment_stage(
    voice: AudioBuffer,
    *,
    config: PipelineConfig,
    paths: AppPaths,
    cancel_token: CancellationToken | None = None,
) -> AudioBuffer:
    """Aplica simulación acústica (IR o algorítmica) y mezcla las capas ambientales activas."""
    if cancel_token:
        cancel_token.raise_if_cancelled()

    catalog = EnvironmentCatalog(paths)
    env_def = catalog.get(config.environment_id)
    acoustics = dict(env_def.acoustics) if env_def else {}
    acoustics.update(config.acoustics)

    ir_manager = ImpulseResponseManager(paths)
    amb_manager = AmbienceManager(paths)

    # 1. Resolver IR si el usuario cargó una local o si existe una descargada para el ambiente
    ir_buffer: AudioBuffer | None = None
    if config.ir_path and Path(config.ir_path).is_file():
        ir_buffer = ir_manager.load_ir(config.ir_path, voice.sample_rate)
    else:
        for res_summary in amb_manager.list_summaries():
            if (
                res_summary["kind"] == "ir"
                and res_summary["status_code"] == "available"
                and res_summary["id"].endswith(config.environment_id)
            ):
                ir_buffer = ir_manager.load_ir(res_summary["local_path"], voice.sample_rate)
                break

    if cancel_token:
        cancel_token.raise_if_cancelled()

    spatial_voice = simulate_acoustic_space(voice, ir_buffer=ir_buffer, acoustics=acoustics)

    # 2. Cargar y mezclar capas ambientales con archivo local o recurso disponible
    loaded_layers: list[tuple[AudioBuffer, dict[str, Any]]] = []
    for layer_cfg in config.ambience_layers:
        if cancel_token:
            cancel_token.raise_if_cancelled()

        file_path: Path | None = None
        if layer_cfg.get("file_path") and Path(str(layer_cfg["file_path"])).is_file():
            file_path = Path(str(layer_cfg["file_path"]))
        elif layer_cfg.get("resource_id"):
            res = amb_manager.get_resource(str(layer_cfg["resource_id"]))
            if res is not None:
                candidate = amb_manager.resolve_local_path(res)
                if candidate.is_file():
                    file_path = candidate

        if file_path is not None:
            try:
                lbuf, _ = load_audio(file_path, paths=paths)
                loaded_layers.append((lbuf, layer_cfg))
            except Exception as exc:
                logger.warning("No se pudo cargar capa ambiental '%s': %s", file_path, exc)

    if not loaded_layers:
        return spatial_voice

    return mix_ambience_layers(
        spatial_voice,
        loaded_layers,
        master_volume=config.master_ambience_volume,
    )
=======
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_mixer import mix_ambience_layers
from engine.audio_types import AudioBuffer
from engine.processing_graph import PipelineConfig
from environments.acoustic_simulator import simulate_acoustic_space
from environments.ambience_manager import AmbienceManager
from environments.environment_catalog import EnvironmentCatalog
from environments.impulse_response_manager import ImpulseResponseManager

logger = logging.getLogger("zentralaudio.environments.engine")


def apply_environment_stage(
    voice: AudioBuffer,
    *,
    config: PipelineConfig,
    paths: AppPaths,
    cancel_token: CancellationToken | None = None,
) -> AudioBuffer:
    """Aplica simulación acústica (IR o algorítmica) y mezcla las capas ambientales activas."""
    if cancel_token:
        cancel_token.raise_if_cancelled()

    catalog = EnvironmentCatalog(paths)
    env_def = catalog.get(config.environment_id)
    acoustics = dict(env_def.acoustics) if env_def else {}
    acoustics.update(config.acoustics)

    ir_manager = ImpulseResponseManager(paths)
    amb_manager = AmbienceManager(paths)

    # 1. Resolver IR si el usuario cargó una local o si existe una descargada para el ambiente
    ir_buffer: AudioBuffer | None = None
    if config.ir_path and Path(config.ir_path).is_file():
        ir_buffer = ir_manager.load_ir(config.ir_path, voice.sample_rate)
    else:
        for res_summary in amb_manager.list_summaries():
            if (
                res_summary["kind"] == "ir"
                and res_summary["status_code"] == "available"
                and res_summary["id"].endswith(config.environment_id)
            ):
                ir_buffer = ir_manager.load_ir(res_summary["local_path"], voice.sample_rate)
                break

    if cancel_token:
        cancel_token.raise_if_cancelled()

    spatial_voice = simulate_acoustic_space(voice, ir_buffer=ir_buffer, acoustics=acoustics)

    # 2. Cargar y mezclar capas ambientales con archivo local o recurso disponible
    loaded_layers: list[tuple[AudioBuffer, dict[str, Any]]] = []
    for layer_cfg in config.ambience_layers:
        if cancel_token:
            cancel_token.raise_if_cancelled()

        file_path: Path | None = None
        if layer_cfg.get("file_path") and Path(str(layer_cfg["file_path"])).is_file():
            file_path = Path(str(layer_cfg["file_path"]))
        elif layer_cfg.get("resource_id"):
            res = amb_manager.get_resource(str(layer_cfg["resource_id"]))
            if res is not None:
                candidate = amb_manager.resolve_local_path(res)
                if candidate.is_file():
                    file_path = candidate

        if file_path is not None:
            try:
                lbuf, _ = load_audio(file_path, paths=paths)
                loaded_layers.append((lbuf, layer_cfg))
            except Exception as exc:
                logger.warning("No se pudo cargar capa ambiental '%s': %s", file_path, exc)

    if not loaded_layers:
        return spatial_voice

    return mix_ambience_layers(
        spatial_voice,
        loaded_layers,
        master_volume=config.master_ambience_volume,
    )

>>>>>>> 2f6beded7ad89e5c8d2dfbb7461aba8cf002655a