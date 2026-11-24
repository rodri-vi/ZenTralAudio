"""Cadena central única de procesamiento con caché multi-etapa y cancelación."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_mixer import dry_wet_mix, mix_ambience_layers
from engine.audio_resampler import resample_audio
from engine.audio_types import AudioBuffer
from engine.dsp.dynamics import apply_compressor, apply_noise_gate
from engine.dsp.equalizer import apply_equalizer
from engine.dsp.safety_limiter import apply_safety_limiter
from engine.dsp.spatial import apply_spatial_stage
from engine.dsp.spectral_denoise import apply_spectral_denoise
from engine.dsp.vocal_treatment import apply_vocal_treatment
from engine.processing_graph import PipelineConfig

logger = logging.getLogger("zentralaudio.pipeline")


def fingerprint_audio(buffer: AudioBuffer) -> str:
    """Genera una huella determinista rápida del AudioBuffer de entrada."""
    h = hashlib.sha256()
    h.update(f"{buffer.sample_rate}:{buffer.channels}:{buffer.num_samples}".encode("ascii"))
    arr = buffer.samples
    if arr.size <= 65536:
        h.update(arr.tobytes())
    else:
        step = max(1, arr.shape[1] // 16384)
        h.update(np.ascontiguousarray(arr[:, ::step]).tobytes())
    return h.hexdigest()[:24]


class PipelineCache:
    """Caché LRU acotada en memoria para evitar recalcular etapas pesadas sin cambios."""

    def __init__(self, max_entries: int = 8) -> None:
        self._max_entries = max(1, max_entries)
        self._lock = threading.Lock()
        self._store: OrderedDict[str, AudioBuffer] = OrderedDict()

    def get(self, key: str) -> AudioBuffer | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            self._store.move_to_end(key)
            return item.copy()

    def put(self, key: str, value: AudioBuffer) -> None:
        with self._lock:
            self._store[key] = value.copy()
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class ProcessingPipeline:
    """Ejecuta de forma coherente IA -> DSP -> Dry/Wet -> Ambiente -> Limitador."""

    def __init__(self) -> None:
        self._ai_cache = PipelineCache(max_entries=4)
        self._full_cache = PipelineCache(max_entries=6)

    def clear_cache(self) -> None:
        self._ai_cache.clear()
        self._full_cache.clear()

    def render(
        self,
        source: AudioBuffer,
        config: PipelineConfig,
        *,
        paths: AppPaths | None = None,
        cancel_token: CancellationToken | None = None,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> AudioBuffer:
        """Renderiza la cadena completa devolviendo un nuevo AudioBuffer seguro."""
        config.validate()
        if cancel_token:
            cancel_token.raise_if_cancelled()

        src_fp = fingerprint_audio(source)
        full_key = config.full_cache_key(src_fp)
        cached_full = self._full_cache.get(full_key)
        if cached_full is not None:
            return cached_full

        app_paths = paths or AppPaths.detect()

        if progress_cb:
            progress_cb(0.05, "Preparando señal...")

        # Atajo técnico: si intensidad=0 y ambiente desactivado, devolver señal original segura
        if config.intensity == 0 and not config.environment_enabled:
            out = apply_safety_limiter(source, ceiling_db=config.true_peak_ceiling_db)
            self._full_cache.put(full_key, out)
            return out

        # Etapa 2: Procesamiento IA (con caché independiente para no re-ejecutar al mover EQ/Ambiente)
        voice_after_ai = self._run_ai_stage(
            source,
            config,
            src_fp=src_fp,
            paths=app_paths,
            cancel_token=cancel_token,
            progress_cb=progress_cb,
        )

        if cancel_token:
            cancel_token.raise_if_cancelled()

        # Etapa 3: Cadena DSP activa + mezcla Dry/Wet según intensidad real
        if progress_cb:
            progress_cb(0.55, "Aplicando cadena DSP vocal...")

        if config.intensity > 0:
            dsp_voice = self._run_dsp_stage(voice_after_ai, config)
            restored_voice = dry_wet_mix(source, dsp_voice, config.intensity / 100.0)
        else:
            restored_voice = source.copy()

        if cancel_token:
            cancel_token.raise_if_cancelled()

        # Etapa 4: Simulación acústica y mezcla de capas ambientales
        if config.environment_enabled:
            if progress_cb:
                progress_cb(0.80, "Simulando ambiente acústico...")
            final_mix = self._run_environment_stage(
                restored_voice, config, paths=app_paths, cancel_token=cancel_token
            )
        else:
            final_mix = restored_voice

        # Etapa 5: Control estricto de picos y seguridad
        if progress_cb:
            progress_cb(0.95, "Control de picos de salida...")
        safe_out = apply_safety_limiter(final_mix, ceiling_db=config.true_peak_ceiling_db)
        self._full_cache.put(full_key, safe_out)
        if progress_cb:
            progress_cb(1.0, "Procesamiento completado")
        return safe_out

    def _run_ai_stage(
        self,
        source: AudioBuffer,
        config: PipelineConfig,
        *,
        src_fp: str,
        paths: AppPaths,
        cancel_token: CancellationToken | None,
        progress_cb: Callable[[float, str], None] | None,
    ) -> AudioBuffer:
        if not config.ai_enabled or config.intensity == 0:
            return source.copy()

        ai_key = config.ai_stage_cache_key(src_fp)
        cached_ai = self._ai_cache.get(ai_key)
        if cached_ai is not None:
            return cached_ai

        try:
            from models.model_registry import get_global_registry  # type: ignore[import-not-found]
        except ImportError:
            logger.debug("Módulo de modelos IA aún no instalado; ejecutando cadena DSP directa.")
            return source.copy()

        registry = get_global_registry(paths)
        adapter = registry.get_adapter(config.model_id)
        if adapter is None or not adapter.is_ready_for_inference():
            logger.info("Modelo '%s' no instalado/disponible localmente; continuando con DSP.", config.model_id)
            return source.copy()

        if progress_cb:
            progress_cb(0.15, f"Ejecutando modelo IA ({config.model_id})...")

        caps = adapter.get_capabilities()
        req_sr = caps.required_sample_rate
        work_buf = resample_audio(source, req_sr) if source.sample_rate != req_sr else source.copy()

        model_params = {
            k: v for k, v in config.advanced_params.items() if str(k).startswith("model_")
        }
        model_params["intensity_hint"] = config.intensity / 100.0

        processed = adapter.process(
            work_buf,
            model_params,
            progress_cb=progress_cb,
            cancel_token=cancel_token,
        )
        if processed.sample_rate != source.sample_rate or processed.num_samples != source.num_samples:
            processed = resample_audio(processed, source.sample_rate, target_length=source.num_samples)

        self._ai_cache.put(ai_key, processed)
        return processed

    @staticmethod
    def _run_dsp_stage(voice: AudioBuffer, config: PipelineConfig) -> AudioBuffer:
        p = config.advanced_params
        current = voice

        if bool(p.get("nr_enabled", True)):
            current = apply_spectral_denoise(
                current,
                amount=float(p.get("nr_amount", 0.50)),
                smoothing=float(p.get("nr_smoothing", 0.40)),
            )

        current = apply_equalizer(
            current,
            hp_freq=float(p.get("eq_hp_freq", 80.0)),
            lp_freq=float(p.get("eq_lp_freq", 18000.0)),
            mid_gain_db=float(p.get("eq_mid_gain", 0.0)),
        )

        if bool(p.get("gate_enabled", False)):
            current = apply_noise_gate(
                current,
                enabled=True,
                threshold_db=float(p.get("gate_threshold", -48.0)),
            )

        if bool(p.get("comp_enabled", True)):
            current = apply_compressor(
                current,
                enabled=True,
                threshold_db=float(p.get("comp_threshold", -18.0)),
                ratio=float(p.get("comp_ratio", 2.5)),
                attack_ms=float(p.get("comp_attack_ms", 12.0)),
                release_ms=float(p.get("comp_release_ms", 140.0)),
            )

        current = apply_vocal_treatment(
            current,
            presence_db=float(p.get("vocal_presence", 1.5)),
            clarity_db=float(p.get("vocal_clarity", 1.0)),
            deesser_amount=float(p.get("deesser_amount", 0.25)),
            saturation=float(p.get("vocal_saturation", 0.0)),
        )

        current = apply_spatial_stage(
            current,
            dereverb_amount=float(p.get("dereverb_amount", 0.0)),
            stereo_width=float(p.get("stereo_width", 1.0)),
            pan=float(p.get("spatial_pan", 0.0)),
        )
        return current

    @staticmethod
    def _run_environment_stage(
        voice: AudioBuffer,
        config: PipelineConfig,
        *,
        paths: AppPaths,
        cancel_token: CancellationToken | None,
    ) -> AudioBuffer:
        try:
            from environments.environment_engine import apply_environment_stage  # type: ignore[import-not-found]

            return apply_environment_stage(
                voice,
                config=config,
                paths=paths,
                cancel_token=cancel_token,
            )
        except ImportError:
            pass

        # Soporte funcional inmediato en Fase 4 para IR local y capas locales antes de Fase 6
        current = voice.copy()
        if config.ir_path and Path(config.ir_path).is_file():
            ir_buf, _ = load_audio(config.ir_path, paths=paths)
            if ir_buf.sample_rate != current.sample_rate:
                ir_buf = resample_audio(ir_buf, current.sample_rate)
            wet_mix = max(0.0, min(1.0, float(config.acoustics.get("reverb_mix", 28.0)) / 100.0))
            current = _convolve_ir_simple(current, ir_buf, wet_mix=wet_mix)

        loaded_layers: list[tuple[AudioBuffer, dict[str, Any]]] = []
        for layer_cfg in config.ambience_layers:
            fp = layer_cfg.get("file_path")
            if fp and Path(str(fp)).is_file():
                try:
                    lbuf, _ = load_audio(str(fp), paths=paths)
                    loaded_layers.append((lbuf, layer_cfg))
                except Exception as exc:
                    logger.warning("No se pudo cargar capa ambiental '%s': %s", fp, exc)

        if loaded_layers:
            current = mix_ambience_layers(
                current,
                loaded_layers,
                master_volume=config.master_ambience_volume,
            )
        return current


def _convolve_ir_simple(voice: AudioBuffer, ir: AudioBuffer, *, wet_mix: float) -> AudioBuffer:
    if wet_mix <= 0.01:
        return voice.copy()
    target_ch = max(voice.channels, ir.channels)
    v = voice.match_channels(target_ch)
    ir_m = ir.match_channels(target_ch)

    # Normalizar energía de la IR para evitar saltos extremos de ganancia
    norm = float(np.sqrt(np.sum(ir_m.samples ** 2) / float(max(1, ir_m.channels))))
    ir_norm = ir_m.samples / max(norm, 1e-6)

    wet_out = np.zeros_like(v.samples, dtype=np.float32)
    n = v.num_samples
    for ch in range(target_ch):
        conv = fftconvolve(v.samples[ch], ir_norm[ch], mode="full")[:n]
        wet_out[ch] = conv.astype(np.float32)

    wet_buf = AudioBuffer(samples=wet_out, sample_rate=v.sample_rate)
    return dry_wet_mix(v, wet_buf, wet_mix)
