"""Adaptador real para MossFormer2_SE_48K (ClearerVoice-Studio, Apache-2.0)."""

from __future__ import annotations

import gc
import importlib.util
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from core.exceptions import ModelInferenceError, ModelLoadError, ModelNotInstalledError
from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_types import AudioBuffer
from models.base_adapter import BaseModelAdapter, ModelCapabilities, ModelStatus, ParameterSpec
from models.isolated_runners.subprocess_runner import find_isolated_python, run_in_isolated_subprocess
from models.model_manager import ModelManager

logger = logging.getLogger("zentralaudio.models.mossformer2")


class MossFormer2Adapter(BaseModelAdapter):
    """Adaptador para alibabasglab/MossFormer2_SE_48K con segmentación solapada segura."""

    MODEL_ID = "mossformer2_se_48k"
    CV_MODEL_NAME = "MossFormer2_SE_48K"

    def __init__(self, paths: AppPaths, manager: ModelManager) -> None:
        self.paths = paths
        self.manager = manager
        self._cv_instance: Any = None
        self._tmp_workdir: Path | None = None

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self.MODEL_ID,
            display_name="MossFormer2_SE_48K",
            version="ClearerVoice",
            required_sample_rate=48000,
            supported_channels=(1,),
            supports_cpu=True,
            supports_gpu=True,
            supports_native_wet_dry=False,
            supports_dereverb=True,
            bandwidth_limit_hz=24000,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            attribution_required=False,
            source_url="https://huggingface.co/alibabasglab/MossFormer2_SE_48K",
            estimated_ram_mb=1400,
            limitations_note=(
                "Modelo de 48 kHz mono (en estéreo procesa canales de forma independiente). "
                "Aplica segmentación por bloques con solapamiento para evitar el límite de 20s del motor upstream."
            ),
        )

    def get_parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                key="model_chunk_seconds",
                label="Duración de segmento (MossFormer2)",
                param_type="int",
                default=12,
                min_val=6,
                max_val=18,
                step=2,
                unit="s",
                tooltip="Divide audios largos en bloques seguros de hasta 18 s con solapamiento cruzado.",
            )
        ]

    def _has_runtime(self) -> bool:
        if importlib.util.find_spec("clearvoice") is not None:
            return True
        return find_isolated_python(self.manager.model_dir(self.MODEL_ID)) is not None

    def get_status(self) -> ModelStatus:
        if self.manager.has_incomplete_files(self.MODEL_ID):
            return ModelStatus.ERROR
        if not self.manager.is_weights_present(self.MODEL_ID):
            return ModelStatus.NOT_INSTALLED
        if not self._has_runtime():
            return ModelStatus.INCOMPATIBLE
        return ModelStatus.AVAILABLE

    def get_status_detail(self) -> str:
        st = self.get_status()
        if st == ModelStatus.NOT_INSTALLED:
            return "Pesos (~222 MB, SHA-256 verificado) no descargados."
        if st == ModelStatus.INCOMPATIBLE:
            return (
                "Pesos presentes, pero falta el runtime 'clearvoice' o un entorno aislado en "
                "models_data/mossformer2_se_48k/.venv."
            )
        return self.get_capabilities().limitations_note

    def load_model(self, device: str = "auto") -> None:
        if not self.manager.is_weights_present(self.MODEL_ID):
            raise ModelNotInstalledError("MossFormer2_SE_48K no está descargado en models_data/.")
        if find_isolated_python(self.manager.model_dir(self.MODEL_ID)) is not None:
            return
        if self._cv_instance is not None:
            return
        try:
            from clearvoice import ClearVoice  # type: ignore[import-not-found]
        except Exception as exc:
            raise ModelLoadError(
                "El paquete 'clearvoice' no está instalado en este entorno.",
                details=str(exc),
            ) from exc

        work = Path(tempfile.mkdtemp(prefix="zentral_mf2_"))
        ckpt_dir = work / "checkpoints" / self.CV_MODEL_NAME
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for item in self.manager.model_dir(self.MODEL_ID).iterdir():
            if item.is_file() and not item.name.startswith("."):
                shutil.copy2(item, ckpt_dir / item.name)

        prev_cwd = os.getcwd()
        os.chdir(str(work))
        try:
            self._cv_instance = ClearVoice(task="speech_enhancement", model_names=[self.CV_MODEL_NAME])
            self._tmp_workdir = work
        except Exception as exc:
            shutil.rmtree(work, ignore_errors=True)
            raise ModelLoadError("No se pudo inicializar MossFormer2_SE_48K.", details=str(exc)) from exc
        finally:
            os.chdir(prev_cwd)

    def unload_model(self) -> None:
        self._cv_instance = None
        if self._tmp_workdir is not None:
            shutil.rmtree(self._tmp_workdir, ignore_errors=True)
            self._tmp_workdir = None
        gc.collect()

    def process(
        self,
        buffer: AudioBuffer,
        params: dict[str, Any],
        *,
        progress_cb: Any = None,
        cancel_token: CancellationToken | None = None,
    ) -> AudioBuffer:
        if cancel_token:
            cancel_token.raise_if_cancelled()

        chunk_sec = int(params.get("model_chunk_seconds", 12))
        chunk_samples = max(48000 * 4, min(48000 * 18, chunk_sec * 48000))
        overlap = 24000  # 0.5 s crossfade

        iso_py = find_isolated_python(self.manager.model_dir(self.MODEL_ID))
        worker = Path(__file__).resolve().parent / "isolated_runners" / "worker_entry.py"

        if iso_py is not None:
            return self._process_chunked(
                buffer,
                chunk_samples=chunk_samples,
                overlap=overlap,
                infer_fn=lambda mono_chunk: run_in_isolated_subprocess(
                    python_exe=iso_py,
                    worker_script=worker,
                    buffer=mono_chunk,
                    model_id=self.MODEL_ID,
                    weights_dir=self.manager.model_dir(self.MODEL_ID),
                    device=str(params.get("device", "cpu")),
                    params=params,
                    cancel_token=cancel_token,
                ),
                progress_cb=progress_cb,
                cancel_token=cancel_token,
            )

        self.load_model()
        return self._process_chunked(
            buffer,
            chunk_samples=chunk_samples,
            overlap=overlap,
            infer_fn=self._infer_inprocess_mono,
            progress_cb=progress_cb,
            cancel_token=cancel_token,
        )

    def _infer_inprocess_mono(self, mono_buf: AudioBuffer) -> AudioBuffer:
        with tempfile.TemporaryDirectory(prefix="zentral_mf2_run_") as tmp_dir:
            in_wav = Path(tmp_dir) / "chunk.wav"
            sf.write(str(in_wav), mono_buf.samples[0], 48000, subtype="FLOAT")
            prev_cwd = os.getcwd()
            if self._tmp_workdir:
                os.chdir(str(self._tmp_workdir))
            try:
                out = self._cv_instance(input_path=str(in_wav), online_write=False)
            except Exception as exc:
                raise ModelInferenceError("Error en inferencia de MossFormer2_SE_48K.", details=str(exc)) from exc
            finally:
                os.chdir(prev_cwd)
            arr = np.asarray(out, dtype=np.float32).reshape(1, -1)
            return AudioBuffer(samples=arr, sample_rate=48000)

    def _process_chunked(
        self,
        buffer: AudioBuffer,
        *,
        chunk_samples: int,
        overlap: int,
        infer_fn: Any,
        progress_cb: Any,
        cancel_token: CancellationToken | None,
    ) -> AudioBuffer:
        out_channels = np.zeros_like(buffer.samples, dtype=np.float32)
        total_len = buffer.num_samples
        step = max(1, chunk_samples - overlap)

        for ch in range(buffer.channels):
            x = buffer.samples[ch]
            if total_len <= chunk_samples:
                res = infer_fn(AudioBuffer(samples=x[np.newaxis, :], sample_rate=48000))
                out_channels[ch] = res.samples[0, :total_len]
                continue

            acc = np.zeros(total_len, dtype=np.float32)
            weights = np.zeros(total_len, dtype=np.float32)
            pos = 0
            while pos < total_len:
                if cancel_token:
                    cancel_token.raise_if_cancelled()
                end = min(total_len, pos + chunk_samples)
                seg = x[pos:end]
                res = infer_fn(AudioBuffer(samples=seg[np.newaxis, :], sample_rate=48000))
                seg_out = res.samples[0, : seg.shape[0]]

                w = np.ones(seg_out.shape[0], dtype=np.float32)
                cf = min(overlap, seg_out.shape[0] // 3)
                if cf > 8:
                    if pos > 0:
                        w[:cf] = np.linspace(0.0, 1.0, cf, dtype=np.float32)
                    if end < total_len:
                        w[-cf:] = np.linspace(1.0, 0.0, cf, dtype=np.float32)

                acc[pos:end] += seg_out * w
                weights[pos:end] += w
                if progress_cb:
                    progress_cb(0.2 + 0.3 * (end / float(total_len)), "Procesando MossFormer2_SE_48K...")
                if end >= total_len:
                    break
                pos += step

            out_channels[ch] = acc / np.maximum(weights, 1e-6)

        return AudioBuffer(samples=out_channels, sample_rate=48000)
