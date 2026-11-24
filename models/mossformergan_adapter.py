"""Adaptador real para MossFormerGAN_SE_16K (ClearerVoice-Studio, Apache-2.0)."""

from __future__ import annotations

import gc
import importlib.util
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


class MossFormerGANAdapter(BaseModelAdapter):
    """Adaptador para alibabasglab/MossFormerGAN_SE_16K (16 kHz mono)."""

    MODEL_ID = "mossformergan_se_16k"
    CV_MODEL_NAME = "MossFormerGAN_SE_16K"

    def __init__(self, paths: AppPaths, manager: ModelManager) -> None:
        self.paths = paths
        self.manager = manager
        self._cv_instance: Any = None
        self._tmp_workdir: Path | None = None

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self.MODEL_ID,
            display_name="MossFormerGAN_SE_16K",
            version="ClearerVoice",
            required_sample_rate=16000,
            supported_channels=(1,),
            supports_cpu=True,
            supports_gpu=True,
            supports_native_wet_dry=False,
            supports_dereverb=False,
            bandwidth_limit_hz=8000,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            attribution_required=False,
            source_url="https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K",
            estimated_ram_mb=1100,
            limitations_note=(
                "Trabaja a 16 kHz (techo físico 8 kHz); no recupera agudos sobre 8 kHz. "
                "Pesos sin SHA-256 oficial publicado en origen."
            ),
        )

    def get_parameter_schema(self) -> list[ParameterSpec]:
        return []

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
            return "Pesos (~136 MB) no descargados."
        if st == ModelStatus.INCOMPATIBLE:
            return "Falta el runtime 'clearvoice' o entorno aislado en models_data/mossformergan_se_16k/.venv."
        return self.get_capabilities().limitations_note

    def load_model(self, device: str = "auto") -> None:
        if not self.manager.is_weights_present(self.MODEL_ID):
            raise ModelNotInstalledError("MossFormerGAN_SE_16K no está descargado.")
        if find_isolated_python(self.manager.model_dir(self.MODEL_ID)) is not None:
            return
        if self._cv_instance is not None:
            return
        try:
            from clearvoice import ClearVoice  # type: ignore[import-not-found]
        except Exception as exc:
            raise ModelLoadError("El paquete 'clearvoice' no está instalado.", details=str(exc)) from exc

        work = Path(tempfile.mkdtemp(prefix="zentral_mfgan_"))
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
            raise ModelLoadError("No se pudo inicializar MossFormerGAN_SE_16K.", details=str(exc)) from exc
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

        iso_py = find_isolated_python(self.manager.model_dir(self.MODEL_ID))
        worker = Path(__file__).resolve().parent / "isolated_runners" / "worker_entry.py"
        out_channels = np.zeros_like(buffer.samples, dtype=np.float32)

        for ch in range(buffer.channels):
            mono = AudioBuffer(samples=buffer.samples[ch : ch + 1], sample_rate=16000)
            if iso_py is not None:
                res = run_in_isolated_subprocess(
                    python_exe=iso_py,
                    worker_script=worker,
                    buffer=mono,
                    model_id=self.MODEL_ID,
                    weights_dir=self.manager.model_dir(self.MODEL_ID),
                    device=str(params.get("device", "cpu")),
                    params=params,
                    cancel_token=cancel_token,
                )
                out_channels[ch] = res.samples[0, : buffer.num_samples]
            else:
                self.load_model()
                with tempfile.TemporaryDirectory(prefix="zentral_mfgan_run_") as tmp_dir:
                    in_wav = Path(tmp_dir) / "in.wav"
                    sf.write(str(in_wav), mono.samples[0], 16000, subtype="FLOAT")
                    prev_cwd = os.getcwd()
                    if self._tmp_workdir:
                        os.chdir(str(self._tmp_workdir))
                    try:
                        raw_out = self._cv_instance(input_path=str(in_wav), online_write=False)
                    except Exception as exc:
                        raise ModelInferenceError("Error en MossFormerGAN_SE_16K.", details=str(exc)) from exc
                    finally:
                        os.chdir(prev_cwd)
                    arr = np.asarray(raw_out, dtype=np.float32).reshape(-1)
                    out_channels[ch] = arr[: buffer.num_samples]

        return AudioBuffer(samples=out_channels, sample_rate=16000)
