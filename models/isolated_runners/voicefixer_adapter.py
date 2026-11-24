"""Adaptador real para VoiceFixer (44.1 kHz, MIT / pesos CC-BY-4.0 con SHA-256 verificado)."""

from __future__ import annotations

import gc
import importlib.util
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


class VoiceFixerAdapter(BaseModelAdapter):
    """Integración con haoheliu/voicefixer priorizando ejecución aislada y pesos locales."""

    MODEL_ID = "voicefixer"

    def __init__(self, paths: AppPaths, manager: ModelManager) -> None:
        self.paths = paths
        self.manager = manager
        self._vf: Any = None

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self.MODEL_ID,
            display_name="VoiceFixer",
            version="0.1.3",
            required_sample_rate=44100,
            supported_channels=(1,),
            supports_cpu=True,
            supports_gpu=True,
            supports_native_wet_dry=False,
            supports_dereverb=True,
            bandwidth_limit_hz=22050,
            license_code="MIT",
            license_weights="CC-BY-4.0 (Zenodo 5600188, requiere atribución a Haohe Liu et al.)",
            attribution_required=True,
            source_url="https://github.com/haoheliu/voicefixer",
            estimated_ram_mb=1800,
            limitations_note=(
                "Usa resíntesis neuronal por vocoder a 44.1 kHz: puede rescatar voces muy dañadas o "
                "antiguas, pero puede alterar sutilmente el timbre en grabaciones ya limpias. "
                "Recomendado en entorno aislado por dependencias antiguas."
            ),
        )

    def get_parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                key="model_vf_mode",
                label="Modo de restauración (VoiceFixer)",
                param_type="enum",
                default=0,
                options=[
                    ("Modo 0 — Restauración estándar", 0),
                    ("Modo 1 — Preprocesar altas frecuencias", 1),
                    ("Modo 2 — Degradación severa / entrenamiento", 2),
                ],
                tooltip="Modos reales oficiales (0, 1, 2) expuestos por VoiceFixer.restore().",
            )
        ]

    def _has_runtime(self) -> bool:
        if importlib.util.find_spec("voicefixer") is not None:
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
            return "Checkpoints (~597 MB, SHA-256 oficial verificado en Zenodo) no descargados."
        if st == ModelStatus.INCOMPATIBLE:
            return (
                "Checkpoints presentes, pero falta el paquete 'voicefixer' o entorno aislado en "
                "models_data/voicefixer/.venv."
            )
        return self.get_capabilities().limitations_note

    def load_model(self, device: str = "auto") -> None:
        if not self.manager.is_weights_present(self.MODEL_ID):
            raise ModelNotInstalledError("Los pesos de VoiceFixer no están descargados.")
        if find_isolated_python(self.manager.model_dir(self.MODEL_ID)) is not None:
            return
        if self._vf is not None:
            return

        # Pre-copiar checkpoints verificados a caché de voicefixer antes de instanciar
        wdir = self.manager.model_dir(self.MODEL_ID)
        cache_root = Path.home() / ".cache" / "voicefixer"
        am_dir = cache_root / "analysis_module" / "checkpoints"
        voc_dir = cache_root / "synthesis_module" / "44100"
        am_dir.mkdir(parents=True, exist_ok=True)
        voc_dir.mkdir(parents=True, exist_ok=True)
        if not (am_dir / "vf.ckpt").exists():
            shutil.copy2(wdir / "vf.ckpt", am_dir / "vf.ckpt")
        if not (voc_dir / "model.ckpt-1490000_trimed.pt").exists():
            shutil.copy2(wdir / "model.ckpt-1490000_trimed.pt", voc_dir / "model.ckpt-1490000_trimed.pt")

        try:
            from voicefixer import VoiceFixer  # type: ignore[import-not-found]

            self._vf = VoiceFixer()
        except Exception as exc:
            raise ModelLoadError("No se pudo cargar VoiceFixer en este entorno.", details=str(exc)) from exc

    def unload_model(self) -> None:
        self._vf = None
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

        mode = int(params.get("model_vf_mode", 0))
        iso_py = find_isolated_python(self.manager.model_dir(self.MODEL_ID))
        worker = Path(__file__).resolve().parent / "isolated_runners" / "worker_entry.py"

        import torch

        use_cuda = str(params.get("device", "auto")) in ("auto", "cuda") and torch.cuda.is_available()
        out_channels = np.zeros_like(buffer.samples, dtype=np.float32)

        for ch in range(buffer.channels):
            mono = AudioBuffer(samples=buffer.samples[ch : ch + 1], sample_rate=44100)
            if iso_py is not None:
                res = run_in_isolated_subprocess(
                    python_exe=iso_py,
                    worker_script=worker,
                    buffer=mono,
                    model_id=self.MODEL_ID,
                    weights_dir=self.manager.model_dir(self.MODEL_ID),
                    device="cuda" if use_cuda else "cpu",
                    params={"model_vf_mode": mode},
                    cancel_token=cancel_token,
                )
                out_channels[ch] = res.samples[0, : buffer.num_samples]
            else:
                self.load_model()
                with tempfile.TemporaryDirectory(prefix="zentral_vf_run_") as tmp_dir:
                    in_wav = Path(tmp_dir) / "in.wav"
                    out_wav = Path(tmp_dir) / "out.wav"
                    sf.write(str(in_wav), mono.samples[0], 44100, subtype="FLOAT")
                    try:
                        self._vf.restore(input=str(in_wav), output=str(out_wav), cuda=use_cuda, mode=mode)
                        out_data, _ = sf.read(str(out_wav), dtype="float32", always_2d=True)
                    except Exception as exc:
                        raise ModelInferenceError("Error en inferencia de VoiceFixer.", details=str(exc)) from exc
                    out_channels[ch] = out_data[:, 0][: buffer.num_samples]

        return AudioBuffer(samples=out_channels, sample_rate=44100)
