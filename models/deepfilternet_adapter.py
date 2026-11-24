"""Adaptador real para DeepFilterNet3 (48 kHz, compatible con CPU/GPU)."""

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from core.exceptions import ModelInferenceError, ModelLoadError, ModelNotInstalledError
from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_types import AudioBuffer
from models.base_adapter import BaseModelAdapter, ModelCapabilities, ModelStatus, ParameterSpec
from models.model_manager import ModelManager

logger = logging.getLogger("zentralaudio.models.deepfilternet3")


class _PythonLoggingStderr:
    """Sink compatible con stderr que reenvía los mensajes al log portable."""

    encoding = "utf-8"
    errors = "replace"

    def write(self, message: str) -> int:
        text = str(message)
        for line in text.splitlines():
            if line.strip():
                logger.info("DeepFilterNet: %s", line.rstrip())
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        # Loguru puede conservar este stream después de restaurar sys.stderr.
        # No se cierra porque el adaptador y sus sinks comparten su ciclo de vida.
        return None


def _call_with_safe_stderr(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Ejecuta una inicialización de terceros aunque el EXE no tenga consola."""
    previous_stderr = sys.stderr
    if previous_stderr is not None:
        return function(*args, **kwargs)

    safe_stderr = _PythonLoggingStderr()
    sys.stderr = safe_stderr
    try:
        return function(*args, **kwargs)
    finally:
        if sys.stderr is safe_stderr:
            sys.stderr = previous_stderr


def _ensure_torchaudio_compat_shim() -> None:
    """Proporciona shim seguro si torchaudio eliminó torchaudio.backend.common."""
    import torchaudio

    if "torchaudio.backend" not in sys.modules:
        backend_mod = types.ModuleType("torchaudio.backend")
        common_mod = types.ModuleType("torchaudio.backend.common")

        class AudioMetaData:  # noqa: D401
            def __init__(self, sample_rate: int = 48000, num_frames: int = 0, num_channels: int = 1) -> None:
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels

        common_mod.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
        backend_mod.common = common_mod  # type: ignore[attr-defined]
        sys.modules["torchaudio.backend"] = backend_mod
        sys.modules["torchaudio.backend.common"] = common_mod
        if not hasattr(torchaudio, "backend"):
            torchaudio.backend = backend_mod  # type: ignore[attr-defined]


class DeepFilterNet3Adapter(BaseModelAdapter):
    """Integración real con Rikorose/DeepFilterNet (df.enhance)."""

    MODEL_ID = "deepfilternet3"

    def __init__(self, paths: AppPaths, manager: ModelManager) -> None:
        self.paths = paths
        self.manager = manager
        self._model: Any = None
        self._df_state: Any = None
        self._loaded_pf: bool | None = None
        self._device: str = "cpu"

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self.MODEL_ID,
            display_name="DeepFilterNet3",
            version="0.5.6",
            required_sample_rate=48000,
            supported_channels=(1, 2),
            supports_cpu=True,
            supports_gpu=True,
            supports_native_wet_dry=True,
            supports_dereverb=False,
            bandwidth_limit_hz=24000,
            license_code="MIT / Apache-2.0",
            license_weights="MIT / Apache-2.0 (repo Rikorose/DeepFilterNet)",
            attribution_required=False,
            source_url="https://github.com/Rikorose/DeepFilterNet",
            estimated_ram_mb=280,
            limitations_note=(
                "48 kHz nativo, muy rápido en CPU. Utiliza 'atten_lim_db' como límite real de supresión "
                "y requiere el paquete 'deepfilternet==0.5.6' con 'numpy<2.0'."
            ),
        )

    def get_parameter_schema(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                key="model_atten_lim_db",
                label="Límite de atenuación (DeepFilterNet3)",
                param_type="float",
                default=100.0,
                min_val=6.0,
                max_val=100.0,
                step=2.0,
                decimals=0,
                unit="dB",
                tooltip="Máxima reducción de ruido permitida en dB (100 dB = supresión completa sin límite).",
            ),
            ParameterSpec(
                key="model_post_filter",
                label="Post-filtro espectral (Post-Filter)",
                param_type="bool",
                default=False,
                tooltip="Activa suavizado adicional en tramos muy ruidosos; puede atenuar ligeramente transitorios.",
            ),
            ParameterSpec(
                key="model_compensate_delay",
                label="Compensar retardo STFT (pad)",
                param_type="bool",
                default=True,
                tooltip="Alinea temporalmente la señal procesada con la original para mezcla sin desfase.",
            ),
        ]

    def _find_extracted_model_dir(self) -> Path | None:
        base = self.manager.model_dir(self.MODEL_ID)
        if not base.is_dir():
            return None
        if (base / "config.ini").is_file():
            return base
        nested = base / "DeepFilterNet3"
        if (nested / "config.ini").is_file():
            return nested
        for candidate in base.rglob("config.ini"):
            return candidate.parent
        return None

    def get_status(self) -> ModelStatus:
        if self.manager.has_incomplete_files(self.MODEL_ID):
            return ModelStatus.ERROR
        weights_dir = self._find_extracted_model_dir()
        pkg_ok = importlib.util.find_spec("df") is not None
        if weights_dir is None:
            return ModelStatus.NOT_INSTALLED
        if not pkg_ok:
            return ModelStatus.INCOMPATIBLE
        return ModelStatus.AVAILABLE

    def get_status_detail(self) -> str:
        status = self.get_status()
        if status == ModelStatus.NOT_INSTALLED:
            return "Pesos no descargados. Puedes descargar DeepFilterNet3 (~8 MB, SHA-256 verificado) desde Gestor de Modelos."
        if status == ModelStatus.INCOMPATIBLE:
            return "Pesos presentes, pero falta el paquete Python 'deepfilternet==0.5.6' (pip install deepfilternet==0.5.6)."
        return self.get_capabilities().limitations_note

    def load_model(self, device: str = "auto", *, post_filter: bool = False) -> None:
        weights_dir = self._find_extracted_model_dir()
        if weights_dir is None:
            raise ModelNotInstalledError("Los pesos de DeepFilterNet3 no están descargados.")

        try:
            import torch

            _ensure_torchaudio_compat_shim()
            from df.enhance import init_df  # type: ignore[import-not-found]
        except Exception as exc:
            raise ModelLoadError(
                "No se pudo importar 'deepfilternet' (df.enhance).",
                details=f"Instala deepfilternet==0.5.6 con numpy<2.0 y torchaudio<=2.8. Detalle: {exc}",
            ) from exc

        target_dev = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        if self._model is not None and self._loaded_pf == post_filter and self._device == target_dev:
            return

        self.unload_model()
        try:
            model, df_state, _ = _call_with_safe_stderr(
                init_df,
                model_base_dir=str(weights_dir),
                post_filter=bool(post_filter),
            )
            model = model.to(target_dev)
            model.eval()
            self._model = model
            self._df_state = df_state
            self._loaded_pf = bool(post_filter)
            self._device = target_dev
            logger.info("DeepFilterNet3 cargado en %s (post_filter=%s)", target_dev, post_filter)
        except Exception as exc:
            raise ModelLoadError("Error al inicializar DeepFilterNet3.", details=str(exc)) from exc

    def unload_model(self) -> None:
        self._model = None
        self._df_state = None
        self._loaded_pf = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

        pf = bool(params.get("model_post_filter", False))
        pad = bool(params.get("model_compensate_delay", True))
        raw_atten = float(params.get("model_atten_lim_db", 100.0))
        atten_lim = None if raw_atten >= 99.0 else max(6.0, raw_atten)

        self.load_model(device=str(params.get("device", "auto")), post_filter=pf)

        try:
            import torch
            from df.enhance import enhance  # type: ignore[import-not-found]

            if progress_cb:
                progress_cb(0.25, "Procesando voz con DeepFilterNet3...")

            audio_t = torch.from_numpy(buffer.samples.copy())
            with torch.no_grad():
                enhanced = enhance(
                    self._model,
                    self._df_state,
                    audio_t,
                    pad=pad,
                    atten_lim_db=atten_lim,
                )
            out_np = np.asarray(enhanced.detach().cpu().numpy(), dtype=np.float32)
            if out_np.ndim == 1:
                out_np = out_np[np.newaxis, :]
            return AudioBuffer(samples=out_np[:, : buffer.num_samples], sample_rate=48000)
        except Exception as exc:
            raise ModelInferenceError("Fallo durante la inferencia con DeepFilterNet3.", details=str(exc)) from exc
