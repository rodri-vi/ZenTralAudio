"""Adaptador honesto para GTCRN (Xiaobin-Rong/gtcrn, MIT) documentando verificación pendiente."""

from __future__ import annotations

from typing import Any

from core.exceptions import ModelLoadError, ModelNotInstalledError
from core.job_manager import CancellationToken
from core.paths import AppPaths
from engine.audio_types import AudioBuffer
from models.base_adapter import BaseModelAdapter, ModelCapabilities, ModelStatus, ParameterSpec
from models.model_manager import ModelManager


class GTCRNAdapter(BaseModelAdapter):
    """Adaptador reservado para GTCRN hasta verificar checkpoints y código fuente exacto."""

    MODEL_ID = "gtcrn"

    def __init__(self, paths: AppPaths, manager: ModelManager) -> None:
        self.paths = paths
        self.manager = manager

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_id=self.MODEL_ID,
            display_name="GTCRN",
            version="pendiente-verificación",
            required_sample_rate=16000,
            supported_channels=(1,),
            supports_cpu=True,
            supports_gpu=True,
            supports_native_wet_dry=False,
            supports_dereverb=False,
            bandwidth_limit_hz=8000,
            license_code="MIT",
            license_weights="Pendiente de verificación en Xiaobin-Rong/gtcrn",
            attribution_required=True,
            source_url="https://github.com/Xiaobin-Rong/gtcrn",
            estimated_ram_mb=90,
            limitations_note=(
                "No disponible para inferencia automática en esta versión: no se han podido verificar "
                "los nombres exactos de archivo ni hashes de checkpoints en Xiaobin-Rong/gtcrn/checkpoints/ "
                "ni existe paquete PyPI oficial. No se muestran parámetros ni descargas inventadas."
            ),
        )

    def get_parameter_schema(self) -> list[ParameterSpec]:
        return []

    def get_status(self) -> ModelStatus:
        # Honestidad estricta: nunca reportar como disponible sin runtime y pesos verificables
        return ModelStatus.INCOMPATIBLE

    def get_status_detail(self) -> str:
        return self.get_capabilities().limitations_note

    def load_model(self, device: str = "auto") -> None:
        raise ModelNotInstalledError(
            "GTCRN no está habilitado automáticamente.",
            details=self.get_capabilities().limitations_note,
        )

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
        raise ModelLoadError(
            "GTCRN no puede ejecutarse porque faltan los pesos y definición verificada de arquitectura.",
            details=self.get_capabilities().limitations_note,
        )
