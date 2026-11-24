"""Interfaz común y contratos estrictos para todos los adaptadores de modelos IA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.job_manager import CancellationToken
from engine.audio_types import AudioBuffer


class ModelStatus(str, Enum):
    """Estado real verificable de un modelo IA en el equipo actual."""

    INSTALLED = "installed"
    NOT_INSTALLED = "not_installed"
    DOWNLOADING = "downloading"
    PREPARING = "preparing"
    AVAILABLE = "available"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"

    @property
    def display_label(self) -> str:
        labels = {
            ModelStatus.INSTALLED: "Instalado",
            ModelStatus.NOT_INSTALLED: "No instalado",
            ModelStatus.DOWNLOADING: "Descargando",
            ModelStatus.PREPARING: "Preparando",
            ModelStatus.AVAILABLE: "Disponible",
            ModelStatus.ERROR: "Error",
            ModelStatus.INCOMPATIBLE: "Incompatible / Incompleto",
        }
        return labels[self]


@dataclass(frozen=True)
class ModelCapabilities:
    """Capacidades reales declaradas por un modelo sin inventar prestaciones."""

    model_id: str
    display_name: str
    version: str
    required_sample_rate: int
    supported_channels: tuple[int, ...] = (1,)
    supports_cpu: bool = True
    supports_gpu: bool = True
    supports_native_wet_dry: bool = False
    supports_dereverb: bool = False
    bandwidth_limit_hz: int | None = None
    license_code: str = ""
    license_weights: str = ""
    attribution_required: bool = False
    source_url: str = ""
    estimated_ram_mb: int = 512
    limitations_note: str = ""


@dataclass(frozen=True)
class ParameterSpec:
    """Especificación real de un parámetro admitido por el adaptador."""

    key: str
    label: str
    param_type: str  # "float" | "int" | "bool" | "enum"
    default: Any
    min_val: float = 0.0
    max_val: float = 1.0
    step: float = 0.05
    decimals: int = 2
    unit: str = ""
    tooltip: str = ""
    options: list[tuple[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    """Resultado de validar un búfer de audio frente a las capacidades del modelo."""

    valid: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


class BaseModelAdapter(ABC):
    """Clase base obligatoria que todo adaptador de modelo IA debe implementar."""

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Devuelve capacidades verificadas y limitaciones reales del modelo."""

    @abstractmethod
    def get_parameter_schema(self) -> list[ParameterSpec]:
        """Devuelve exclusivamente los parámetros reales implementados por este modelo."""

    @abstractmethod
    def get_status(self) -> ModelStatus:
        """Comprueba en disco y entorno si el modelo puede ejecutarse realmente."""

    def get_status_detail(self) -> str:
        """Explica por qué el modelo está disponible, no instalado o incompatible."""
        return self.get_capabilities().limitations_note

    def is_ready_for_inference(self) -> bool:
        return self.get_status() in (ModelStatus.INSTALLED, ModelStatus.AVAILABLE)

    def validate_input(self, buffer: AudioBuffer) -> ValidationResult:
        caps = self.get_capabilities()
        warnings: list[str] = []
        if buffer.num_samples == 0:
            return ValidationResult(valid=False, reason="El audio de entrada está vacío.")
        if buffer.sample_rate != caps.required_sample_rate:
            warnings.append(
                f"El audio ({buffer.sample_rate} Hz) se convertirá a {caps.required_sample_rate} Hz para el modelo."
            )
        if caps.bandwidth_limit_hz and buffer.sample_rate > caps.required_sample_rate:
            warnings.append(
                f"Este modelo trabaja a {caps.required_sample_rate} Hz (techo ~{caps.bandwidth_limit_hz} Hz); "
                "no reconstruirá información por encima de ese límite."
            )
        return ValidationResult(valid=True, warnings=warnings)

    @abstractmethod
    def load_model(self, device: str = "auto") -> None:
        """Carga pesos y sesión de inferencia en el dispositivo solicitado."""

    @abstractmethod
    def unload_model(self) -> None:
        """Libera memoria RAM/VRAM asociada al modelo cargado."""

    @abstractmethod
    def process(
        self,
        buffer: AudioBuffer,
        params: dict[str, Any],
        *,
        progress_cb: Any = None,
        cancel_token: CancellationToken | None = None,
    ) -> AudioBuffer:
        """Ejecuta la inferencia real sobre el búfer validado."""
