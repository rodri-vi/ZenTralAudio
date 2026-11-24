"""Registro central de adaptadores IA y puente hacia la UI y el motor de audio."""

from __future__ import annotations

import threading
from typing import Any

from app.widgets.dynamic_param_panel import UIParamDefinition
from core.paths import AppPaths
from models.base_adapter import BaseModelAdapter
from models.deepfilternet_adapter import DeepFilterNet3Adapter
from models.frcrn_adapter import FRCRNAdapter
from models.gtcrn_adapter import GTCRNAdapter
from models.model_manager import ModelManager
from models.mossformer2_adapter import MossFormer2Adapter
from models.mossformergan_adapter import MossFormerGANAdapter


class ModelRegistry:
    """Agrupa todos los adaptadores soportados por ZenTralAudio sin acoplar el motor."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.manager = ModelManager(paths)
        self._adapters: dict[str, BaseModelAdapter] = {}
        for adapter in (
            DeepFilterNet3Adapter(paths, self.manager),
            MossFormer2Adapter(paths, self.manager),
            MossFormerGANAdapter(paths, self.manager),
            FRCRNAdapter(paths, self.manager),
            VoiceFixerAdapter(paths, self.manager),
            GTCRNAdapter(paths, self.manager),
        ):
            self._adapters[adapter.get_capabilities().model_id] = adapter

    def get_adapter(self, model_id: str) -> BaseModelAdapter | None:
        return self._adapters.get(model_id)

    def list_adapters(self) -> list[BaseModelAdapter]:
        return list(self._adapters.values())

    def list_ui_summaries(self) -> list[dict[str, Any]]:
        """Genera metadatos reales y estado en tiempo real para selector y Gestor de Modelos."""
        summaries: list[dict[str, Any]] = []
        for adapter in self._adapters.values():
            caps = adapter.get_capabilities()
            status = adapter.get_status()
            entry = self.manager.get_entry(caps.model_id)
            total_mb = sum(f.approx_size_mb for f in entry.files) if entry else 0
            summaries.append(
                {
                    "model_id": caps.model_id,
                    "display_name": caps.display_name,
                    "sample_rate": caps.required_sample_rate,
                    "license_summary": f"{caps.license_code} / {caps.license_weights}",
                    "size_label": f"~{total_mb} MB" if total_mb > 0 else "N/D",
                    "source_url": caps.source_url,
                    "status_code": status.value,
                    "status_label": status.display_label,
                    "download_supported": bool(entry and entry.download_supported),
                    "note": adapter.get_status_detail(),
                }
            )
        return summaries

    def get_ui_schema_for_model(self, model_id: str) -> list[UIParamDefinition]:
        """Convierte los parámetros reales del modelo en definiciones dinámicas de interfaz."""
        adapter = self.get_adapter(model_id)
        if adapter is None:
            return []
        defs: list[UIParamDefinition] = []
        for spec in adapter.get_parameter_schema():
            defs.append(
                UIParamDefinition(
                    key=spec.key,
                    label=spec.label,
                    category="Modelo IA",
                    param_type=spec.param_type,
                    default=spec.default,
                    min_val=spec.min_val,
                    max_val=spec.max_val,
                    step=spec.step,
                    decimals=spec.decimals,
                    unit=spec.unit,
                    tooltip=spec.tooltip,
                    options=list(spec.options),
                )
            )
        return defs

    def unload_all(self) -> None:
        for adapter in self._adapters.values():
            adapter.unload_model()


_REGISTRY_LOCK = threading.Lock()
_REGISTRIES: dict[str, ModelRegistry] = {}


def get_global_registry(paths: AppPaths) -> ModelRegistry:
    """Devuelve la instancia única del registro para la raíz portable dada."""
    key = str(paths.app_root.resolve())
    with _REGISTRY_LOCK:
        reg = _REGISTRIES.get(key)
        if reg is None:
            reg = ModelRegistry(paths)
            _REGISTRIES[key] = reg
        return reg
