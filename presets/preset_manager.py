"""Gestión de presets profesionales adaptativos y presets personalizados del usuario."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.exceptions import ValidationError
from core.paths import AppPaths
from core.validation import sanitize_filename

logger = logging.getLogger("zentralaudio.presets")


@dataclass(frozen=True)
class PresetResolution:
    """Resultado de adaptar un preset a las capacidades reales de un modelo IA."""

    preset_id: str
    name: str
    intensity: int
    params: dict[str, Any]
    notes: list[str]


class PresetManager:
    """Carga presets oficiales, los adapta por modelo y administra presets personalizados."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._builtin = self._load_builtin()

    def _load_builtin(self) -> list[dict[str, Any]]:
        path = self.paths.presets_dir / "restoration" / "builtin_presets.json"
        if not path.is_file():
            path = self.paths.bundle_root / "presets" / "restoration" / "builtin_presets.json"
        if not path.is_file():
            logger.warning("No se encontró builtin_presets.json en %s", path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [dict(item, is_custom=False) for item in raw.get("presets", [])]

    def list_all(self) -> list[dict[str, Any]]:
        items = [dict(p) for p in self._builtin]
        cdir = self.paths.custom_presets_dir
        if cdir.is_dir():
            for file in sorted(cdir.glob("*.json")):
                try:
                    data = json.loads(file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "name" in data:
                        items.append(
                            {
                                "preset_id": f"custom:{file.stem}",
                                "name": str(data["name"]),
                                "description": str(data.get("description", "Preset personalizado")),
                                "intensity": int(data.get("intensity", 75)),
                                "requires_dereverb": bool(data.get("requires_dereverb", False)),
                                "params": dict(data.get("params", {})),
                                "is_custom": True,
                                "file_path": str(file),
                            }
                        )
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Preset personalizado inválido %s: %s", file, exc)
        return items

    def get_by_id(self, preset_id: str) -> dict[str, Any] | None:
        for item in self.list_all():
            if item["preset_id"] == preset_id:
                return item
        return None

    def resolve_for_model(self, preset_id: str, model_id: str) -> PresetResolution:
        """Adapta parámetros del preset a las capacidades reales del modelo seleccionado."""
        preset = self.get_by_id(preset_id)
        if preset is None:
            raise ValidationError(f"Preset no encontrado: '{preset_id}'")

        params = dict(preset.get("params") or {})
        intensity = int(preset.get("intensity", 75))
        notes: list[str] = []

        # Consultar capacidades reales del modelo si Fase 5 está presente
        caps = None
        valid_model_keys: set[str] = set()
        try:
            from models.model_registry import get_global_registry

            adapter = get_global_registry(self.paths).get_adapter(model_id)
            if adapter is not None:
                caps = adapter.get_capabilities()
                valid_model_keys = {s.key for s in adapter.get_parameter_schema()}
        except ImportError:
            pass

        # Filtrar claves model_* que no pertenecen al modelo actual
        filtered: dict[str, Any] = {}
        for k, v in params.items():
            if str(k).startswith("model_") and valid_model_keys and k not in valid_model_keys:
                continue
            filtered[k] = v

        if caps is not None:
            if preset.get("requires_dereverb") and not caps.supports_dereverb:
                # Si el modelo no tiene dereverb nativo, reforzar módulo DSP compatible
                filtered["dereverb_amount"] = max(float(filtered.get("dereverb_amount", 0.0)), 0.65)
                notes.append(
                    f"{caps.display_name} no incluye reducción nativa de eco; se activó el módulo DSP de dereverb."
                )
            if caps.bandwidth_limit_hz and caps.bandwidth_limit_hz <= 8000:
                notes.append(
                    f"Limitación de {caps.display_name}: opera a {caps.required_sample_rate} Hz (hasta {caps.bandwidth_limit_hz} Hz)."
                )

        return PresetResolution(
            preset_id=preset_id,
            name=str(preset["name"]),
            intensity=intensity,
            params=filtered,
            notes=notes,
        )

    def save_custom(
        self,
        name: str,
        *,
        intensity: int,
        params: dict[str, Any],
        model_id: str = "",
        description: str = "",
    ) -> Path:
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError("El nombre del preset personalizado no puede estar vacío.")
        safe_stem = sanitize_filename(clean_name, fallback="custom_preset")
        self.paths.custom_presets_dir.mkdir(parents=True, exist_ok=True)
        target = self.paths.custom_presets_dir / f"{safe_stem}.json"
        payload = {
            "name": clean_name,
            "description": description or "Preset personalizado guardado por el usuario",
            "model_id": model_id,
            "intensity": int(intensity),
            "requires_dereverb": float(params.get("dereverb_amount", 0.0)) > 0.25,
            "params": dict(params),
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def delete_custom(self, preset_id: str) -> bool:
        if not preset_id.startswith("custom:"):
            raise ValidationError("Solo se pueden eliminar presets personalizados.")
        stem = preset_id.removeprefix("custom:")
        target = self.paths.custom_presets_dir / f"{stem}.json"
        if target.is_file():
            target.unlink()
            return True
        return False
