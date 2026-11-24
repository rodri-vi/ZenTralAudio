"""Persistencia tipada y atómica de la configuración portable de ZenTralAudio."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.exceptions import ConfigurationError
from core.validation import validate_numeric_range

logger = logging.getLogger("zentralaudio.config")

CONFIG_SCHEMA_VERSION = 1
CONFIG_FILENAME = "settings.json"
VALID_COMPUTE_DEVICES = ("auto", "cpu", "cuda")


@dataclass
class AppSettings:
    """Preferencias generales persistentes de ZenTralAudio."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    preferred_device: str = "auto"  # auto | cpu | cuda
    max_worker_threads: int = 2
    output_device_index: int | None = None
    loudness_match_preview: bool = True
    preview_monitor_volume: float = 0.85
    ffmpeg_custom_path: str | None = None
    last_open_dir: str | None = None
    last_export_dir: str | None = None
    last_selected_model: str = "deepfilternet3"
    consent_external_downloads: bool = False
    debug_logging: bool = False
    recent_files: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Valida invariantes de configuración antes de usar o persistir."""
        if self.preferred_device not in VALID_COMPUTE_DEVICES:
            raise ConfigurationError(
                f"Dispositivo de cómputo inválido en configuración: '{self.preferred_device}'."
            )
        self.max_worker_threads = int(
            validate_numeric_range(
                self.max_worker_threads,
                name="max_worker_threads",
                min_value=1,
                max_value=16,
            )
        )
        self.preview_monitor_volume = validate_numeric_range(
            self.preview_monitor_volume,
            name="preview_monitor_volume",
            min_value=0.0,
            max_value=1.5,
        )
        self.recent_files = [str(p) for p in self.recent_files[:10] if isinstance(p, str)]


class ConfigManager:
    """Gestiona la carga, actualización y guardado atómico de settings.json."""

    def __init__(self, config_dir: Path, filename: str = CONFIG_FILENAME) -> None:
        self.config_dir = config_dir
        self.config_path = config_dir / filename
        self._settings = AppSettings()

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def load(self) -> AppSettings:
        """Carga la configuración desde disco o crea el archivo inicial si no existe."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            logger.info("Archivo de configuración no encontrado; creando valores por defecto.")
            self._settings = AppSettings()
            self.save()
            return self._settings

        try:
            raw_text = self.config_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                raise ConfigurationError("El archivo settings.json no contiene un objeto JSON válido.")
            self._settings = self._from_dict(data)
            self._settings.validate()
            logger.info("Configuración cargada desde %s", self.config_path)
            return self._settings
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"No se pudo cargar la configuración en {self.config_path}.",
                details=str(exc),
            ) from exc

    def save(self, settings: AppSettings | None = None) -> None:
        """Valida y guarda la configuración de forma atómica sin corromper archivos previos."""
        if settings is not None:
            self._settings = settings
        self._settings.validate()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(asdict(self._settings), indent=2, ensure_ascii=False)
        tmp_path = self.config_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, self.config_path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise ConfigurationError(
                f"No se pudo guardar la configuración en {self.config_path}.",
                details=str(exc),
            ) from exc

    def add_recent_file(self, file_path: Path) -> None:
        """Registra un archivo abierto recientemente manteniendo máximo 10 entradas."""
        normalized = str(file_path.resolve())
        recents = [p for p in self._settings.recent_files if p != normalized]
        recents.insert(0, normalized)
        self._settings.recent_files = recents[:10]
        self._settings.last_open_dir = str(file_path.resolve().parent)
        self.save()

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> AppSettings:
        defaults = asdict(AppSettings())
        merged: dict[str, Any] = {}
        for key, default_val in defaults.items():
            merged[key] = data.get(key, default_val)
        return AppSettings(**merged)
