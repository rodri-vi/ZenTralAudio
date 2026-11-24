"""Gestión de rutas portables para ejecución desde Python o ejecutable PyInstaller."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from core.exceptions import PathPermissionError

APP_NAME = "ZenTralAudio"
APP_VERSION = "0.1.0"
APP_ORGANIZATION = "ZenTralAudio"


@dataclass(frozen=True)
class AppPaths:
    """Contenedor inmutable de rutas relativas a la ubicación ejecutable."""

    is_frozen: bool
    app_root: Path
    bundle_root: Path
    models_dir: Path
    ambience_dir: Path
    cache_dir: Path
    projects_dir: Path
    exports_dir: Path
    config_dir: Path
    logs_dir: Path
    presets_dir: Path
    custom_presets_dir: Path
    resources_dir: Path
    bin_dir: Path

    @classmethod
    def detect(
        cls,
        *,
        frozen: bool | None = None,
        executable_path: Path | None = None,
        meipass_path: Path | None = None,
    ) -> AppPaths:
        """Detecta la raíz portable y construye todas las rutas de la aplicación.

        Los parámetros opcionales permiten pruebas deterministas simulando entornos
        congelados (PyInstaller) sin alterar el proceso actual.
        """
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen

        if is_frozen:
            exe = Path(executable_path or sys.executable).resolve()
            app_root = exe.parent
            raw_meipass = meipass_path or getattr(sys, "_MEIPASS", None)
            bundle_root = Path(raw_meipass).resolve() if raw_meipass else app_root
        else:
            app_root = Path(__file__).resolve().parent.parent
            bundle_root = app_root

        presets_dir = cls._resolve_bundle_subdir(app_root, bundle_root, "presets")
        resources_dir = cls._resolve_bundle_subdir(app_root, bundle_root, "resources")

        return cls(
            is_frozen=is_frozen,
            app_root=app_root,
            bundle_root=bundle_root,
            models_dir=app_root / "models_data",
            ambience_dir=app_root / "ambience_data",
            cache_dir=app_root / "cache",
            projects_dir=app_root / "projects",
            exports_dir=app_root / "exports",
            config_dir=app_root / "config",
            logs_dir=app_root / "logs",
            presets_dir=presets_dir,
            custom_presets_dir=app_root / "presets" / "custom",
            resources_dir=resources_dir,
            bin_dir=app_root / "bin",
        )

    @staticmethod
    def _resolve_bundle_subdir(app_root: Path, bundle_root: Path, relative: str) -> Path:
        external = app_root / relative
        if external.exists():
            return external
        bundled = bundle_root / relative
        if bundled.exists():
            return bundled
        # En tests congelados simulados y herramientas de desarrollo, los datos
        # inmutables pueden seguir en el checkout fuente aunque no exista _MEIPASS.
        source_bundle = Path(__file__).resolve().parent.parent / relative
        if source_bundle.exists():
            return source_bundle
        return external

    def ensure_writable_directories(self) -> None:
        """Crea las carpetas portables necesarias y verifica permisos de escritura."""
        mutable_dirs = [
            self.models_dir,
            self.ambience_dir,
            self.cache_dir,
            self.projects_dir,
            self.exports_dir,
            self.config_dir,
            self.logs_dir,
            self.custom_presets_dir,
        ]
        for directory in mutable_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                probe = directory / ".write_probe.tmp"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
            except OSError as exc:
                raise PathPermissionError(
                    f"No hay permisos de escritura en la carpeta portable: {directory}",
                    details=str(exc),
                ) from exc

    def resolve_resource(self, *parts: str) -> Path:
        """Resuelve un recurso empaquetado priorizando overrides externos."""
        candidate_external = self.app_root.joinpath("resources", *parts)
        if candidate_external.exists():
            return candidate_external

        candidate_bundle = self.bundle_root.joinpath("resources", *parts)
        if candidate_bundle.exists():
            return candidate_bundle

        # Permite ejecutar pruebas con rutas portables simuladas sin copiar
        # los recursos inmutables del checkout a cada directorio temporal.
        candidate_source = Path(__file__).resolve().parent.parent.joinpath("resources", *parts)
        if candidate_source.exists():
            return candidate_source

        return candidate_bundle

    def find_ffmpeg(self, custom_path: str | Path | None = None) -> Path | None:
        """Localiza ffmpeg priorizando binarios portables junto al ejecutable."""
        exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        candidates: list[Path] = []

        if custom_path:
            custom = Path(custom_path)
            if custom.is_dir():
                candidates.append(custom / exe_name)
            else:
                candidates.append(custom)

        candidates.extend(
            [
                self.bin_dir / exe_name,
                self.app_root / exe_name,
                self.bundle_root / "bin" / exe_name,
                self.bundle_root / exe_name,
            ]
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        system_ffmpeg = shutil.which(exe_name)
        return Path(system_ffmpeg).resolve() if system_ffmpeg else None

    def find_ffprobe(self, custom_ffmpeg_path: str | Path | None = None) -> Path | None:
        """Localiza ffprobe asociado a ffmpeg o en el sistema."""
        exe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        ffmpeg = self.find_ffmpeg(custom_ffmpeg_path)
        if ffmpeg is not None:
            sibling = ffmpeg.parent / exe_name
            if sibling.is_file():
                return sibling

        for candidate in (
            self.bin_dir / exe_name,
            self.bundle_root / "bin" / exe_name,
        ):
            if candidate.is_file():
                return candidate.resolve()

        system_ffprobe = shutil.which(exe_name)
        return Path(system_ffprobe).resolve() if system_ffprobe else None

    def to_diagnostic_dict(self) -> dict[str, str]:
        """Devuelve un resumen legible de todas las rutas detectadas."""
        return {
            "is_frozen": str(self.is_frozen),
            "app_root": str(self.app_root),
            "bundle_root": str(self.bundle_root),
            "models_dir": str(self.models_dir),
            "ambience_dir": str(self.ambience_dir),
            "cache_dir": str(self.cache_dir),
            "exports_dir": str(self.exports_dir),
            "config_dir": str(self.config_dir),
            "logs_dir": str(self.logs_dir),
            "resources_dir": str(self.resources_dir),
            "ffmpeg": str(self.find_ffmpeg() or "No detectado"),
        }
