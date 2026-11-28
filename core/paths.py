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
    data_root: Path | None = None
    layout: str = "portable"

    @classmethod
    def detect(
        cls,
        *,
        frozen: bool | None = None,
        executable_path: Path | None = None,
        meipass_path: Path | None = None,
        platform: str | None = None,
        user_home: Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> AppPaths:
        """Detecta la raíz portable y construye todas las rutas de la aplicación.

        Disposición de los datos modificables (config, modelos, logs, caché...):

        - Windows / ejecución desde el código fuente: junto a la aplicación
          (modo portable, comportamiento histórico).
        - ``.app`` de macOS: en las carpetas estándar del usuario
          (``~/Library/Application Support``, ``Caches``, ``Logs`` y
          ``~/Music/ZenTralAudio`` para exportaciones), porque el bundle puede
          estar en /Applications (sin permisos de escritura) y macOS invalida la
          firma de un bundle modificado. Si junto a la ``.app`` existe
          ``models_data`` o el archivo ``ZenTralAudio.portable``, se mantiene el
          modo portable de la edición Offline AI.
        - ``ZENTRALAUDIO_MACOS_USER_DIRS=1`` fuerza las carpetas estándar de
          macOS (lo usa el lanzador de desarrollo para compartir modelos y
          ajustes con la ``.app``) y ``ZENTRALAUDIO_DATA_DIR`` permite una raíz
          de datos personalizada.

        Los parámetros opcionales permiten pruebas deterministas simulando entornos
        congelados (PyInstaller) sin alterar el proceso actual.
        """
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
        platform_name = platform or sys.platform
        env = os.environ if environ is None else environ
        home = Path(user_home) if user_home is not None else Path.home()
        in_macos_bundle = False

        if is_frozen:
            exe = Path(executable_path or sys.executable).resolve()
            raw_meipass = meipass_path or getattr(sys, "_MEIPASS", None)

            # PyInstaller coloca el binario de una app macOS en:
            # <carpeta>/ZenTralAudio.app/Contents/MacOS/ZenTralAudio
            # Los datos modificables nunca se guardan dentro del .app.
            if (
                exe.parent.name == "MacOS"
                and exe.parent.parent.name == "Contents"
                and exe.parent.parent.parent.suffix == ".app"
            ):
                in_macos_bundle = True
                app_bundle = exe.parent.parent.parent
                app_root = app_bundle.parent
                bundle_root = app_bundle / "Contents" / "Resources"
            else:
                app_root = exe.parent
                bundle_root = Path(raw_meipass).resolve() if raw_meipass else app_root
        else:
            app_root = Path(__file__).resolve().parent.parent
            bundle_root = app_root

        presets_dir = cls._resolve_bundle_subdir(app_root, bundle_root, "presets")
        resources_dir = cls._resolve_bundle_subdir(app_root, bundle_root, "resources")

        custom_root = env.get("ZENTRALAUDIO_DATA_DIR", "").strip()
        portable_marker = (app_root / "models_data").is_dir() or (app_root / "ZenTralAudio.portable").is_file()
        use_macos_user_dirs = platform_name == "darwin" and (
            (in_macos_bundle and not portable_marker)
            or env.get("ZENTRALAUDIO_MACOS_USER_DIRS", "") == "1"
        )

        if custom_root:
            data_root = Path(custom_root).expanduser().resolve()
            layout = "custom"
            cache_dir = data_root / "cache"
            logs_dir = data_root / "logs"
            exports_dir = data_root / "exports"
        elif use_macos_user_dirs:
            library = home / "Library"
            data_root = library / "Application Support" / APP_NAME
            layout = "macos-user"
            cache_dir = library / "Caches" / APP_NAME
            logs_dir = library / "Logs" / APP_NAME
            exports_dir = home / "Music" / APP_NAME
        else:
            data_root = app_root
            layout = "portable"
            cache_dir = data_root / "cache"
            logs_dir = data_root / "logs"
            exports_dir = data_root / "exports"

        return cls(
            is_frozen=is_frozen,
            app_root=app_root,
            bundle_root=bundle_root,
            models_dir=data_root / "models_data",
            ambience_dir=data_root / "ambience_data",
            cache_dir=cache_dir,
            projects_dir=data_root / "projects",
            exports_dir=exports_dir,
            config_dir=data_root / "config",
            logs_dir=logs_dir,
            presets_dir=presets_dir,
            custom_presets_dir=data_root / "presets" / "custom",
            resources_dir=resources_dir,
            bin_dir=data_root / "bin",
            data_root=data_root,
            layout=layout,
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
                self.app_root / "bin" / exe_name,
                self.app_root / exe_name,
                self.bundle_root / "bin" / exe_name,
                self.bundle_root / exe_name,
            ]
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        system_ffmpeg = shutil.which(exe_name)
        if system_ffmpeg:
            return Path(system_ffmpeg).resolve()

        # Las apps abiertas desde Finder no heredan el PATH de la terminal, así que
        # se buscan también las rutas habituales de Homebrew.
        if os.name != "nt":
            for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
                candidate = Path(prefix) / exe_name
                if candidate.is_file():
                    return candidate.resolve()

        # Binario estático de imageio-ffmpeg (dependencia de desarrollo en macOS).
        try:
            import imageio_ffmpeg

            bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if bundled.is_file():
                return bundled.resolve()
        except Exception:
            pass
        return None

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
            "layout": self.layout,
            "app_root": str(self.app_root),
            "data_root": str(self.data_root or self.app_root),
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
