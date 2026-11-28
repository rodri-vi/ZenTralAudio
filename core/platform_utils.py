"""Utilidades multiplataforma para integrar la app con el sistema operativo."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("zentralaudio.platform")

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = os.name == "nt"


def open_in_file_manager(folder: str | Path) -> bool:
    """Abre una carpeta en Finder / Explorador / gestor de archivos.

    Usa QDesktopServices (nativo en las tres plataformas) y, si no hay Qt
    disponible, el comando propio del sistema (``open`` en macOS,
    ``os.startfile`` en Windows y ``xdg-open`` en Linux).
    """
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            return True
    except Exception:
        logger.debug("QDesktopServices no pudo abrir %s", target, exc_info=True)

    try:
        if IS_WINDOWS:
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif IS_MACOS:
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True
    except Exception:
        logger.warning("No se pudo abrir la carpeta %s", target, exc_info=True)
        return False


def compute_device_label() -> str:
    """Descripción del hardware de inferencia para la barra superior."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    if IS_MACOS:
        import platform

        arch = platform.machine()
        return "CPU Apple Silicon (arm64)" if arch == "arm64" else f"CPU Intel ({arch})"
    return "Modo CPU"
