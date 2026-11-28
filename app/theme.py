"""Paleta oficial ZenTralAudio y utilidades de carga de estilos e iconos."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import QApplication

from core.paths import AppPaths

logger = logging.getLogger("zentralaudio.theme")


class Palette:
    """Constantes de color oficiales de la identidad visual ZenTralAudio."""

    BG_MAIN = "#111318"
    PANEL_PRIMARY = "#191D24"
    PANEL_SECONDARY = "#202630"
    BORDER = "#303744"
    TEXT_PRIMARY = "#F2F4F7"
    TEXT_SECONDARY = "#A0A9B7"
    ACCENT_PRIMARY = "#32B6A5"
    ACCENT_SECONDARY = "#5597E8"
    WARNING = "#E7AE54"
    ERROR = "#E16A70"

    @staticmethod
    def qcolor(hex_code: str, alpha: int = 255) -> QColor:
        color = QColor(hex_code)
        color.setAlpha(alpha)
        return color


_FALLBACK_QSS = f"""
QWidget {{ background-color: {Palette.BG_MAIN}; color: {Palette.TEXT_PRIMARY}; font-family: "Segoe UI"; }}
QFrame, QGroupBox {{ background-color: {Palette.PANEL_PRIMARY}; border: 1px solid {Palette.BORDER}; }}
QPushButton {{ background-color: {Palette.PANEL_SECONDARY}; border: 1px solid {Palette.BORDER}; padding: 6px 12px; border-radius: 6px; }}
"""


def _adapt_fonts_for_platform(qss: str) -> str:
    """Sustituye Segoe UI (solo Windows) por la fuente del sistema en macOS/Linux.

    Sin esto Qt registra un aviso y pierde tiempo buscando alias de una
    familia inexistente en cada arranque.
    """
    if sys.platform == "win32" or "Segoe UI" not in qss:
        return qss
    family = "Helvetica Neue" if sys.platform == "darwin" else "sans-serif"
    try:
        from PySide6.QtGui import QFontDatabase, QGuiApplication

        if QGuiApplication.instance() is not None:
            family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    except Exception:
        pass
    return qss.replace('"Segoe UI", ', "").replace('"Segoe UI"', f'"{family}"').replace(
        'font-family: "Inter"', f'font-family: "{family}", "Inter"'
    )


def load_stylesheet(paths: AppPaths) -> str:
    """Carga la hoja de estilos QSS desde resources/styles con respaldo seguro."""
    qss_path: Path = paths.resolve_resource("styles", "zentral_dark.qss")
    if qss_path.is_file():
        try:
            return _adapt_fonts_for_platform(qss_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("No se pudo leer %s: %s", qss_path, exc)
    else:
        logger.warning("Hoja de estilos no encontrada en %s; usando tema mínimo.", qss_path)
    return _adapt_fonts_for_platform(_FALLBACK_QSS)


def load_app_icon(paths: AppPaths) -> QIcon:
    """Carga el icono SVG oficial de ZenTralAudio."""
    svg_path = paths.resolve_resource("icons", "zentralaudio.svg")
    if svg_path.is_file():
        return QIcon(str(svg_path))
    return QIcon()


def apply_zentral_theme(app: QApplication, paths: AppPaths) -> None:
    """Aplica la hoja de estilos oscura y el icono global a QApplication."""
    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet(paths))
    icon = load_app_icon(paths)
    if not icon.isNull():
        app.setWindowIcon(icon)
