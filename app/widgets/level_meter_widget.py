"""Medidor de nivel RMS y pico estéreo/mono con indicador de clipping."""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter
from PySide6.QtWidgets import QWidget

from app.theme import Palette


class LevelMeterWidget(QWidget):
    """Medidor horizontal compacto de señal en dBFS (-60 a 0 dBFS)."""

    MIN_DB = -60.0
    MAX_DB = 0.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(130, 24)
        self.setMaximumHeight(28)
        self.setToolTip("Nivel de salida (clic para reiniciar indicador de pico/clip).")
        self._rms_db: list[float] = [self.MIN_DB, self.MIN_DB]
        self._peak_db: list[float] = [self.MIN_DB, self.MIN_DB]
        self._clipped: bool = False

    def set_levels(self, rms_linear: list[float], peak_linear: list[float]) -> None:
        """Actualiza los niveles en escala lineal [0.0..1.0+]."""
        ch_count = max(1, min(2, len(rms_linear)))
        new_rms: list[float] = []
        new_peak: list[float] = []
        for idx in range(ch_count):
            rms_val = max(1e-6, float(rms_linear[idx]))
            peak_val = max(1e-6, float(peak_linear[idx] if idx < len(peak_linear) else rms_val))
            rms_db = max(self.MIN_DB, min(self.MAX_DB, 20.0 * math.log10(rms_val)))
            peak_db = max(self.MIN_DB, min(3.0, 20.0 * math.log10(peak_val)))
            if peak_val >= 0.999:
                self._clipped = True
            new_rms.append(rms_db)
            new_peak.append(min(self.MAX_DB, peak_db))
        self._rms_db = new_rms
        self._peak_db = new_peak
        self.update()

    def reset(self) -> None:
        """Reinicia lecturas y bandera de clipping."""
        self._rms_db = [self.MIN_DB, self.MIN_DB]
        self._peak_db = [self.MIN_DB, self.MIN_DB]
        self._clipped = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._clipped = False
            self.update()
        super().mousePressEvent(event)

    def _db_to_ratio(self, db: float) -> float:
        return max(0.0, min(1.0, (db - self.MIN_DB) / (self.MAX_DB - self.MIN_DB)))

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 2, -1, -2)
        painter.fillRect(rect, Palette.qcolor(Palette.BG_MAIN))
        painter.setPen(Palette.qcolor(Palette.BORDER))
        painter.drawRoundedRect(rect, 3, 3)

        clip_width = 10
        bar_area_width = max(10, rect.width() - clip_width - 6)
        channels = len(self._rms_db)
        bar_height = max(4.0, (rect.height() - 6.0 - (channels - 1) * 2.0) / channels)

        for ch in range(channels):
            y = rect.top() + 3.0 + ch * (bar_height + 2.0)
            track_rect = QRectF(rect.left() + 3.0, y, bar_area_width, bar_height)
            painter.fillRect(track_rect, Palette.qcolor(Palette.PANEL_SECONDARY))

            rms_ratio = self._db_to_ratio(self._rms_db[ch])
            peak_ratio = self._db_to_ratio(self._peak_db[ch])

            if rms_ratio > 0.0:
                fill_w = bar_area_width * rms_ratio
                # Verde/Turquesa hasta -12 dB, Ámbar hasta -3 dB, Rojo encima
                color_hex = Palette.ACCENT_PRIMARY
                if self._rms_db[ch] > -3.0:
                    color_hex = Palette.ERROR
                elif self._rms_db[ch] > -12.0:
                    color_hex = Palette.WARNING
                painter.fillRect(QRectF(track_rect.left(), y, fill_w, bar_height), QColor(color_hex))

            if peak_ratio > 0.02:
                peak_x = track_rect.left() + bar_area_width * peak_ratio
                painter.setPen(Palette.qcolor(Palette.TEXT_PRIMARY))
                painter.drawLine(int(peak_x), int(y), int(peak_x), int(y + bar_height))

        clip_rect = QRectF(rect.right() - clip_width - 2.0, rect.top() + 3.0, clip_width, rect.height() - 6.0)
        clip_color = Palette.ERROR if self._clipped else Palette.PANEL_SECONDARY
        painter.fillRect(clip_rect, QColor(clip_color))
        painter.end()
