"""Visualizador interactivo de forma de onda con selección de fragmento y drag & drop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.theme import Palette


class WaveformWidget(QWidget):
    """Renderiza forma de onda mono/estéreo, cursor de reproducción y selección temporal."""

    seek_requested = Signal(float)  # segundos
    selection_changed = Signal(float, float)  # (inicio_s, fin_s)
    selection_cleared = Signal()
    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setToolTip(
            "Clic izquierdo: mover cursor | Shift + arrastrar (o clic derecho): seleccionar fragmento | Doble clic: limpiar selección | Arrastra aquí un audio."
        )

        self._peaks_min: np.ndarray | None = None  # shape (channels, bins)
        self._peaks_max: np.ndarray | None = None
        self._duration: float = 0.0
        self._position: float = 0.0
        self._selection: tuple[float, float] | None = None
        self._dragging_selection = False
        self._drag_anchor_s: float = 0.0
        self._title: str = "Arrastra un archivo de audio aquí o pulsa 'Abrir Audio'"

    def set_audio(self, samples: np.ndarray | None, sample_rate: int, title: str = "") -> None:
        """Calcula envolvente de picos min/max sin bloquear la pintura."""
        if samples is None or samples.size == 0 or sample_rate <= 0:
            self._peaks_min = None
            self._peaks_max = None
            self._duration = 0.0
            self._position = 0.0
            self._selection = None
            self._title = title or "Arrastra un archivo de audio aquí o pulsa 'Abrir Audio'"
            self.update()
            return

        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arr = arr[:2]  # Visualizar hasta 2 canales

        total_samples = arr.shape[1]
        self._duration = total_samples / float(sample_rate)
        self._position = 0.0
        self._selection = None
        self._title = title

        target_bins = min(max(400, self.width() * 2), 4000, total_samples)
        chunk = max(1, total_samples // target_bins)
        usable = (total_samples // chunk) * chunk
        trimmed = arr[:, :usable].reshape(arr.shape[0], -1, chunk)
        self._peaks_min = trimmed.min(axis=2)
        self._peaks_max = trimmed.max(axis=2)
        self.update()

    def set_position(self, seconds: float) -> None:
        self._position = max(0.0, min(self._duration, seconds))
        self.update()

    def set_selection(self, start_s: float | None, end_s: float | None) -> None:
        if start_s is None or end_s is None or self._duration <= 0:
            self._selection = None
        else:
            a = max(0.0, min(self._duration, start_s))
            b = max(0.0, min(self._duration, end_s))
            self._selection = (min(a, b), max(a, b))
        self.update()

    def _x_to_time(self, x: float) -> float:
        if self._duration <= 0 or self.width() <= 0:
            return 0.0
        ratio = max(0.0, min(1.0, x / float(self.width())))
        return ratio * self._duration

    def _time_to_x(self, seconds: float) -> float:
        if self._duration <= 0:
            return 0.0
        return (seconds / self._duration) * float(self.width())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._duration <= 0:
            return
        t = self._x_to_time(event.position().x())
        if (
            event.button() == Qt.MouseButton.RightButton
            or (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            self._dragging_selection = True
            self._drag_anchor_s = t
            self._selection = (t, t)
            self.update()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._position = t
            self.seek_requested.emit(t)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_selection and self._duration > 0:
            t = self._x_to_time(event.position().x())
            self._selection = (min(self._drag_anchor_s, t), max(self._drag_anchor_s, t))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging_selection:
            self._dragging_selection = False
            if self._selection and (self._selection[1] - self._selection[0]) >= 0.05:
                self.selection_changed.emit(self._selection[0], self._selection[1])
            else:
                self._selection = None
                self.selection_cleared.emit()
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._selection is not None:
            self._selection = None
            self.selection_cleared.emit()
            self.update()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_file():
                    self.file_dropped.emit(str(path))
                    event.acceptProposedAction()
                    return
        event.ignore()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()

        painter.fillRect(rect, Palette.qcolor(Palette.BG_MAIN))
        painter.setPen(Palette.qcolor(Palette.BORDER))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        if self._peaks_min is None or self._peaks_max is None or self._duration <= 0:
            painter.setPen(Palette.qcolor(Palette.TEXT_SECONDARY))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._title)
            painter.end()
            return

        # Selección de fragmento
        if self._selection is not None:
            sx = self._time_to_x(self._selection[0])
            ex = self._time_to_x(self._selection[1])
            sel_rect = QRectF(sx, 1, max(2.0, ex - sx), rect.height() - 2)
            painter.fillRect(sel_rect, Palette.qcolor(Palette.ACCENT_SECONDARY, 45))

        channels, bins = self._peaks_min.shape
        lane_h = rect.height() / float(channels)
        colors = [Palette.ACCENT_PRIMARY, Palette.ACCENT_SECONDARY]

        for ch in range(channels):
            cy = (ch + 0.5) * lane_h
            amp = lane_h * 0.44
            painter.setPen(Palette.qcolor(Palette.BORDER, 140))
            painter.drawLine(0, int(cy), rect.width(), int(cy))

            pen = QPen(Palette.qcolor(colors[ch % len(colors)], 220))
            pen.setWidth(1)
            painter.setPen(pen)

            for px in range(rect.width()):
                b0 = min(bins - 1, int(px * bins / max(1, rect.width())))
                b1 = min(bins, max(b0 + 1, int((px + 1) * bins / max(1, rect.width()))))
                vmin = float(self._peaks_min[ch, b0:b1].min())
                vmax = float(self._peaks_max[ch, b0:b1].max())
                y_top = cy - vmax * amp
                y_bot = cy - vmin * amp
                painter.drawLine(QPointF(px, y_top), QPointF(px, y_bot))

        # Cursor de reproducción
        cursor_x = int(self._time_to_x(self._position))
        painter.setPen(QPen(Palette.qcolor(Palette.TEXT_PRIMARY), 2))
        painter.drawLine(cursor_x, 0, cursor_x, rect.height())
        painter.end()
