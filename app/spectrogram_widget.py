"""Visualizador de espectrograma STFT real en dBFS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from app.theme import Palette
from core.job_manager import CancellationToken


def _build_spectrogram_lut() -> np.ndarray:
    """Construye tabla LUT RGBA de 256 entradas acorde a la paleta ZenTralAudio."""
    stops = [
        (0.00, (17, 19, 24)),
        (0.30, (32, 48, 72)),
        (0.60, (85, 151, 232)),
        (0.82, (50, 182, 165)),
        (1.00, (231, 174, 84)),
    ]
    lut = np.zeros((256, 4), dtype=np.uint8)
    for idx in range(256):
        t = idx / 255.0
        for s_idx in range(len(stops) - 1):
            t0, c0 = stops[s_idx]
            t1, c1 = stops[s_idx + 1]
            if t0 <= t <= t1:
                local = (t - t0) / max(1e-6, t1 - t0)
                r = int(c0[0] + (c1[0] - c0[0]) * local)
                g = int(c0[1] + (c1[1] - c0[1]) * local)
                b = int(c0[2] + (c1[2] - c0[2]) * local)
                lut[idx] = (r, g, b, 255)
                break
    return lut


_LUT = _build_spectrogram_lut()


def compute_spectrogram_image(
    samples: np.ndarray,
    sample_rate: int,
    max_cols: int = 900,
    *,
    cancel_token: CancellationToken | None = None,
) -> QImage | None:
    """Calcula espectrograma STFT en dB y devuelve QImage listo para pintar."""
    if samples.size < 256 or sample_rate <= 0:
        return None

    mono = samples.mean(axis=0) if samples.ndim > 1 else samples
    mono = np.asarray(mono, dtype=np.float32)

    n_fft = 1024
    if mono.shape[0] < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.shape[0]))

    total_frames = mono.shape[0] - n_fft + 1
    step = max(256, total_frames // max(1, max_cols))
    starts = np.arange(0, total_frames, step, dtype=np.int64)
    if starts.size == 0:
        return None

    window = np.hanning(n_fft).astype(np.float32)
    indices = np.empty((starts.size, n_fft // 2 + 1), dtype=np.uint8)
    # Calcular por bloques permite interrumpir el análisis de grabaciones largas
    # sin construir de una sola vez una matriz temporal gigantesca.
    block_frames = 64
    for block_start in range(0, starts.size, block_frames):
        if cancel_token:
            cancel_token.raise_if_cancelled()
        block_starts = starts[block_start : block_start + block_frames]
        frames = np.stack([mono[s : s + n_fft] * window for s in block_starts], axis=0)
        spectrum = np.abs(np.fft.rfft(frames, axis=1)) / (n_fft * 0.5)
        db = 20.0 * np.log10(np.maximum(spectrum, 1e-5))
        norm = np.clip((db + 90.0) / 90.0, 0.0, 1.0)
        indices[block_start : block_start + block_starts.size] = (norm * 255.0).astype(np.uint8)

    if cancel_token:
        cancel_token.raise_if_cancelled()
    indices_flipped = indices[:, ::-1].T.copy()
    rgba = _LUT[indices_flipped]
    h, w, _ = rgba.shape
    image = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return image.copy()


class SpectrogramWidget(QWidget):
    """Muestra espectrograma STFT real con marcas de frecuencia y cursor sincronizado."""

    seek_requested = Signal(float)
    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setAcceptDrops(True)
        self.setToolTip("Espectrograma de frecuencias (Hz/kHz). Clic para mover cursor.")
        self._image: QImage | None = None
        self._duration: float = 0.0
        self._sample_rate: int = 0
        self._position: float = 0.0

    def set_spectrogram(self, image: QImage | None, duration: float, sample_rate: int) -> None:
        self._image = image
        self._duration = max(0.0, duration)
        self._sample_rate = max(0, sample_rate)
        self._position = 0.0
        self.update()

    def set_position(self, seconds: float) -> None:
        self._position = max(0.0, min(self._duration, seconds))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._duration > 0 and self.width() > 0:
            t = max(0.0, min(1.0, event.position().x() / float(self.width()))) * self._duration
            self._position = t
            self.seek_requested.emit(t)
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

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, Palette.qcolor(Palette.BG_MAIN))

        if self._image is None or self._duration <= 0:
            painter.setPen(Palette.qcolor(Palette.TEXT_SECONDARY))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Espectrograma disponible al cargar un audio")
            painter.setPen(Palette.qcolor(Palette.BORDER))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            painter.end()
            return

        painter.drawImage(rect, self._image)

        nyquist = self._sample_rate / 2.0
        if nyquist > 0:
            painter.setPen(Palette.qcolor(Palette.TEXT_SECONDARY, 190))
            painter.drawText(6, 14, f"{nyquist / 1000.0:.1f} kHz")
            painter.drawText(6, rect.height() // 2, f"{(nyquist * 0.5) / 1000.0:.1f} kHz")
            painter.drawText(6, rect.height() - 6, "0 Hz")

        cursor_x = int((self._position / self._duration) * rect.width())
        painter.setPen(QPen(Palette.qcolor(Palette.TEXT_PRIMARY), 2))
        painter.drawLine(cursor_x, 0, cursor_x, rect.height())

        painter.setPen(Palette.qcolor(Palette.BORDER))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.end()
