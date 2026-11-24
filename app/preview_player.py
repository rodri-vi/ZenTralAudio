"""Reproductor de previsualización permanente anclado en la parte inferior."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.theme import Palette
from app.widgets.level_meter_widget import LevelMeterWidget


def format_timecode(seconds: float) -> str:
    """Formatea segundos como MM:SS.d."""
    sec = max(0.0, float(seconds))
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:04.1f}"


class PreviewPlayer(QFrame):
    """Barra permanente de transporte, comparación ON/OFF, A/B y estado de motor."""

    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    seek_requested = Signal(float)
    loop_toggled = Signal(bool)
    bypass_toggled = Signal(bool)
    ab_slot_changed = Signal(str)
    snapshot_a_requested = Signal()
    loudness_match_toggled = Signal(bool)
    monitor_volume_changed = Signal(float)
    clear_selection_requested = Signal()
    cancel_processing_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewPlayerBar")
        self._duration: float = 0.0
        self._user_scrubbing = False

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 8, 14, 10)
        main_layout.setSpacing(6)

        timeline_row = QHBoxLayout()
        self._current_lbl = QLabel("00:00.0")
        self._current_lbl.setFixedWidth(56)

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setEnabled(False)
        self._seek_slider.sliderPressed.connect(self._on_scrub_start)
        self._seek_slider.sliderReleased.connect(self._on_scrub_end)
        self._seek_slider.sliderMoved.connect(self._on_scrub_move)

        self._total_lbl = QLabel("00:00.0")
        self._total_lbl.setFixedWidth(56)
        self._total_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._selection_lbl = QLabel("Fragmento: Completo")
        self._selection_lbl.setObjectName("SecondaryText")
        self._clear_sel_btn = QPushButton("Limpiar fragmento")
        self._clear_sel_btn.setVisible(False)
        self._clear_sel_btn.clicked.connect(self.clear_selection_requested.emit)

        timeline_row.addWidget(self._current_lbl)
        timeline_row.addWidget(self._seek_slider, stretch=1)
        timeline_row.addWidget(self._total_lbl)
        timeline_row.addSpacing(12)
        timeline_row.addWidget(self._selection_lbl)
        timeline_row.addWidget(self._clear_sel_btn)
        main_layout.addLayout(timeline_row)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        self._play_btn = QPushButton("Play")
        self._play_btn.setObjectName("PrimaryButton")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self.play_clicked.emit)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self.pause_clicked.emit)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)

        self._loop_btn = QPushButton("Loop")
        self._loop_btn.setObjectName("TogglePill")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setToolTip("Repetir continuamente el audio o el fragmento seleccionado.")
        self._loop_btn.toggled.connect(self.loop_toggled.emit)

        controls_row.addWidget(self._play_btn)
        controls_row.addWidget(self._pause_btn)
        controls_row.addWidget(self._stop_btn)
        controls_row.addWidget(self._loop_btn)
        controls_row.addSpacing(10)

        self._bypass_btn = QPushButton("PROCESADO: ON")
        self._bypass_btn.setObjectName("BypassButton")
        self._bypass_btn.setCheckable(True)
        self._bypass_btn.setChecked(True)
        self._bypass_btn.setToolTip(
            "ON: escucha la cadena activa (IA + DSP + Ambiente). OFF: escucha el audio original intacto."
        )
        self._bypass_btn.toggled.connect(self._on_bypass_toggled)
        controls_row.addWidget(self._bypass_btn)

        self._btn_a = QPushButton("A")
        self._btn_a.setObjectName("TogglePill")
        self._btn_a.setCheckable(True)
        self._btn_a.setFixedWidth(34)
        self._btn_a.setToolTip("Escuchar el estado guardado en memoria A.")
        self._btn_a.clicked.connect(lambda: self._select_ab("A"))

        self._btn_b = QPushButton("B")
        self._btn_b.setObjectName("TogglePill")
        self._btn_b.setCheckable(True)
        self._btn_b.setChecked(True)
        self._btn_b.setFixedWidth(34)
        self._btn_b.setToolTip("Escuchar el procesamiento actual (estado B).")
        self._btn_b.clicked.connect(lambda: self._select_ab("B"))

        self._save_a_btn = QPushButton("Guardar B->A")
        self._save_a_btn.setToolTip("Guarda una instantánea del procesamiento actual en A para comparar.")
        self._save_a_btn.clicked.connect(self.snapshot_a_requested.emit)

        self._loudness_chk = QCheckBox("Compensar sonoridad")
        self._loudness_chk.setChecked(True)
        self._loudness_chk.setToolTip("Iguala el volumen percibido al comparar Original y Procesado.")
        self._loudness_chk.toggled.connect(self.loudness_match_toggled.emit)

        controls_row.addWidget(self._btn_a)
        controls_row.addWidget(self._btn_b)
        controls_row.addWidget(self._save_a_btn)
        controls_row.addWidget(self._loudness_chk)
        controls_row.addStretch(1)

        vol_lbl = QLabel("Monitor:")
        vol_lbl.setObjectName("SecondaryText")
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 125)
        self._vol_slider.setValue(85)
        self._vol_slider.setFixedWidth(95)
        self._vol_slider.setToolTip("Volumen de escucha (no altera el archivo exportado).")
        self._vol_slider.valueChanged.connect(lambda v: self.monitor_volume_changed.emit(v / 100.0))

        self.level_meter = LevelMeterWidget()

        self._status_lbl = QLabel("Sin audio cargado")
        self._status_lbl.setStyleSheet(f"color: {Palette.TEXT_SECONDARY}; font-size: 12px;")
        self._progress = QProgressBar()
        self._progress.setFixedWidth(120)
        self._progress.setVisible(False)
        self._cancel_btn = QPushButton("Cancelar")
        self._cancel_btn.setObjectName("DangerButton")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_processing_requested.emit)

        controls_row.addWidget(vol_lbl)
        controls_row.addWidget(self._vol_slider)
        controls_row.addWidget(self.level_meter)
        controls_row.addSpacing(8)
        controls_row.addWidget(self._status_lbl)
        controls_row.addWidget(self._progress)
        controls_row.addWidget(self._cancel_btn)
        main_layout.addLayout(controls_row)

    def set_bypass_processed(self, enabled: bool) -> None:
        """Cambia programáticamente el estado ON (procesado) / OFF (original)."""
        self._bypass_btn.setChecked(enabled)

    def set_audio_loaded(self, duration_seconds: float, *, enabled: bool = True) -> None:
        self._duration = max(0.0, duration_seconds)
        self._total_lbl.setText(format_timecode(self._duration))
        self._current_lbl.setText("00:00.0")
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(enabled and self._duration > 0)
        self._play_btn.setEnabled(enabled and self._duration > 0)
        self._pause_btn.setEnabled(enabled and self._duration > 0)
        self._stop_btn.setEnabled(enabled and self._duration > 0)

    def set_position(self, seconds: float) -> None:
        if self._user_scrubbing or self._duration <= 0:
            return
        clamped = max(0.0, min(self._duration, seconds))
        self._current_lbl.setText(format_timecode(clamped))
        self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(int((clamped / self._duration) * 1000))
        self._seek_slider.blockSignals(False)

    def set_selection_range(self, start_s: float | None, end_s: float | None) -> None:
        if start_s is None or end_s is None:
            self._selection_lbl.setText("Fragmento: Completo")
            self._clear_sel_btn.setVisible(False)
        else:
            self._selection_lbl.setText(f"Fragmento: {format_timecode(start_s)} - {format_timecode(end_s)}")
            self._clear_sel_btn.setVisible(True)

    def set_processing_state(self, busy: bool, message: str, progress: float = 0.0) -> None:
        self._status_lbl.setText(message)
        color = Palette.WARNING if busy else Palette.ACCENT_PRIMARY
        self._status_lbl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")
        self._progress.setVisible(busy)
        self._cancel_btn.setVisible(busy)
        if busy:
            if progress < 0:
                self._progress.setRange(0, 0)
            else:
                self._progress.setRange(0, 100)
                self._progress.setValue(int(max(0.0, min(1.0, progress)) * 100))

    def set_error_state(self, message: str) -> None:
        self._status_lbl.setText(message)
        self._status_lbl.setStyleSheet(f"color: {Palette.ERROR}; font-size: 12px; font-weight: 600;")
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)

    def set_monitor_volume(self, volume: float) -> None:
        self._vol_slider.blockSignals(True)
        self._vol_slider.setValue(int(max(0, min(125, round(volume * 100)))))
        self._vol_slider.blockSignals(False)

    def _on_bypass_toggled(self, checked: bool) -> None:
        self._bypass_btn.setText("PROCESADO: ON" if checked else "ORIGINAL: OFF")
        self.bypass_toggled.emit(checked)

    def _select_ab(self, slot: str) -> None:
        self._btn_a.blockSignals(True)
        self._btn_b.blockSignals(True)
        self._btn_a.setChecked(slot == "A")
        self._btn_b.setChecked(slot == "B")
        self._btn_a.blockSignals(False)
        self._btn_b.blockSignals(False)
        self.ab_slot_changed.emit(slot)

    def _on_scrub_start(self) -> None:
        self._user_scrubbing = True

    def _on_scrub_move(self, value: int) -> None:
        if self._duration > 0:
            self._current_lbl.setText(format_timecode((value / 1000.0) * self._duration))

    def _on_scrub_end(self) -> None:
        self._user_scrubbing = False
        if self._duration > 0:
            target = (self._seek_slider.value() / 1000.0) * self._duration
            self.seek_requested.emit(target)
