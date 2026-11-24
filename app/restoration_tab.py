"""Pestaña 1: Restauración IA, carga de audio, presets, intensidad y opciones avanzadas."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.spectrogram_widget import SpectrogramWidget
from app.theme import Palette
from app.waveform_widget import WaveformWidget
from app.widgets.dynamic_param_panel import DynamicParamPanel, UIParamDefinition


class RestorationTab(QWidget):
    """Centro principal de restauración de voz de ZenTralAudio."""

    open_file_requested = Signal(str)
    play_original_requested = Signal()
    model_selected = Signal(str)
    manage_models_requested = Signal()
    preset_selected = Signal(str)
    save_custom_preset_requested = Signal()
    delete_custom_preset_requested = Signal(str)
    intensity_changed = Signal(int)
    advanced_param_changed = Signal(str, object)
    advanced_reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        file_group = QGroupBox("1. Grabación de Entrada")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)

        file_top = QHBoxLayout()
        self._open_btn = QPushButton("Abrir Audio...")
        self._open_btn.setObjectName("PrimaryButton")
        self._open_btn.setToolTip("Cargar archivo WAV, MP3, FLAC, M4A, AAC u otro compatible.")
        self._open_btn.clicked.connect(self._browse_file)

        self._play_orig_btn = QPushButton("Escuchar Original")
        self._play_orig_btn.setEnabled(False)
        self._play_orig_btn.setToolTip("Reproducir la grabación original sin procesamiento.")
        self._play_orig_btn.clicked.connect(self.play_original_requested.emit)

        self._file_name_lbl = QLabel("Ningún archivo cargado")
        self._file_name_lbl.setStyleSheet("font-weight: 600;")

        self._view_mode_combo = QComboBox()
        self._view_mode_combo.addItem("Forma de onda", "waveform")
        self._view_mode_combo.addItem("Espectrograma", "spectrogram")
        self._view_mode_combo.addItem("Vista dividida", "split")
        self._view_mode_combo.setCurrentIndex(2)
        self._view_mode_combo.currentIndexChanged.connect(self._update_visualizer_mode)

        file_top.addWidget(self._open_btn)
        file_top.addWidget(self._play_orig_btn)
        file_top.addSpacing(8)
        file_top.addWidget(self._file_name_lbl, stretch=1)
        file_top.addWidget(QLabel("Vista:"))
        file_top.addWidget(self._view_mode_combo)
        file_layout.addLayout(file_top)

        meta_row = QHBoxLayout()
        self._meta_duration = QLabel("Duración: —")
        self._meta_sr = QLabel("Frecuencia: —")
        self._meta_bits = QLabel("Profundidad: —")
        self._meta_channels = QLabel("Canales: —")
        for lbl in (self._meta_duration, self._meta_sr, self._meta_bits, self._meta_channels):
            lbl.setObjectName("SecondaryText")
            meta_row.addWidget(lbl)
            meta_row.addSpacing(14)
        meta_row.addStretch(1)
        file_layout.addLayout(meta_row)

        self._vis_splitter = QSplitter(Qt.Orientation.Vertical)
        self.waveform = WaveformWidget()
        self.spectrogram = SpectrogramWidget()
        self.waveform.file_dropped.connect(self.open_file_requested.emit)
        self.spectrogram.file_dropped.connect(self.open_file_requested.emit)
        self._vis_splitter.addWidget(self.waveform)
        self._vis_splitter.addWidget(self.spectrogram)
        self._vis_splitter.setSizes([150, 140])
        file_layout.addWidget(self._vis_splitter, stretch=1)

        root.addWidget(file_group, stretch=2)

        controls_frame = QFrame()
        controls_frame.setObjectName("PanelPrimary")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(10)

        model_row = QHBoxLayout()
        model_lbl = QLabel("Modelo IA:")
        model_lbl.setFixedWidth(95)
        self._model_combo = QComboBox()
        self._model_combo.setToolTip("Selecciona el modelo de inteligencia artificial para restaurar la voz.")
        self._model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        self._model_status_badge = QLabel("No instalado")
        self._model_status_badge.setObjectName("StatusBadgeWarn")

        self._manage_models_btn = QPushButton("Gestionar modelos...")
        self._manage_models_btn.setToolTip("Ver estado, licencias, descargas e integridad de modelos IA.")
        self._manage_models_btn.clicked.connect(self.manage_models_requested.emit)

        model_row.addWidget(model_lbl)
        model_row.addWidget(self._model_combo, stretch=1)
        model_row.addWidget(self._model_status_badge)
        model_row.addWidget(self._manage_models_btn)
        controls_layout.addLayout(model_row)

        self._model_note_lbl = QLabel("")
        self._model_note_lbl.setObjectName("SecondaryText")
        self._model_note_lbl.setWordWrap(True)
        controls_layout.addWidget(self._model_note_lbl)

        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Preset:")
        preset_lbl.setFixedWidth(95)
        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip("Perfiles profesionales adaptados al modelo y módulos DSP activos.")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)

        self._save_preset_btn = QPushButton("Guardar preset...")
        self._save_preset_btn.setToolTip("Guarda la configuración actual como preset personalizado.")
        self._save_preset_btn.clicked.connect(self.save_custom_preset_requested.emit)

        self._del_preset_btn = QPushButton("Eliminar")
        self._del_preset_btn.setObjectName("DangerButton")
        self._del_preset_btn.setEnabled(False)
        self._del_preset_btn.setToolTip("Elimina el preset personalizado seleccionado.")
        self._del_preset_btn.clicked.connect(self._on_delete_preset)

        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self._preset_combo, stretch=1)
        preset_row.addWidget(self._save_preset_btn)
        preset_row.addWidget(self._del_preset_btn)
        controls_layout.addLayout(preset_row)

        self._preset_note_lbl = QLabel("")
        self._preset_note_lbl.setStyleSheet(f"color: {Palette.WARNING}; font-size: 12px;")
        self._preset_note_lbl.setWordWrap(True)
        self._preset_note_lbl.setVisible(False)
        controls_layout.addWidget(self._preset_note_lbl)

        intensity_row = QHBoxLayout()
        int_lbl = QLabel("Intensidad:")
        int_lbl.setFixedWidth(95)
        self._intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self._intensity_slider.setRange(0, 100)
        self._intensity_slider.setValue(75)
        self._intensity_slider.setToolTip(
            "Control real (0% original a 100% procesamiento completo con mezcla alineada y parámetros del modelo)."
        )
        self._intensity_slider.valueChanged.connect(self._on_intensity_changed)

        self._intensity_val_lbl = QLabel("75 %")
        self._intensity_val_lbl.setFixedWidth(48)
        self._intensity_val_lbl.setStyleSheet(f"color: {Palette.ACCENT_PRIMARY}; font-weight: 700;")

        self._adv_toggle_btn = QPushButton("OPCIONES AVANZADAS v")
        self._adv_toggle_btn.setObjectName("SecondaryAccentButton")
        self._adv_toggle_btn.setCheckable(True)
        self._adv_toggle_btn.setChecked(False)
        self._adv_toggle_btn.setToolTip("Despliega los parámetros reales del modelo IA y módulos DSP.")
        self._adv_toggle_btn.toggled.connect(self._on_advanced_toggled)

        intensity_row.addWidget(int_lbl)
        intensity_row.addWidget(self._intensity_slider, stretch=1)
        intensity_row.addWidget(self._intensity_val_lbl)
        intensity_row.addSpacing(12)
        intensity_row.addWidget(self._adv_toggle_btn)
        controls_layout.addLayout(intensity_row)

        root.addWidget(controls_frame)

        self.advanced_panel = DynamicParamPanel()
        self.advanced_panel.setVisible(False)
        self.advanced_panel.parameter_changed.connect(self.advanced_param_changed.emit)
        self.advanced_panel.reset_requested.connect(self.advanced_reset_requested.emit)
        root.addWidget(self.advanced_panel, stretch=2)

    def set_models_list(self, models: list[dict[str, Any]], selected_id: str | None = None) -> None:
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in models:
            self._model_combo.addItem(m["display_name"], m)
        if selected_id:
            for idx in range(self._model_combo.count()):
                data = self._model_combo.itemData(idx)
                if isinstance(data, dict) and data.get("model_id") == selected_id:
                    self._model_combo.setCurrentIndex(idx)
                    break
        self._model_combo.blockSignals(False)
        self._refresh_current_model_badge()

    def set_presets_list(self, presets: list[dict[str, Any]], selected_id: str | None = None) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for p in presets:
            label = f"{p['name']} (Personalizado)" if p.get("is_custom") else p["name"]
            self._preset_combo.addItem(label, p)
        if selected_id:
            for idx in range(self._preset_combo.count()):
                data = self._preset_combo.itemData(idx)
                if isinstance(data, dict) and data.get("preset_id") == selected_id:
                    self._preset_combo.setCurrentIndex(idx)
                    break
        self._preset_combo.blockSignals(False)
        self._update_delete_preset_state()

    def set_preset_notes(self, notes: list[str]) -> None:
        text = " | ".join(notes).strip()
        self._preset_note_lbl.setText(text)
        self._preset_note_lbl.setVisible(bool(text))

    def set_metadata(
        self,
        *,
        filename: str,
        duration_s: float,
        sample_rate: int,
        bit_depth: int | None,
        channels: int,
    ) -> None:
        self._file_name_lbl.setText(filename)
        self._meta_duration.setText(f"Duración: {duration_s:.2f} s")
        self._meta_sr.setText(f"Frecuencia: {sample_rate} Hz")
        self._meta_bits.setText(f"Profundidad: {bit_depth} bits" if bit_depth else "Profundidad: Comprimido/N/D")
        ch_text = "Mono (1)" if channels == 1 else f"Estéreo ({channels})" if channels == 2 else f"{channels} ch"
        self._meta_channels.setText(f"Canales: {ch_text}")
        self._play_orig_btn.setEnabled(True)

    def set_intensity(self, value: int) -> None:
        clamped = max(0, min(100, int(value)))
        self._intensity_slider.blockSignals(True)
        self._intensity_slider.setValue(clamped)
        self._intensity_slider.blockSignals(False)
        self._intensity_val_lbl.setText(f"{clamped} %")

    def set_advanced_schema(self, definitions: list[UIParamDefinition], values: dict[str, Any]) -> None:
        self.advanced_panel.set_schema(definitions, values)

    def _browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir grabación de audio",
            "",
            "Archivos de audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus *.wma *.aiff);;Todos (*.*)",
        )
        if file_path:
            self.open_file_requested.emit(file_path)

    def _update_visualizer_mode(self) -> None:
        mode = self._view_mode_combo.currentData()
        self.waveform.setVisible(mode in ("waveform", "split"))
        self.spectrogram.setVisible(mode in ("spectrogram", "split"))

    def _on_model_combo_changed(self) -> None:
        data = self._model_combo.currentData()
        self._refresh_current_model_badge()
        if isinstance(data, dict):
            self.model_selected.emit(str(data["model_id"]))

    def _refresh_current_model_badge(self) -> None:
        data = self._model_combo.currentData()
        if not isinstance(data, dict):
            return
        status = str(data.get("status_label", "No instalado"))
        status_code = str(data.get("status_code", "not_installed"))
        note = str(data.get("note", ""))

        self._model_status_badge.setText(status)
        if status_code in ("installed", "available"):
            self._model_status_badge.setObjectName("StatusBadgeOk")
        elif status_code in ("downloading", "preparing"):
            self._model_status_badge.setObjectName("StatusBadgeInfo")
        elif status_code in ("error", "incompatible"):
            self._model_status_badge.setObjectName("StatusBadgeError")
        else:
            self._model_status_badge.setObjectName("StatusBadgeWarn")
        self._model_status_badge.style().unpolish(self._model_status_badge)
        self._model_status_badge.style().polish(self._model_status_badge)
        self._model_note_lbl.setText(note)

    def _on_preset_combo_changed(self) -> None:
        self._update_delete_preset_state()
        data = self._preset_combo.currentData()
        if isinstance(data, dict):
            self.preset_selected.emit(str(data["preset_id"]))

    def _update_delete_preset_state(self) -> None:
        data = self._preset_combo.currentData()
        self._del_preset_btn.setEnabled(bool(isinstance(data, dict) and data.get("is_custom")))

    def _on_delete_preset(self) -> None:
        data = self._preset_combo.currentData()
        if isinstance(data, dict) and data.get("is_custom"):
            self.delete_custom_preset_requested.emit(str(data["preset_id"]))

    def _on_intensity_changed(self, value: int) -> None:
        self._intensity_val_lbl.setText(f"{value} %")
        self.intensity_changed.emit(value)

    def _on_advanced_toggled(self, checked: bool) -> None:
        self.advanced_panel.setVisible(checked)
        self._adv_toggle_btn.setText("OPCIONES AVANZADAS ^" if checked else "OPCIONES AVANZADAS v")
