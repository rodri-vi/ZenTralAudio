"""Pestaña 2: Ambientes Acústicos, simulación IR/algorítmica y mezclador multicapa."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.theme import Palette
from app.widgets.ambience_layer_widget import AmbienceLayerWidget

BUILTIN_ENVIRONMENT_NAMES: list[tuple[str, str]] = [
    ("estudio", "Estudio de grabación"),
    ("habitacion", "Habitación"),
    ("oficina", "Oficina"),
    ("pasillo", "Pasillo"),
    ("wc_bano", "WC o baño"),
    ("restaurante", "Restaurante"),
    ("cafeteria", "Cafetería"),
    ("iglesia", "Iglesia"),
    ("estacion", "Estación"),
    ("calle", "Calle"),
    ("exterior_urbano", "Exterior urbano"),
    ("bosque", "Bosque"),
    ("playa", "Playa"),
    ("montana", "Montaña"),
]


class EnvironmentTab(QWidget):
    """Configuración acústica del espacio y mezclador de capas sonoras ambientales."""

    environment_enabled_toggled = Signal(bool)
    environment_selected = Signal(str)
    acoustic_param_changed = Signal(str, object)
    load_local_ir_requested = Signal(str)
    clear_ir_requested = Signal()
    layer_changed = Signal(str, dict)
    layer_solo_toggled = Signal(str, bool)
    add_local_layer_requested = Signal(str)
    replace_layer_file_requested = Signal(str, str)
    remove_layer_requested = Signal(str)
    master_ambience_volume_changed = Signal(float)
    manage_resources_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_widgets: dict[str, AmbienceLayerWidget] = {}
        self._acoustic_sliders: dict[str, tuple[QSlider, QLabel, str, float]] = {}
        self._syncing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        top_group = QGroupBox("Entorno Acústico Destino")
        top_layout = QHBoxLayout(top_group)

        self._enable_env_chk = QCheckBox("Activar simulación de ambiente acústico")
        self._enable_env_chk.setChecked(False)
        self._enable_env_chk.setStyleSheet("font-weight: 600;")
        self._enable_env_chk.toggled.connect(self.environment_enabled_toggled.emit)

        self._env_combo = QComboBox()
        for env_id, env_name in BUILTIN_ENVIRONMENT_NAMES:
            self._env_combo.addItem(env_name, env_id)
        self._env_combo.currentIndexChanged.connect(
            lambda: self.environment_selected.emit(str(self._env_combo.currentData()))
        )

        self._resources_btn = QPushButton("Gestor de recursos ambientales...")
        self._resources_btn.setToolTip("Descargar bajo consentimiento o gestionar archivos de ambiente e IRs.")
        self._resources_btn.clicked.connect(self.manage_resources_requested.emit)

        top_layout.addWidget(self._enable_env_chk)
        top_layout.addSpacing(12)
        top_layout.addWidget(QLabel("Ambiente:"))
        top_layout.addWidget(self._env_combo, stretch=1)
        top_layout.addWidget(self._resources_btn)
        root.addWidget(top_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        acoustics_group = QGroupBox("Simulación Acústica (Convolución IR / Algorítmica)")
        ac_layout = QVBoxLayout(acoustics_group)
        ac_layout.setSpacing(8)

        ir_row = QHBoxLayout()
        self._ir_status_lbl = QLabel("Modo: Algorítmico paramétrico (sin IR cargada)")
        self._ir_status_lbl.setStyleSheet(f"color: {Palette.ACCENT_SECONDARY}; font-size: 12px;")
        self._load_ir_btn = QPushButton("Cargar IR local...")
        self._load_ir_btn.setToolTip("Cargar una respuesta impulsional WAV/FLAC real para convolución.")
        self._load_ir_btn.clicked.connect(self._browse_ir_file)
        self._clear_ir_btn = QPushButton("Usar algorítmica")
        self._clear_ir_btn.setEnabled(False)
        self._clear_ir_btn.clicked.connect(self.clear_ir_requested.emit)

        ir_row.addWidget(self._ir_status_lbl, stretch=1)
        ir_row.addWidget(self._load_ir_btn)
        ir_row.addWidget(self._clear_ir_btn)
        ac_layout.addLayout(ir_row)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        acoustic_specs = [
            ("room_size", "Tamaño aparente del espacio", 5, 100, 45, "%", 1.0, "Escala las reflexiones tempranas y la amplitud espacial."),
            ("reverb_mix", "Intensidad de reverberación", 0, 100, 28, "%", 1.0, "Cantidad global de reverberación aplicada a la voz."),
            ("decay_s", "Tiempo de decay", 10, 800, 120, "s", 0.01, "Tiempo de caída RT60 aproximado de la cola reverberante."),
            ("predelay_ms", "Predelay", 0, 120, 14, "ms", 1.0, "Retardo inicial antes de las primeras reflexiones del recinto."),
            ("damping", "Damping (absorción aguda)", 0, 100, 45, "%", 1.0, "Atenuación de frecuencias altas en la cola reverberante."),
            ("direct_level", "Nivel sonido directo", 0, 150, 100, "%", 1.0, "Nivel de la voz seca frente a la sala."),
            ("reflected_level", "Nivel sonido reflejado", 0, 150, 40, "%", 1.0, "Nivel de las reflexiones tempranas y cola difusa."),
            ("stereo_width", "Anchura estéreo", 0, 150, 100, "%", 1.0, "Amplitud de la imagen estéreo resultante."),
            ("apparent_distance", "Distancia aparente de la voz", 0, 100, 20, "%", 1.0, "Simula alejamiento mediante balance directo/reflejado y absorción aérea."),
            ("pan", "Panorámica de la voz", -100, 100, 0, "%", 1.0, "Posición izquierda (-100) / centro (0) / derecha (+100)."),
        ]

        for row, (key, label, min_i, max_i, def_i, unit, scale, tip) in enumerate(acoustic_specs):
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_i, max_i)
            slider.setValue(def_i)
            slider.setToolTip(tip)
            val_lbl = QLabel(self._format_acoustic_value(def_i, scale, unit))
            val_lbl.setFixedWidth(64)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            slider.valueChanged.connect(
                lambda iv, k=key, vl=val_lbl, sc=scale, u=unit: self._on_acoustic_slider(k, iv, vl, sc, u)
            )
            grid.addWidget(lbl, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val_lbl, row, 2)
            self._acoustic_sliders[key] = (slider, val_lbl, unit, scale)

        ac_layout.addLayout(grid)
        ac_layout.addStretch(1)
        splitter.addWidget(acoustics_group)

        mixer_group = QGroupBox("Mezclador de Capas Ambientales")
        mix_layout = QVBoxLayout(mixer_group)
        mix_layout.setSpacing(8)

        master_row = QHBoxLayout()
        master_row.addWidget(QLabel("Volumen general ambiente:"))
        self._master_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._master_vol_slider.setRange(0, 150)
        self._master_vol_slider.setValue(60)
        self._master_vol_slider.valueChanged.connect(self._on_master_vol_changed)
        self._master_vol_lbl = QLabel("60 %")
        self._master_vol_lbl.setFixedWidth(48)
        self._add_layer_btn = QPushButton("+ Añadir capa local...")
        self._add_layer_btn.setObjectName("SecondaryAccentButton")
        self._add_layer_btn.setToolTip("Añade una grabación ambiental local WAV/FLAC/MP3 como capa.")
        self._add_layer_btn.clicked.connect(self._browse_add_layer)

        master_row.addWidget(self._master_vol_slider, stretch=1)
        master_row.addWidget(self._master_vol_lbl)
        master_row.addWidget(self._add_layer_btn)
        mix_layout.addLayout(master_row)

        self._layers_scroll = QScrollArea()
        self._layers_scroll.setWidgetResizable(True)
        self._layers_container = QWidget()
        self._layers_layout = QVBoxLayout(self._layers_container)
        self._layers_layout.setContentsMargins(0, 0, 4, 0)
        self._layers_layout.setSpacing(8)
        self._layers_layout.addStretch(1)
        self._layers_scroll.setWidget(self._layers_container)
        mix_layout.addWidget(self._layers_scroll, stretch=1)

        splitter.addWidget(mixer_group)
        splitter.setSizes([480, 520])
        root.addWidget(splitter, stretch=1)

    def set_layers(self, layers: list[dict[str, Any]]) -> None:
        while self._layers_layout.count():
            item = self._layers_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._layer_widgets.clear()

        if not layers:
            empty = QLabel("Este ambiente no tiene capas activas. Puedes añadir grabaciones locales o descargar recursos verificados.")
            empty.setObjectName("SecondaryText")
            empty.setWordWrap(True)
            self._layers_layout.addWidget(empty)
        else:
            for layer_data in layers:
                widget = AmbienceLayerWidget(layer_data)
                widget.layer_changed.connect(self.layer_changed.emit)
                widget.solo_toggled.connect(self.layer_solo_toggled.emit)
                widget.replace_requested.connect(self._browse_replace_layer)
                widget.remove_requested.connect(self.remove_layer_requested.emit)
                self._layers_layout.addWidget(widget)
                self._layer_widgets[widget.layer_id] = widget

        self._layers_layout.addStretch(1)

    def set_ir_status(self, ir_name: str | None) -> None:
        if ir_name:
            self._ir_status_lbl.setText(f"Modo: Convolución IR ({ir_name})")
            self._ir_status_lbl.setStyleSheet(f"color: {Palette.ACCENT_PRIMARY}; font-size: 12px; font-weight: 600;")
            self._clear_ir_btn.setEnabled(True)
        else:
            self._ir_status_lbl.setText("Modo: Algorítmico paramétrico (sin IR cargada)")
            self._ir_status_lbl.setStyleSheet(f"color: {Palette.ACCENT_SECONDARY}; font-size: 12px;")
            self._clear_ir_btn.setEnabled(False)

    def set_acoustic_values(self, values: dict[str, float]) -> None:
        self._syncing = True
        try:
            for key, (slider, lbl, unit, scale) in self._acoustic_sliders.items():
                if key not in values:
                    continue
                iv = int(round(float(values[key]) / scale))
                iv = max(slider.minimum(), min(slider.maximum(), iv))
                slider.setValue(iv)
                lbl.setText(self._format_acoustic_value(iv, scale, unit))
        finally:
            self._syncing = False

    def _on_acoustic_slider(self, key: str, iv: int, lbl: QLabel, scale: float, unit: str) -> None:
        lbl.setText(self._format_acoustic_value(iv, scale, unit))
        if not self._syncing:
            self.acoustic_param_changed.emit(key, round(iv * scale, 4))

    def _on_master_vol_changed(self, value: int) -> None:
        self._master_vol_lbl.setText(f"{value} %")
        self.master_ambience_volume_changed.emit(value / 100.0)

    def _browse_ir_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Respuesta Impulsional (IR)",
            "",
            "Audio IR (*.wav *.flac);;Todos (*.*)",
        )
        if path:
            self.load_local_ir_requested.emit(path)

    def _browse_add_layer(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Añadir capa de ambiente local",
            "",
            "Archivos de audio (*.wav *.flac *.mp3 *.ogg *.m4a);;Todos (*.*)",
        )
        if path:
            self.add_local_layer_requested.emit(path)

    def _browse_replace_layer(self, layer_id: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Asignar archivo de audio a la capa",
            "",
            "Archivos de audio (*.wav *.flac *.mp3 *.ogg *.m4a);;Todos (*.*)",
        )
        if path:
            self.replace_layer_file_requested.emit(layer_id, path)

    @staticmethod
    def _format_acoustic_value(iv: int, scale: float, unit: str) -> str:
        val = iv * scale
        if scale < 1.0:
            return f"{val:.2f} {unit}"
        return f"{int(round(val))} {unit}"
