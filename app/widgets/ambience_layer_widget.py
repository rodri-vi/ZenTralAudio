"""Control de canal individual para una capa de sonido ambiental."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.theme import Palette


class AmbienceLayerWidget(QFrame):
    """Fila de mezclador para activar, ajustar volumen, pan, solo y archivo de una capa."""

    layer_changed = Signal(str, dict)  # (layer_id, {"enabled", "volume", "pan", "solo"})
    solo_toggled = Signal(str, bool)
    replace_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, layer_data: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelSecondary")
        self.layer_id = str(layer_data["layer_id"])
        self._syncing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        top_row = QHBoxLayout()
        self._enable_chk = QCheckBox(str(layer_data.get("name", self.layer_id)))
        self._enable_chk.setChecked(bool(layer_data.get("enabled", True)))
        self._enable_chk.setStyleSheet("font-weight: 600;")
        self._enable_chk.toggled.connect(self._emit_state)

        self._file_lbl = QLabel(self._format_file_label(layer_data))
        self._file_lbl.setObjectName("SecondaryText")
        self._file_lbl.setToolTip(str(layer_data.get("license_info", "Sin información de archivo.")))

        self._solo_btn = QPushButton("S")
        self._solo_btn.setObjectName("SoloButton")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedWidth(28)
        self._solo_btn.setChecked(bool(layer_data.get("solo", False)))
        self._solo_btn.setToolTip("Escuchar esta capa ambiental en Solo.")
        self._solo_btn.toggled.connect(self._on_solo)

        self._replace_btn = QPushButton("Archivo...")
        self._replace_btn.setToolTip("Asignar o sustituir con un archivo WAV/FLAC/MP3 local.")
        self._replace_btn.clicked.connect(lambda: self.replace_requested.emit(self.layer_id))

        self._remove_btn = QPushButton("Quitar")
        self._remove_btn.setObjectName("DangerButton")
        self._remove_btn.setToolTip("Eliminar esta capa del mezclador actual.")
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.layer_id))

        top_row.addWidget(self._enable_chk)
        top_row.addWidget(self._file_lbl, stretch=1)
        top_row.addWidget(self._solo_btn)
        top_row.addWidget(self._replace_btn)
        top_row.addWidget(self._remove_btn)
        root.addLayout(top_row)

        controls_row = QHBoxLayout()
        vol_lbl = QLabel("Volumen:")
        vol_lbl.setObjectName("SecondaryText")
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 150)
        self._vol_slider.setValue(int(round(float(layer_data.get("volume", 0.35)) * 100)))
        self._vol_slider.setToolTip("Nivel relativo de mezcla de la capa ambiental (0 a 150 %).")
        self._vol_val_lbl = QLabel(f"{self._vol_slider.value()} %")
        self._vol_val_lbl.setFixedWidth(44)
        self._vol_slider.valueChanged.connect(self._on_vol_changed)

        pan_lbl = QLabel("Pan:")
        pan_lbl.setObjectName("SecondaryText")
        self._pan_slider = QSlider(Qt.Orientation.Horizontal)
        self._pan_slider.setRange(-100, 100)
        self._pan_slider.setValue(int(round(float(layer_data.get("pan", 0.0)) * 100)))
        self._pan_slider.setMaximumWidth(130)
        self._pan_slider.setToolTip("Posición estéreo izquierda (-100) a derecha (+100).")
        self._pan_val_lbl = QLabel(self._format_pan(self._pan_slider.value()))
        self._pan_val_lbl.setFixedWidth(36)
        self._pan_slider.valueChanged.connect(self._on_pan_changed)

        controls_row.addWidget(vol_lbl)
        controls_row.addWidget(self._vol_slider, stretch=1)
        controls_row.addWidget(self._vol_val_lbl)
        controls_row.addSpacing(10)
        controls_row.addWidget(pan_lbl)
        controls_row.addWidget(self._pan_slider)
        controls_row.addWidget(self._pan_val_lbl)
        root.addLayout(controls_row)

    def update_file_info(self, layer_data: dict[str, Any]) -> None:
        self._file_lbl.setText(self._format_file_label(layer_data))
        self._file_lbl.setToolTip(str(layer_data.get("license_info", "")))

    def _on_vol_changed(self, value: int) -> None:
        self._vol_val_lbl.setText(f"{value} %")
        self._emit_state()

    def _on_pan_changed(self, value: int) -> None:
        self._pan_val_lbl.setText(self._format_pan(value))
        self._emit_state()

    def _on_solo(self, checked: bool) -> None:
        if not self._syncing:
            # The controller applies exclusive Solo state to all layer widgets.
            # Emitting a second generic state event here caused duplicate renders.
            self.solo_toggled.emit(self.layer_id, checked)

    def set_solo_checked(self, checked: bool) -> None:
        """Sincroniza el botón Solo sin volver a emitir señales."""
        self._syncing = True
        try:
            self._solo_btn.setChecked(bool(checked))
        finally:
            self._syncing = False

    def _emit_state(self) -> None:
        if self._syncing:
            return
        self.layer_changed.emit(
            self.layer_id,
            {
                "enabled": self._enable_chk.isChecked(),
                "volume": self._vol_slider.value() / 100.0,
                "pan": self._pan_slider.value() / 100.0,
                "solo": self._solo_btn.isChecked(),
            },
        )

    @staticmethod
    def _format_pan(pan_int: int) -> str:
        if pan_int == 0:
            return "C"
        return f"L{abs(pan_int)}" if pan_int < 0 else f"R{pan_int}"

    @staticmethod
    def _format_file_label(layer_data: dict[str, Any]) -> str:
        file_name = layer_data.get("filename")
        status = layer_data.get("status", "Sin archivo asignado")
        if file_name:
            return f"[{status}] {file_name}"
        return f"[{status}]"
