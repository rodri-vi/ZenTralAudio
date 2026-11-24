"""Panel dinámico de opciones avanzadas construido desde esquemas reales de parámetros."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.theme import Palette


@dataclass
class UIParamDefinition:
    """Definición declarativa de un parámetro real conectable a la cadena."""

    key: str
    label: str
    category: str
    param_type: str  # "float" | "int" | "bool" | "enum"
    default: Any
    min_val: float = 0.0
    max_val: float = 1.0
    step: float = 0.05
    decimals: int = 2
    unit: str = ""
    tooltip: str = ""
    options: list[tuple[str, Any]] = field(default_factory=list)


class DynamicParamPanel(QWidget):
    """Genera controles por categoría estrictamente a partir de parámetros implementados."""

    parameter_changed = Signal(str, object)  # (key, new_value)
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._definitions: dict[str, UIParamDefinition] = {}
        self._widgets: dict[str, QWidget] = {}
        self._values: dict[str, Any] = {}
        self._syncing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Parámetros Avanzados Reales de la Cadena")
        title.setStyleSheet(f"color: {Palette.ACCENT_SECONDARY}; font-weight: 600;")
        self._reset_btn = QPushButton("Restablecer valores")
        self._reset_btn.setToolTip("Devuelve los parámetros avanzados a los valores del preset/modelo.")
        self._reset_btn.clicked.connect(self.reset_requested.emit)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._reset_btn)
        outer.addLayout(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(2, 2, 8, 2)
        self._container_layout.setSpacing(10)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, stretch=1)

    def set_schema(
        self,
        definitions: list[UIParamDefinition],
        initial_values: dict[str, Any] | None = None,
    ) -> None:
        """Reconstruye el panel de controles según el esquema actual del modelo + DSP."""
        self._clear_layout(self._container_layout)
        self._definitions = {d.key: d for d in definitions}
        self._widgets.clear()
        self._values.clear()

        if not definitions:
            empty_lbl = QLabel("No hay parámetros avanzados configurables para la selección actual.")
            empty_lbl.setObjectName("SecondaryText")
            self._container_layout.addWidget(empty_lbl)
            self._container_layout.addStretch(1)
            return

        grouped: dict[str, list[UIParamDefinition]] = {}
        for item in definitions:
            grouped.setdefault(item.category, []).append(item)

        values = initial_values or {}
        for category, items in grouped.items():
            box = QFrame()
            box.setObjectName("PanelSecondary")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 8, 10, 10)
            box_layout.setSpacing(8)

            cat_label = QLabel(category.upper())
            cat_label.setStyleSheet(f"color: {Palette.ACCENT_PRIMARY}; font-size: 11px; font-weight: 700;")
            box_layout.addWidget(cat_label)

            grid = QGridLayout()
            grid.setColumnStretch(0, 2)
            grid.setColumnStretch(1, 3)
            grid.setColumnStretch(2, 1)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(6)

            for row, param in enumerate(items):
                current_val = values.get(param.key, param.default)
                self._values[param.key] = current_val

                lbl = QLabel(param.label)
                lbl.setToolTip(param.tooltip or param.label)
                grid.addWidget(lbl, row, 0)

                if param.param_type == "bool":
                    chk = QCheckBox("Activado")
                    chk.setChecked(bool(current_val))
                    chk.setToolTip(param.tooltip)
                    chk.toggled.connect(lambda checked, k=param.key: self._on_value_edited(k, bool(checked)))
                    grid.addWidget(chk, row, 1, 1, 2)
                    self._widgets[param.key] = chk

                elif param.param_type == "enum":
                    combo = QComboBox()
                    combo.setToolTip(param.tooltip)
                    for opt_label, opt_val in param.options:
                        combo.addItem(opt_label, opt_val)
                    idx = combo.findData(current_val)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    combo.currentIndexChanged.connect(
                        lambda _i, c=combo, k=param.key: self._on_value_edited(k, c.currentData())
                    )
                    grid.addWidget(combo, row, 1, 1, 2)
                    self._widgets[param.key] = combo

                elif param.param_type == "int":
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setRange(int(param.min_val), int(param.max_val))
                    slider.setValue(int(current_val))
                    slider.setToolTip(param.tooltip)

                    spin = QSpinBox()
                    spin.setRange(int(param.min_val), int(param.max_val))
                    spin.setSingleStep(max(1, int(param.step)))
                    spin.setValue(int(current_val))
                    if param.unit:
                        spin.setSuffix(f" {param.unit}")
                    spin.setToolTip(param.tooltip)

                    slider.valueChanged.connect(lambda v, s=spin: self._sync_spin(s, v))
                    spin.valueChanged.connect(lambda v, sl=slider, k=param.key: self._sync_slider_emit(sl, k, int(v)))

                    grid.addWidget(slider, row, 1)
                    grid.addWidget(spin, row, 2)
                    self._widgets[param.key] = spin

                else:  # float
                    scale = 10 ** param.decimals
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setRange(int(round(param.min_val * scale)), int(round(param.max_val * scale)))
                    slider.setValue(int(round(float(current_val) * scale)))
                    slider.setToolTip(param.tooltip)

                    spin = QDoubleSpinBox()
                    spin.setRange(float(param.min_val), float(param.max_val))
                    spin.setDecimals(param.decimals)
                    spin.setSingleStep(float(param.step))
                    spin.setValue(float(current_val))
                    if param.unit:
                        spin.setSuffix(f" {param.unit}")
                    spin.setToolTip(param.tooltip)

                    slider.valueChanged.connect(
                        lambda iv, s=spin, sc=scale: self._sync_spin(s, iv / sc)
                    )
                    spin.valueChanged.connect(
                        lambda fv, sl=slider, sc=scale, k=param.key: self._sync_slider_emit(
                            sl, k, round(float(fv), param.decimals), int(round(float(fv) * sc))
                        )
                    )

                    grid.addWidget(slider, row, 1)
                    grid.addWidget(spin, row, 2)
                    self._widgets[param.key] = spin

            box_layout.addLayout(grid)
            self._container_layout.addWidget(box)

        self._container_layout.addStretch(1)

    def set_values(self, values: dict[str, Any]) -> None:
        """Actualiza valores visibles en el panel sin disparar bucles de señales."""
        self._syncing = True
        try:
            for key, val in values.items():
                widget = self._widgets.get(key)
                if widget is None:
                    continue
                self._values[key] = val
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(val))
                elif isinstance(widget, QComboBox):
                    idx = widget.findData(val)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(val))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(val))
        finally:
            self._syncing = False

    def get_values(self) -> dict[str, Any]:
        return dict(self._values)

    def _sync_spin(self, spin: QSpinBox | QDoubleSpinBox, value: int | float) -> None:
        if spin.value() != value:
            spin.setValue(value)

    def _sync_slider_emit(
        self,
        slider: QSlider,
        key: str,
        value: int | float,
        slider_int: int | None = None,
    ) -> None:
        target = int(value) if slider_int is None else slider_int
        if slider.value() != target:
            slider.blockSignals(True)
            slider.setValue(target)
            slider.blockSignals(False)
        self._on_value_edited(key, value)

    def _on_value_edited(self, key: str, value: Any) -> None:
        self._values[key] = value
        if not self._syncing:
            self.parameter_changed.emit(key, value)

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
