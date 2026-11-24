"""Diálogo de gestión, verificación, progreso de descarga y licencias de modelos IA."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.paths import AppPaths


class ModelManagerDialog(QDialog):
    """Muestra estado real, licencias, requisitos, descarga con progreso y eliminación."""

    download_requested = Signal(str)
    cancel_download_requested = Signal()
    delete_requested = Signal(str)
    refresh_requested = Signal()

    def __init__(self, paths: AppPaths, models: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gestor de Modelos IA — ZenTralAudio")
        self.resize(920, 470)
        self._paths = paths
        self._models = models
        self._busy = False

        layout = QVBoxLayout(self)
        info = QLabel(
            "Ningún peso de modelo se descarga sin tu consentimiento explícito. "
            "Los pesos descargados se almacenan en la carpeta portable 'models_data/'."
        )
        info.setObjectName("SecondaryText")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Modelo", "Frecuencia", "Licencia (Código / Pesos)", "Estado", "Tamaño aprox.", "Notas técnicas"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self._table, stretch=1)

        prog_row = QHBoxLayout()
        self._status_lbl = QLabel("Listo")
        self._status_lbl.setObjectName("SecondaryText")
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._cancel_dl_btn = QPushButton("Cancelar descarga")
        self._cancel_dl_btn.setObjectName("DangerButton")
        self._cancel_dl_btn.setVisible(False)
        self._cancel_dl_btn.clicked.connect(self.cancel_download_requested.emit)
        prog_row.addWidget(self._status_lbl, stretch=1)
        prog_row.addWidget(self._progress)
        prog_row.addWidget(self._cancel_dl_btn)
        layout.addLayout(prog_row)

        btn_row = QHBoxLayout()
        self._download_btn = QPushButton("Descargar / Instalar...")
        self._download_btn.setObjectName("PrimaryButton")
        self._download_btn.clicked.connect(self._on_download)

        self._delete_btn = QPushButton("Eliminar pesos locales")
        self._delete_btn.setObjectName("DangerButton")
        self._delete_btn.clicked.connect(self._on_delete)

        self._open_dir_btn = QPushButton("Abrir carpeta models_data")
        self._open_dir_btn.clicked.connect(self._open_models_dir)

        self._refresh_btn = QPushButton("Actualizar estado")
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)

        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._download_btn)
        btn_row.addWidget(self._delete_btn)
        btn_row.addWidget(self._open_dir_btn)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.populate(models)

    def populate(self, models: list[dict[str, Any]]) -> None:
        self._models = models
        self._table.setRowCount(len(models))
        for row, m in enumerate(models):
            self._table.setItem(row, 0, QTableWidgetItem(str(m.get("display_name", m["model_id"]))))
            self._table.setItem(row, 1, QTableWidgetItem(f"{m.get('sample_rate', '—')} Hz"))
            self._table.setItem(row, 2, QTableWidgetItem(str(m.get("license_summary", "Ver repositorio"))))
            self._table.setItem(row, 3, QTableWidgetItem(str(m.get("status_label", "No instalado"))))
            self._table.setItem(row, 4, QTableWidgetItem(str(m.get("size_label", "N/D"))))
            self._table.setItem(row, 5, QTableWidgetItem(str(m.get("note", ""))))
        self._update_buttons()

    def set_download_progress(self, busy: bool, message: str, fraction: float = 0.0) -> None:
        self._busy = busy
        self._status_lbl.setText(message)
        self._progress.setVisible(busy)
        self._cancel_dl_btn.setVisible(busy)
        if busy:
            self._progress.setRange(0, 100)
            self._progress.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        self._update_buttons()

    def _selected_model(self) -> dict[str, Any] | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._models):
            return self._models[row]
        return None

    def _update_buttons(self) -> None:
        m = self._selected_model()
        if m is None or self._busy:
            self._download_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._refresh_btn.setEnabled(not self._busy)
            return
        self._refresh_btn.setEnabled(True)
        can_dl = bool(m.get("download_supported", False))
        installed = m.get("status_code") in ("installed", "available")
        has_files = installed or m.get("status_code") in ("incompatible", "error")
        self._download_btn.setEnabled(can_dl and not installed)
        self._delete_btn.setEnabled(has_files)

    def _on_download(self) -> None:
        m = self._selected_model()
        if m is None:
            return
        answer = QMessageBox.question(
            self,
            "Confirmar descarga de modelo",
            f"¿Autorizas descargar los pesos de '{m['display_name']}'?\n\n"
            f"Fuente: {m.get('source_url', 'Oficial')}\n"
            f"Licencia: {m.get('license_summary', '')}",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.download_requested.emit(str(m["model_id"]))

    def _on_delete(self) -> None:
        m = self._selected_model()
        if m is None:
            return
        answer = QMessageBox.question(
            self,
            "Eliminar modelo local",
            f"¿Seguro que deseas eliminar los archivos locales de '{m['display_name']}'?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(str(m["model_id"]))

    def _open_models_dir(self) -> None:
        self._paths.models_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(self._paths.models_dir))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(self._paths.models_dir)])
