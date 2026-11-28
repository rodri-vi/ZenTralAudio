"""Diálogo de gestión de recursos ambientales y respuestas impulsionales (IR)."""

from __future__ import annotations

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
from core.platform_utils import open_in_file_manager


class ResourceManagerDialog(QDialog):
    """Gestiona descargas autorizadas, eliminación y licencias de sonidos e IRs."""

    download_resource_requested = Signal(str)
    cancel_download_requested = Signal()
    delete_resource_requested = Signal(str)

    def __init__(
        self,
        paths: AppPaths,
        resources: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gestor de Recursos Ambientales e IR — ZenTralAudio")
        self.resize(920, 460)
        self._paths = paths
        self._resources = resources
        self._busy = False

        layout = QVBoxLayout(self)
        info = QLabel(
            "Los sonidos ambientales y respuestas impulsionales documentan su fuente, autor y licencia. "
            "Puedes importar tus propios archivos WAV/FLAC/MP3 desde la pestaña Ambientes o descargar recursos verificados."
        )
        info.setObjectName("SecondaryText")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Recurso", "Tipo / Ambiente", "Fuente y Autor", "Licencia", "Estado", "Restricciones / Notas"]
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
        self._cancel_dl_btn = QPushButton("Cancelar")
        self._cancel_dl_btn.setObjectName("DangerButton")
        self._cancel_dl_btn.setVisible(False)
        self._cancel_dl_btn.clicked.connect(self.cancel_download_requested.emit)
        prog_row.addWidget(self._status_lbl, stretch=1)
        prog_row.addWidget(self._progress)
        prog_row.addWidget(self._cancel_dl_btn)
        layout.addLayout(prog_row)

        btn_row = QHBoxLayout()
        self._dl_btn = QPushButton("Descargar recurso...")
        self._dl_btn.setObjectName("PrimaryButton")
        self._dl_btn.clicked.connect(self._on_download)

        self._del_btn = QPushButton("Eliminar archivo local")
        self._del_btn.setObjectName("DangerButton")
        self._del_btn.clicked.connect(self._on_delete)

        open_btn = QPushButton("Abrir carpeta ambience_data")
        open_btn.clicked.connect(self._open_ambience_dir)

        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._dl_btn)
        btn_row.addWidget(self._del_btn)
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.populate(resources)

    def populate(self, resources: list[dict[str, Any]]) -> None:
        self._resources = resources
        self._table.setRowCount(len(resources))
        for row, r in enumerate(resources):
            self._table.setItem(row, 0, QTableWidgetItem(str(r.get("name", r["id"]))))
            self._table.setItem(row, 1, QTableWidgetItem(str(r.get("kind_label", ""))))
            self._table.setItem(row, 2, QTableWidgetItem(str(r.get("source_author", ""))))
            self._table.setItem(row, 3, QTableWidgetItem(str(r.get("license", ""))))
            self._table.setItem(row, 4, QTableWidgetItem(str(r.get("status_label", "No descargado"))))
            self._table.setItem(row, 5, QTableWidgetItem(str(r.get("restrictions", ""))))
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

    def _selected(self) -> dict[str, Any] | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._resources):
            return self._resources[row]
        return None

    def _update_buttons(self) -> None:
        r = self._selected()
        if r is None or self._busy:
            self._dl_btn.setEnabled(False)
            self._del_btn.setEnabled(False)
            return
        available = r.get("status_code") == "available"
        has_url = bool(r.get("download_url"))
        self._dl_btn.setEnabled(has_url and not available)
        self._del_btn.setEnabled(available)

    def _on_download(self) -> None:
        r = self._selected()
        if r is None:
            return
        ans = QMessageBox.question(
            self,
            "Confirmar descarga de recurso",
            f"¿Autorizas descargar '{r['name']}'?\n\nFuente: {r.get('download_url', '')}\nLicencia: {r.get('license', '')}",
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.download_resource_requested.emit(str(r["id"]))

    def _on_delete(self) -> None:
        r = self._selected()
        if r is None:
            return
        ans = QMessageBox.question(
            self,
            "Eliminar recurso local",
            f"¿Eliminar '{r['name']}' de ambience_data?",
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.delete_resource_requested.emit(str(r["id"]))

    def _open_ambience_dir(self) -> None:
        open_in_file_manager(self._paths.ambience_dir)
