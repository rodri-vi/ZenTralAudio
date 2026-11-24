"""Diálogo de ajustes generales y diagnóstico portable de ZenTralAudio."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.configuration import AppSettings, ConfigManager
from core.paths import AppPaths


class SettingsDialog(QDialog):
    """Permite configurar dispositivo de cómputo, salida de audio, FFmpeg y rutas portables."""

    def __init__(
        self,
        paths: AppPaths,
        config_manager: ConfigManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ajustes de ZenTralAudio")
        self.resize(580, 420)
        self._paths = paths
        self._config_manager = config_manager
        settings = config_manager.settings

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self._device_combo = QComboBox()
        self._device_combo.addItem("Automático (GPU CUDA si existe, si no CPU)", "auto")
        self._device_combo.addItem("Solo CPU", "cpu")
        self._device_combo.addItem("GPU NVIDIA (CUDA)", "cuda")
        idx = self._device_combo.findData(settings.preferred_device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)
        form.addRow("Dispositivo IA:", self._device_combo)

        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 16)
        self._workers_spin.setValue(settings.max_worker_threads)
        form.addRow("Hilos de procesamiento:", self._workers_spin)

        self._audio_out_combo = QComboBox()
        self._populate_audio_devices(settings.output_device_index)
        form.addRow("Salida de audio:", self._audio_out_combo)

        ffmpeg_row = QHBoxLayout()
        self._ffmpeg_edit = QLineEdit(settings.ffmpeg_custom_path or "")
        self._ffmpeg_edit.setPlaceholderText("Automático (bin/ffmpeg.exe o PATH del sistema)")
        browse_ffmpeg = QPushButton("Buscar...")
        browse_ffmpeg.clicked.connect(self._browse_ffmpeg)
        test_ffmpeg = QPushButton("Verificar")
        test_ffmpeg.clicked.connect(self._check_ffmpeg)
        ffmpeg_row.addWidget(self._ffmpeg_edit, stretch=1)
        ffmpeg_row.addWidget(browse_ffmpeg)
        ffmpeg_row.addWidget(test_ffmpeg)
        form.addRow("Ejecutable FFmpeg:", ffmpeg_row)

        self._loudness_chk = QCheckBox("Activar compensación de sonoridad en comparación ON/OFF")
        self._loudness_chk.setChecked(settings.loudness_match_preview)
        form.addRow("", self._loudness_chk)

        self._consent_chk = QCheckBox("Recordar consentimiento para descargas externas autorizadas")
        self._consent_chk.setChecked(settings.consent_external_downloads)
        form.addRow("", self._consent_chk)

        self._debug_chk = QCheckBox("Activar registro detallado (DEBUG) en logs")
        self._debug_chk.setChecked(settings.debug_logging)
        form.addRow("", self._debug_chk)

        layout.addLayout(form)

        portable_lbl = QLabel(f"Raíz portable actual: {paths.app_root}")
        portable_lbl.setObjectName("SecondaryText")
        portable_lbl.setWordWrap(True)
        layout.addWidget(portable_lbl)

        folder_row = QHBoxLayout()
        open_root_btn = QPushButton("Abrir carpeta portable")
        open_root_btn.clicked.connect(lambda: self._open_folder(paths.app_root))
        open_logs_btn = QPushButton("Abrir carpeta de logs")
        open_logs_btn.clicked.connect(lambda: self._open_folder(paths.logs_dir))
        folder_row.addWidget(open_root_btn)
        folder_row.addWidget(open_logs_btn)
        folder_row.addStretch(1)
        layout.addLayout(folder_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_audio_devices(self, current_index: int | None) -> None:
        self._audio_out_combo.addItem("Dispositivo predeterminado del sistema", None)
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if int(dev.get("max_output_channels", 0)) > 0:
                    self._audio_out_combo.addItem(f"[{idx}] {dev['name']}", idx)
        except Exception:
            pass
        found = self._audio_out_combo.findData(current_index)
        if found >= 0:
            self._audio_out_combo.setCurrentIndex(found)

    def _browse_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar ejecutable ffmpeg",
            "",
            "Ejecutables (*.exe);;Todos (*.*)",
        )
        if path:
            self._ffmpeg_edit.setText(path)

    def _check_ffmpeg(self) -> None:
        custom = self._ffmpeg_edit.text().strip() or None
        found = self._paths.find_ffmpeg(custom)
        if found is None:
            QMessageBox.warning(
                self,
                "FFmpeg no detectado",
                "No se encontró ffmpeg. Colócalo en la carpeta 'bin/' o añade ffmpeg al PATH para habilitar MP3/M4A/AAC.",
            )
        else:
            QMessageBox.information(self, "FFmpeg detectado", f"FFmpeg válido encontrado en:\n{found}")

    def _open_folder(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def _save_and_accept(self) -> None:
        cur = self._config_manager.settings
        updated = AppSettings(
            preferred_device=str(self._device_combo.currentData()),
            max_worker_threads=int(self._workers_spin.value()),
            output_device_index=self._audio_out_combo.currentData(),
            loudness_match_preview=self._loudness_chk.isChecked(),
            preview_monitor_volume=cur.preview_monitor_volume,
            ffmpeg_custom_path=self._ffmpeg_edit.text().strip() or None,
            last_open_dir=cur.last_open_dir,
            last_export_dir=cur.last_export_dir,
            last_selected_model=cur.last_selected_model,
            consent_external_downloads=self._consent_chk.isChecked(),
            debug_logging=self._debug_chk.isChecked(),
            recent_files=list(cur.recent_files),
        )
        self._config_manager.save(updated)
        self.accept()
