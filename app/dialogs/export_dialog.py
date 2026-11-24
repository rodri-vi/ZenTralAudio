"""Diálogo de configuración y lanzamiento de exportación de audio."""

from __future__ import annotations

from dataclasses import dataclass
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
    QVBoxLayout,
    QWidget,
)

from core.exceptions import ValidationError
from core.validation import (
    VALID_BIT_DEPTHS,
    VALID_MP3_BITRATES_KBPS,
    VALID_SAMPLE_RATES,
    sanitize_filename,
    validate_export_target,
)


@dataclass(frozen=True)
class ExportRequest:
    """Parámetros validados para exportar con la cadena actual."""

    output_path: Path
    export_format: str
    sample_rate: int
    bit_depth: int
    mp3_bitrate_kbps: int
    channels: int
    include_environment: bool
    allow_overwrite: bool


class ExportDialog(QDialog):
    """Permite configurar formato, frecuencia, canales, alcance y destino sin sobrescribir el original."""

    def __init__(
        self,
        *,
        default_dir: Path,
        source_path: Path | None,
        source_sample_rate: int,
        source_channels: int,
        ffmpeg_available: bool,
        environment_active: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exportar Audio — ZenTralAudio")
        self.resize(560, 380)
        self._source_path = source_path
        self._ffmpeg_available = ffmpeg_available
        self.export_request: ExportRequest | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(str(default_dir))
        dir_btn = QPushButton("Elegir...")
        dir_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(self._dir_edit, stretch=1)
        dir_row.addWidget(dir_btn)
        form.addRow("Carpeta destino:", dir_row)

        base = sanitize_filename(source_path.stem if source_path else "zentralaudio_export") + "_restaurado"
        self._name_edit = QLineEdit(base)
        form.addRow("Nombre archivo:", self._name_edit)

        self._format_combo = QComboBox()
        self._format_combo.addItem("WAV (sin compresión PCM)", "wav")
        self._format_combo.addItem("FLAC (sin pérdida comprimido)", "flac")
        mp3_label = "MP3 (LAME vía FFmpeg)" if ffmpeg_available else "MP3 (requiere FFmpeg detectado)"
        self._format_combo.addItem(mp3_label, "mp3")
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Formato:", self._format_combo)

        self._sr_combo = QComboBox()
        for sr in VALID_SAMPLE_RATES:
            label = f"{sr} Hz (Original)" if sr == source_sample_rate else f"{sr} Hz"
            self._sr_combo.addItem(label, sr)
        idx_sr = self._sr_combo.findData(source_sample_rate)
        if idx_sr >= 0:
            self._sr_combo.setCurrentIndex(idx_sr)
        form.addRow("Frecuencia de muestreo:", self._sr_combo)

        self._bits_combo = QComboBox()
        for bits in VALID_BIT_DEPTHS:
            self._bits_combo.addItem(f"{bits} bits", bits)
        self._bits_combo.setCurrentIndex(1)
        form.addRow("Profundidad de bits:", self._bits_combo)

        self._bitrate_combo = QComboBox()
        for br in VALID_MP3_BITRATES_KBPS:
            self._bitrate_combo.addItem(f"{br} kbps", br)
        self._bitrate_combo.setCurrentIndex(4)
        form.addRow("Bitrate MP3:", self._bitrate_combo)

        self._channels_combo = QComboBox()
        self._channels_combo.addItem("Mono (1 canal)", 1)
        self._channels_combo.addItem("Estéreo (2 canales)", 2)
        self._channels_combo.setCurrentIndex(0 if source_channels == 1 else 1)
        form.addRow("Canales de salida:", self._channels_combo)

        self._scope_combo = QComboBox()
        self._scope_combo.addItem("Audio restaurado sin ambiente", False)
        self._scope_combo.addItem("Audio restaurado con ambiente acústico", True)
        self._scope_combo.setCurrentIndex(1 if environment_active else 0)
        form.addRow("Contenido a exportar:", self._scope_combo)

        self._overwrite_chk = QCheckBox("Permitir sobrescribir si el archivo de destino ya existe")
        form.addRow("", self._overwrite_chk)

        layout.addLayout(form)

        parity_note = QLabel(
            "La exportación ejecutará exactamente la misma cadena de procesamiento, orden de efectos y "
            "capas que escuchas en el reproductor de previsualización."
        )
        parity_note.setObjectName("SecondaryText")
        parity_note.setWordWrap(True)
        layout.addWidget(parity_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Exportar ahora")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_format_changed()

    def _on_format_changed(self) -> None:
        fmt = str(self._format_combo.currentData())
        self._bits_combo.setEnabled(fmt in ("wav", "flac"))
        self._bitrate_combo.setEnabled(fmt == "mp3")

    def _browse_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de exportación", self._dir_edit.text())
        if chosen:
            self._dir_edit.setText(chosen)

    def _validate_and_accept(self) -> None:
        fmt = str(self._format_combo.currentData())
        if fmt == "mp3" and not self._ffmpeg_available:
            QMessageBox.warning(
                self,
                "FFmpeg requerido para MP3",
                "No se detectó ffmpeg.exe. Configúralo en Ajustes o exporta en WAV/FLAC.",
            )
            return

        folder = Path(self._dir_edit.text().strip())
        stem = sanitize_filename(self._name_edit.text().strip(), fallback="zentralaudio_export")
        out_path = folder / f"{stem}.{fmt}"
        sample_rate = int(self._sr_combo.currentData())
        allow_overwrite = self._overwrite_chk.isChecked()

        try:
            validated = validate_export_target(
                out_path,
                source_path=self._source_path,
                export_format=fmt,
                sample_rate=sample_rate,
                allow_overwrite=allow_overwrite,
            )
        except ValidationError as exc:
            QMessageBox.warning(self, "Destino de exportación inválido", exc.format_for_user())
            return

        self.export_request = ExportRequest(
            output_path=validated,
            export_format=fmt,
            sample_rate=sample_rate,
            bit_depth=int(self._bits_combo.currentData()),
            mp3_bitrate_kbps=int(self._bitrate_combo.currentData()),
            channels=int(self._channels_combo.currentData()),
            include_environment=bool(self._scope_combo.currentData()),
            allow_overwrite=allow_overwrite,
        )
        self.accept()
