"""Funciones de validación estricta para archivos, parámetros y exportaciones."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.exceptions import ValidationError

SUPPORTED_INPUT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".aiff",
        ".aif",
    }
)

SUPPORTED_EXPORT_FORMATS: frozenset[str] = frozenset({"wav", "flac", "mp3"})
VALID_SAMPLE_RATES: tuple[int, ...] = (16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)
VALID_BIT_DEPTHS: tuple[int, ...] = (16, 24, 32)
VALID_MP3_BITRATES_KBPS: tuple[int, ...] = (128, 160, 192, 256, 320)


def validate_existing_file(path: str | Path, *, label: str = "archivo") -> Path:
    """Comprueba que la ruta exista y apunte a un archivo regular legible."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ValidationError(f"El {label} no existe: {resolved}")
    if not resolved.is_file():
        raise ValidationError(f"La ruta de {label} no es un archivo válido: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValidationError(f"El {label} está vacío (0 bytes): {resolved}")
    return resolved.resolve()


def validate_audio_input_path(path: str | Path) -> Path:
    """Valida un archivo de entrada de audio por existencia, tamaño y extensión."""
    resolved = validate_existing_file(path, label="archivo de audio")
    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValidationError(
            f"Extensión de audio no soportada: '{ext or '(sin extensión)'}'.",
            details=f"Extensiones admitidas: {', '.join(sorted(SUPPORTED_INPUT_EXTENSIONS))}",
        )
    return resolved


def validate_numeric_range(
    value: Any,
    *,
    name: str,
    min_value: float,
    max_value: float,
) -> float:
    """Valida que un valor numérico sea finito y esté dentro de [min_value, max_value]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"El parámetro '{name}' debe ser numérico, recibido {type(value).__name__}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError(f"El parámetro '{name}' contiene un valor no finito ({numeric}).")
    if numeric < min_value or numeric > max_value:
        raise ValidationError(
            f"El parámetro '{name}' ({numeric}) está fuera de rango [{min_value}, {max_value}]."
        )
    return numeric


def validate_export_target(
    output_path: str | Path,
    *,
    source_path: Path | None,
    export_format: str,
    sample_rate: int,
    allow_overwrite: bool = False,
) -> Path:
    """Valida destino, formato y protección estricta contra sobrescritura del original."""
    fmt = export_format.lower().strip()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        raise ValidationError(
            f"Formato de exportación no válido: '{export_format}'.",
            details=f"Formatos disponibles: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}",
        )

    if sample_rate not in VALID_SAMPLE_RATES:
        raise ValidationError(
            f"Frecuencia de muestreo de exportación no válida: {sample_rate} Hz.",
            details=f"Frecuencias permitidas: {VALID_SAMPLE_RATES}",
        )

    target = Path(output_path).expanduser().resolve()
    if not target.parent.exists():
        raise ValidationError(f"La carpeta de destino no existe: {target.parent}")

    if source_path is not None and target == source_path.resolve():
        raise ValidationError(
            "Por seguridad no se permite sobrescribir directamente el archivo original de entrada.",
            details="Elige otro nombre o carpeta para conservar siempre intacta la grabación fuente.",
        )

    if target.exists() and not allow_overwrite:
        raise ValidationError(
            f"El archivo de destino ya existe: {target.name}",
            details="Confirma la sobrescritura o cambia el nombre del archivo de salida.",
        )

    return target


def sanitize_filename(name: str, *, fallback: str = "zentralaudio_output") -> str:
    """Limpia caracteres inválidos en Windows para nombres de archivo o presets."""
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid_chars or ord(ch) < 32 else ch for ch in name).strip(" .")
    return cleaned or fallback
