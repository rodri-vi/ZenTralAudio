"""Jerarquía explícita de excepciones de dominio para ZenTralAudio."""

from __future__ import annotations


class ZenTralAudioError(Exception):
    """Excepción base de toda la aplicación ZenTralAudio."""

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def format_for_user(self) -> str:
        """Devuelve un mensaje legible para mostrar en diálogos de interfaz."""
        if self.details:
            return f"{self.message}\n\nDetalle técnico: {self.details}"
        return self.message


class ConfigurationError(ZenTralAudioError):
    """Error al leer, validar o guardar la configuración de la aplicación."""


class PathPermissionError(ZenTralAudioError):
    """Error de acceso o permisos de escritura en rutas portables."""


class ValidationError(ZenTralAudioError):
    """Error de validación de archivos, parámetros o esquemas."""


class AudioIOError(ZenTralAudioError):
    """Error al leer, decodificar o escribir archivos de audio."""


class AudioFormatUnsupportedError(AudioIOError):
    """Formato o códec de audio no soportado o corrupto."""


class FFmpegNotFoundError(AudioIOError):
    """No se encontró el ejecutable ffmpeg requerido para la operación."""


class ProcessingError(ZenTralAudioError):
    """Error durante la ejecución de la cadena DSP o mezcla de audio."""


class ModelError(ZenTralAudioError):
    """Error general relacionado con modelos de inteligencia artificial."""


class ModelNotInstalledError(ModelError):
    """El modelo seleccionado no está instalado ni descargado localmente."""


class ModelLoadError(ModelError):
    """Error al inicializar o cargar los pesos de un modelo en CPU/GPU."""


class ModelInferenceError(ModelError):
    """Error durante el procesamiento de audio por un modelo IA."""


class ResourceDownloadError(ZenTralAudioError):
    """Error de red o almacenamiento al descargar modelos o recursos ambientales."""


class IntegrityCheckError(ResourceDownloadError):
    """Fallo de verificación de integridad SHA-256 o tamaño de archivo."""


class JobCancelledError(ZenTralAudioError):
    """Señala que una tarea en segundo plano fue cancelada de forma controlada."""


class ExportError(ZenTralAudioError):
    """Error al exportar el audio final o verificar el archivo generado."""
