"""Configuración centralizada de logging rotativo en modo portable."""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

LOG_FILENAME = "zentralaudio.log"
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MiB por archivo
BACKUP_COUNT = 3


def setup_logging(logs_dir: Path, *, debug: bool = False) -> Path:
    """Inicializa el logger raíz con salida rotativa a archivo y consola.

    Devuelve la ruta del archivo de log activo.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / LOG_FILENAME

    root_logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(level)

    # Evitar duplicar handlers si se llama más de una vez (p. ej. en tests)
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
        existing.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # En ejecutables GUI sin consola (--noconsole), sys.stderr puede ser None
    if sys.stderr is not None:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    install_exception_hooks()
    logging.getLogger("zentralaudio.boot").info(
        "Logging inicializado (nivel=%s, archivo=%s)",
        logging.getLevelName(level),
        log_path,
    )
    return log_path


def install_exception_hooks() -> None:
    """Registra hooks globales para capturar excepciones no controladas en logs."""

    def _handle_main_exception(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("zentralaudio.unhandled").critical(
            "Excepción no controlada en el hilo principal",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        logging.getLogger("zentralaudio.unhandled").error(
            "Excepción no controlada en hilo '%s'",
            args.thread.name if args.thread else "desconocido",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _handle_main_exception
    threading.excepthook = _handle_thread_exception
