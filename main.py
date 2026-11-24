"""Punto de entrada de ZenTralAudio (modo escritorio y comprobación portable)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from core.configuration import ConfigManager
from core.job_manager import JobManager
from core.logging_config import setup_logging
from core.paths import APP_NAME, APP_ORGANIZATION, AppPaths

logger = logging.getLogger("zentralaudio.boot")


def prepare_runtime(paths: AppPaths | None = None) -> tuple[AppPaths, ConfigManager]:
    """Prepara carpetas, logging y configuración para desarrollo o ejecución portable."""
    runtime_paths = paths or AppPaths.detect()
    runtime_paths.ensure_writable_directories()

    # Inicializar el archivo de log antes de cargar ajustes para registrar errores
    # de configuración. Si debug_logging está activo, se eleva después el nivel.
    setup_logging(runtime_paths.logs_dir)
    config_manager = ConfigManager(runtime_paths.config_dir)
    settings = config_manager.load()
    if settings.debug_logging:
        setup_logging(runtime_paths.logs_dir, debug=True)

    return runtime_paths, config_manager


def run_startup_check(paths: AppPaths | None = None) -> int:
    """Comprueba el arranque portable y crea los archivos iniciales requeridos."""
    runtime_paths, config_manager = prepare_runtime(paths)
    logger.info(
        "Comprobación de arranque superada (raíz=%s, configuración=%s)",
        runtime_paths.app_root,
        config_manager.config_path,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Arranca la interfaz gráfica o ejecuta el chequeo headless solicitado por CI."""
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Restauración de audio con IA y DSP")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Prepara rutas, logging y configuración y termina sin abrir la interfaz.",
    )
    args = parser.parse_args(argv)

    try:
        runtime_paths, config_manager = prepare_runtime()
        if args.check:
            logger.info("Modo --check completado correctamente.")
            return 0

        # Importar la UI después de atender --check mantiene el chequeo portable
        # ligero y evita crear una ventana en procesos de diagnóstico.
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication([sys.argv[0] if sys.argv else APP_NAME])
        app.setApplicationName(APP_NAME)
        app.setOrganizationName(APP_ORGANIZATION)

        job_manager = JobManager(max_workers=config_manager.settings.max_worker_threads)
        try:
            window = MainWindow(
                paths=runtime_paths,
                config_manager=config_manager,
                job_manager=job_manager,
            )
            window.show()
            logger.info("Interfaz de ZenTralAudio iniciada.")
            return app.exec()
        finally:
            # MainWindow detiene la reproducción y cancela sus tareas al cerrarse.
            # Aquí se libera además el pool de hilos antes de salir del proceso.
            job_manager.shutdown(wait=False)

    except Exception as exc:
        logger.exception("No se pudo iniciar ZenTralAudio: %s", exc)
        app_instance = None
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app_instance = QApplication.instance()
            if app_instance is not None:
                QMessageBox.critical(None, f"{APP_NAME} — Error de inicio", str(exc))
        except Exception:
            app_instance = None

        if app_instance is None and sys.__stderr__ is not None:
            print(f"No se pudo iniciar {APP_NAME}: {exc}", file=sys.__stderr__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
