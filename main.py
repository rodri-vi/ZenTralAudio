"""Punto de entrada de ZenTralAudio (modo escritorio y comprobación portable)."""

from __future__ import annotations

import argparse
import json
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


def run_environment_report(paths: AppPaths) -> int:
    """Diagnóstico legible del entorno real (útil dentro de la .app empaquetada)."""
    import platform

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "arch": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "paths": paths.to_diagnostic_dict(),
    }
    failures: list[str] = []
    try:
        import numpy
        import scipy
        import sounddevice
        import soundfile
        import torch

        report["libs"] = {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "soundfile": soundfile.__version__,
            "libsndfile": soundfile.__libsndfile_version__,
            "portaudio": sounddevice.get_portaudio_version()[1],
        }
        report["default_output"] = str(sounddevice.query_devices(kind="output")["name"])
    except Exception as exc:
        failures.append(f"bibliotecas: {exc}")

    try:
        from engine.dsp import kernels

        report["numba_kernels"] = kernels.warm_up()
    except Exception as exc:
        failures.append(f"núcleos DSP: {exc}")

    try:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(paths)
        report["models"] = {
            adapter.get_capabilities().model_id: adapter.get_status().value
            for adapter in registry.list_adapters()
        }
    except Exception as exc:
        failures.append(f"modelos: {exc}")

    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def _warm_up_dsp_kernels_in_background() -> None:
    """Compila los núcleos Numba mientras se construye la interfaz (≈1 s)."""
    import threading

    def _warm() -> None:
        try:
            from engine.dsp import kernels

            kernels.warm_up()
        except Exception:
            logger.debug("No se pudieron precompilar los núcleos DSP", exc_info=True)

    threading.Thread(target=_warm, name="ZenTralDSPWarmup", daemon=True).start()


class _FileOpenFilter:
    """Recibe los archivos que macOS entrega a la app (QFileOpenEvent)."""

    def __new__(cls, parent):  # noqa: ANN001
        from PySide6.QtCore import QEvent, QObject

        class _Filter(QObject):
            def __init__(self, owner) -> None:  # noqa: ANN001
                super().__init__(owner)
                self._pending: list[str] = []
                self._callback = None

            def attach(self, callback) -> None:  # noqa: ANN001
                self._callback = callback
                pending, self._pending = self._pending, []
                for path in pending[-1:]:
                    callback(path)

            def eventFilter(self, obj, event) -> bool:  # noqa: ANN001, N802
                if event.type() == QEvent.Type.FileOpen:
                    path = event.file()
                    if path:
                        if self._callback is None:
                            self._pending.append(path)
                        else:
                            self._callback(path)
                    return True
                return False

        return _Filter(parent)


def main(argv: Sequence[str] | None = None) -> int:
    """Arranca la interfaz gráfica o ejecuta el chequeo headless solicitado por CI."""
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Restauración de audio con IA y DSP")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Prepara rutas, logging y configuración y termina sin abrir la interfaz.",
    )
    parser.add_argument(
        "--check-offline",
        action="store_true",
        help="Comprueba que pesos y runtimes IA incluidos en esta edición son portables.",
    )
    parser.add_argument(
        "--smoke-test-model",
        choices=(
            "deepfilternet3",
            "mossformer2_se_48k",
            "mossformergan_se_16k",
            "frcrn_se_16k",
            "voicefixer",
        ),
        help="Ejecuta una inferencia corta real con un modelo IA instalado.",
    )
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="Muestra rutas, FFmpeg, núcleos DSP y estado de los modelos IA, y termina.",
    )
    parser.add_argument(
        "--playback-selftest",
        metavar="AUDIO",
        help="Abre la interfaz, reproduce AUDIO y mide los callbacks de audio en tiempo real.",
    )
    parser.add_argument("--selftest-seconds", type=float, default=15.0)
    parser.add_argument(
        "--selftest-volume",
        type=float,
        default=0.0,
        help="Volumen de escucha durante el autodiagnóstico (0 = silencio, misma temporización).",
    )
    # macOS añade -psn_* al abrir apps antiguas desde Finder; se ignoran argumentos ajenos.
    args, _unknown = parser.parse_known_args(argv)

    try:
        runtime_paths, config_manager = prepare_runtime()
        if args.check:
            logger.info("Modo --check completado correctamente.")
            return 0
        if args.check_environment:
            return run_environment_report(runtime_paths)
        if args.check_offline:
            from models.base_adapter import ModelStatus
            from models.isolated_runners.subprocess_runner import find_bundled_worker
            from models.model_registry import ModelRegistry

            registry = ModelRegistry(runtime_paths)
            manifest_path = runtime_paths.app_root / "MODELS_OFFLINE_MANIFEST.json"
            failures: list[str] = []
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_models = tuple(
                    str(item["model_id"]) for item in manifest.get("models", [])
                )
            except Exception as exc:
                expected_models = ()
                failures.append(f"manifiesto offline no válido: {exc}")

            if not expected_models:
                failures.append("el manifiesto no contiene modelos incluidos")

            worker_models = {
                "mossformer2_se_48k",
                "mossformergan_se_16k",
                "frcrn_se_16k",
                "voicefixer",
            }
            for model_id in expected_models:
                adapter = registry.get_adapter(model_id)
                if adapter is None:
                    failures.append(f"{model_id}: adaptador no registrado")
                    continue
                status = adapter.get_status()
                if status != ModelStatus.AVAILABLE:
                    failures.append(f"{model_id}: {status.value} — {adapter.get_status_detail()}")
                if model_id in worker_models and find_bundled_worker(registry.manager.model_dir(model_id)) is None:
                    failures.append(f"{model_id}: falta el worker portable de IA")

            if failures:
                for failure in failures:
                    logger.error("Offline AI: %s", failure)
                return 1
            logger.info(
                "Offline AI verificado: %d modelo(s) con pesos y runtimes locales.",
                len(expected_models),
            )
            return 0

        if args.smoke_test_model:
            import numpy as np

            from engine.audio_types import AudioBuffer
            from models.model_registry import ModelRegistry

            registry = ModelRegistry(runtime_paths)
            adapter = registry.get_adapter(args.smoke_test_model)
            if adapter is None:
                raise RuntimeError(f"No existe el adaptador {args.smoke_test_model}.")
            if adapter.get_status().value != "available":
                raise RuntimeError(
                    f"{args.smoke_test_model} no está disponible: {adapter.get_status_detail()}"
                )

            sample_rate = adapter.get_capabilities().required_sample_rate
            seconds = 2
            rng = np.random.default_rng(20261124)
            time_axis = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
            speech_like = (
                0.08 * np.sin(2.0 * np.pi * 180.0 * time_axis)
                + 0.04 * np.sin(2.0 * np.pi * 360.0 * time_axis)
                + 0.01 * rng.standard_normal(time_axis.shape[0], dtype=np.float32)
            ).astype(np.float32)
            source = AudioBuffer(samples=speech_like[np.newaxis, :], sample_rate=sample_rate)
            try:
                result = adapter.process(
                    source,
                    {
                        "device": "cpu",
                        "model_atten_lim_db": 30.0,
                        "model_post_filter": False,
                        "model_compensate_delay": True,
                    },
                )
                if result.sample_rate != sample_rate:
                    raise RuntimeError(
                        f"Frecuencia de salida inesperada: {result.sample_rate} Hz."
                    )
                if result.samples.shape != source.samples.shape:
                    raise RuntimeError(
                        f"Forma de salida inesperada: {result.samples.shape}."
                    )
                if not np.isfinite(result.samples).all():
                    raise RuntimeError("El modelo generó muestras NaN o infinitas.")
                logger.info(
                    "Inferencia real superada: %s, %d muestras a %d Hz.",
                    args.smoke_test_model,
                    result.num_samples,
                    result.sample_rate,
                )
            finally:
                adapter.unload_model()
            return 0

        # Importar la UI después de atender --check mantiene el chequeo portable
        # ligero y evita crear una ventana en procesos de diagnóstico.
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication([sys.argv[0] if sys.argv else APP_NAME])
        app.setApplicationName(APP_NAME)
        app.setOrganizationName(APP_ORGANIZATION)
        file_open_filter = _FileOpenFilter(app)
        app.installEventFilter(file_open_filter)
        _warm_up_dsp_kernels_in_background()

        job_manager = JobManager(max_workers=config_manager.settings.max_worker_threads)
        try:
            window = MainWindow(
                paths=runtime_paths,
                config_manager=config_manager,
                job_manager=job_manager,
            )
            window.showMaximized()
            # Archivos abiertos desde Finder ("Abrir con" / arrastrar al Dock).
            file_open_filter.attach(window.controller.load_audio_file)
            logger.info("Interfaz de ZenTralAudio iniciada en ventana maximizada.")
            if args.playback_selftest:
                from app.diagnostics import format_report, run_playback_selftest

                report = run_playback_selftest(
                    window,
                    args.playback_selftest,
                    seconds=args.selftest_seconds,
                    volume=args.selftest_volume,
                )
                logger.info("Autodiagnóstico de reproducción: %s", report)
                print(format_report(report), flush=True)
                window.close()
                return 0 if report["ok"] else 1
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
