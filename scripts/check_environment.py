"""Comprueba entorno Python 3.11, imports de todos los módulos, FFmpeg y rutas portables."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths  # noqa: E402

REQUIRED_MODULES = [
    "core.paths",
    "core.exceptions",
    "core.configuration",
    "core.logging_config",
    "core.validation",
    "core.job_manager",
    "engine.audio_types",
    "engine.audio_io",
    "engine.audio_resampler",
    "engine.audio_mixer",
    "engine.dsp.spectral_denoise",
    "engine.dsp.equalizer",
    "engine.dsp.dynamics",
    "engine.dsp.vocal_treatment",
    "engine.dsp.spatial",
    "engine.dsp.safety_limiter",
    "engine.processing_graph",
    "engine.processing_pipeline",
    "engine.preview_engine",
    "engine.export_engine",
    "engine.audio_engine",
    "models.base_adapter",
    "models.model_manager",
    "models.deepfilternet_adapter",
    "models.mossformer2_adapter",
    "models.mossformergan_adapter",
    "models.frcrn_adapter",
    "models.isolated_runners.voicefixer_adapter",
    "models.gtcrn_adapter",
    "models.model_registry",
    "models.isolated_runners.runner_protocol",
    "models.isolated_runners.subprocess_runner",
    "environments.environment_catalog",
    "environments.ambience_manager",
    "environments.impulse_response_manager",
    "environments.acoustic_simulator",
    "environments.environment_engine",
    "environments.environment_presets",
    "presets.preset_manager",
    "app.theme",
    "app.widgets.level_meter_widget",
    "app.widgets.dynamic_param_panel",
    "app.widgets.ambience_layer_widget",
    "app.waveform_widget",
    "app.spectrogram_widget",
    "app.preview_player",
    "app.restoration_tab",
    "app.environment_tab",
    "app.dialogs.settings_dialog",
    "app.dialogs.model_manager_dialog",
    "app.dialogs.resource_manager_dialog",
    "app.dialogs.export_dialog",
    "app.app_controller",
    "app.main_window",
]


def main() -> int:
    print(f"Python: {sys.version}")
    if sys.version_info[:2] != (3, 11):
        print("AVISO: Se recomienda Python 3.11.x para máxima compatibilidad con wheels de audio.")

    paths = AppPaths.detect()
    paths.ensure_writable_directories()
    print(f"Raíz portable: {paths.app_root}")
    print(f"FFmpeg: {paths.find_ffmpeg() or 'NO DETECTADO (opcional para WAV/FLAC, requerido para MP3/M4A)'}")

    failures: list[str] = []
    for mod_name in REQUIRED_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            failures.append(f"{mod_name}: {exc}")

    if failures:
        print("\nERRORES DE IMPORTACIÓN:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"Todos los {len(REQUIRED_MODULES)} módulos de ZenTralAudio se importaron correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
