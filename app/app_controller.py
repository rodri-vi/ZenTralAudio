"""Controlador central que conecta la interfaz gráfica con el motor y los módulos del sistema."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QInputDialog, QMessageBox

from app.dialogs.export_dialog import ExportDialog, ExportRequest
from app.dialogs.model_manager_dialog import ModelManagerDialog
from app.dialogs.resource_manager_dialog import ResourceManagerDialog
from app.dialogs.settings_dialog import SettingsDialog
from app.environment_tab import EnvironmentTab
from app.preview_player import PreviewPlayer
from app.restoration_tab import RestorationTab
from app.spectrogram_widget import compute_spectrogram_image
from app.widgets.dynamic_param_panel import UIParamDefinition
from core.configuration import ConfigManager
from core.exceptions import ZenTralAudioError
from core.job_manager import JobContext, JobManager
from core.paths import AppPaths
from core.validation import sanitize_filename, validate_audio_input_path

logger = logging.getLogger("zentralaudio.controller")

_KNOWN_MODELS_METADATA: list[dict[str, Any]] = [
    {
        "model_id": "mossformer2_se_48k",
        "display_name": "MossFormer2_SE_48K",
        "sample_rate": 48000,
        "license_summary": "Apache-2.0 (código y pesos)",
        "size_label": "~222 MB",
        "source_url": "https://huggingface.co/alibabasglab/MossFormer2_SE_48K",
        "status_code": "not_installed",
        "status_label": "No instalado",
        "download_supported": False,
        "note": "48 kHz mono. Alta calidad vocal; requiere integración de adaptador en Fase 5.",
    },
    {
        "model_id": "mossformergan_se_16k",
        "display_name": "MossFormerGAN_SE_16K",
        "sample_rate": 16000,
        "license_summary": "Apache-2.0 (código y pesos)",
        "size_label": "Por verificar",
        "source_url": "https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K",
        "status_code": "not_installed",
        "status_label": "No instalado",
        "download_supported": False,
        "note": "16 kHz (techo de 8 kHz). Integración y verificación en Fase 5.",
    },
    {
        "model_id": "frcrn_se_16k",
        "display_name": "FRCRN_SE_16K",
        "sample_rate": 16000,
        "license_summary": "Apache-2.0 (código y pesos)",
        "size_label": "~161 MB",
        "source_url": "https://huggingface.co/alibabasglab/FRCRN_SE_16K",
        "status_code": "not_installed",
        "status_label": "No instalado",
        "download_supported": False,
        "note": "16 kHz (techo de 8 kHz). Integración en Fase 5.",
    },
    {
        "model_id": "deepfilternet3",
        "display_name": "DeepFilterNet3",
        "sample_rate": 48000,
        "license_summary": "MIT / Apache-2.0 (código)",
        "size_label": "~8 MB",
        "source_url": "https://github.com/Rikorose/DeepFilterNet",
        "status_code": "not_installed",
        "status_label": "No instalado",
        "download_supported": False,
        "note": "48 kHz, rápido en CPU. Adaptador real en Fase 5.",
    },
    {
        "model_id": "voicefixer",
        "display_name": "VoiceFixer",
        "sample_rate": 44100,
        "license_summary": "MIT (código) / CC-BY-4.0 (pesos)",
        "size_label": "~596 MB",
        "source_url": "https://github.com/haoheliu/voicefixer",
        "status_code": "not_installed",
        "status_label": "No instalado",
        "download_supported": False,
        "note": "44.1 kHz. Resíntesis neuronal para grabaciones muy degradadas (atribución CC-BY-4.0).",
    },
    {
        "model_id": "gtcrn",
        "display_name": "GTCRN",
        "sample_rate": 16000,
        "license_summary": "MIT (código)",
        "size_label": "< 1 MB",
        "source_url": "https://github.com/Xiaobin-Rong/gtcrn",
        "status_code": "not_installed",
        "status_label": "No instalado",
        "download_supported": False,
        "note": "16 kHz ultraligero para CPU. Checkpoints pendientes de verificación en Fase 5.",
    },
]

_BUILTIN_PRESETS: list[dict[str, Any]] = [
    {"preset_id": "studio_voice", "name": "Studio Voice", "intensity": 75, "params": {"nr_amount": 0.45, "eq_hp_freq": 80, "comp_ratio": 3.0, "vocal_presence": 2.5, "deesser_amount": 0.35}},
    {"preset_id": "podcast_pro", "name": "Podcast Pro", "intensity": 80, "params": {"nr_amount": 0.55, "eq_hp_freq": 85, "comp_ratio": 3.8, "vocal_presence": 3.0, "deesser_amount": 0.40}},
    {"preset_id": "noise_removal", "name": "Noise Removal", "intensity": 85, "params": {"nr_amount": 0.70, "eq_hp_freq": 75, "comp_ratio": 2.0, "vocal_presence": 1.0, "deesser_amount": 0.20}},
    {"preset_id": "heavy_noise", "name": "Heavy Noise", "intensity": 95, "params": {"nr_amount": 0.90, "eq_hp_freq": 100, "comp_ratio": 2.5, "vocal_presence": 1.5, "deesser_amount": 0.30}},
    {"preset_id": "echo_removal", "name": "Echo Removal", "intensity": 80, "params": {"nr_amount": 0.40, "dereverb_amount": 0.65, "eq_hp_freq": 90, "comp_ratio": 2.2}},
    {"preset_id": "old_recording", "name": "Old Recording", "intensity": 85, "params": {"nr_amount": 0.75, "eq_hp_freq": 110, "eq_lp_freq": 12000, "vocal_clarity": 3.0, "comp_ratio": 2.8}},
    {"preset_id": "telephone_voice", "name": "Telephone Voice", "intensity": 75, "params": {"nr_amount": 0.50, "eq_hp_freq": 280, "eq_lp_freq": 3600, "comp_ratio": 4.2, "vocal_presence": 3.5}},
    {"preset_id": "natural_voice", "name": "Natural Voice", "intensity": 55, "params": {"nr_amount": 0.30, "eq_hp_freq": 70, "comp_ratio": 1.8, "vocal_presence": 1.0, "deesser_amount": 0.20}},
    {"preset_id": "maximum_recovery", "name": "Maximum Recovery", "intensity": 100, "params": {"nr_amount": 0.95, "dereverb_amount": 0.55, "eq_hp_freq": 95, "comp_ratio": 3.5, "vocal_clarity": 3.5}},
]


def build_dsp_parameter_schema() -> list[UIParamDefinition]:
    """Esquema base real de parámetros DSP que la cadena de Fase 4 implementará directamente."""
    return [
        UIParamDefinition("nr_enabled", "Activar reducción de ruido DSP", "Reducción de ruido", "bool", True, tooltip="Aplica reducción espectral complementaria."),
        UIParamDefinition("nr_amount", "Intensidad reducción ruido", "Reducción de ruido", "float", 0.50, 0.0, 1.0, 0.05, 2, "", "Nivel de atenuación del ruido estacionario."),
        UIParamDefinition("nr_smoothing", "Suavizado temporal", "Reducción de ruido", "float", 0.40, 0.0, 1.0, 0.05, 2, "", "Reduce el ruido musical entre tramas."),
        UIParamDefinition("eq_hp_freq", "Filtro paso alto", "Ecualización", "int", 80, 20, 400, 5, 0, "Hz", "Elimina retumbes y ruido subgrave."),
        UIParamDefinition("eq_lp_freq", "Filtro paso bajo", "Ecualización", "int", 18000, 3000, 22000, 250, 0, "Hz", "Recorta siseo ultrasónico o aspereza extrema."),
        UIParamDefinition("eq_mid_gain", "Realce/corte medios (2.5 kHz)", "Ecualización", "float", 0.0, -12.0, 12.0, 0.5, 1, "dB", "Ajusta inteligibilidad en la zona media-alta."),
        UIParamDefinition("comp_enabled", "Activar compresor", "Dinámica", "bool", True, tooltip="Controla picos y nivela el volumen vocal."),
        UIParamDefinition("comp_threshold", "Threshold", "Dinámica", "float", -18.0, -45.0, 0.0, 0.5, 1, "dB", "Umbral a partir del cual actúa la compresión."),
        UIParamDefinition("comp_ratio", "Ratio", "Dinámica", "float", 2.5, 1.0, 12.0, 0.2, 1, ":1", "Relación de compresión."),
        UIParamDefinition("comp_attack_ms", "Attack", "Dinámica", "float", 12.0, 1.0, 100.0, 1.0, 1, "ms", "Velocidad de respuesta ante transitorios."),
        UIParamDefinition("comp_release_ms", "Release", "Dinámica", "float", 140.0, 20.0, 600.0, 10.0, 0, "ms", "Velocidad de recuperación del compresor."),
        UIParamDefinition("vocal_presence", "Presencia vocal", "Tratamiento vocal", "float", 1.5, -6.0, 9.0, 0.5, 1, "dB", "Realce controlado alrededor de 3.5 kHz."),
        UIParamDefinition("vocal_clarity", "Claridad / aire", "Tratamiento vocal", "float", 1.0, -6.0, 9.0, 0.5, 1, "dB", "Brillo suave en frecuencias altas de voz."),
        UIParamDefinition("deesser_amount", "Control de sibilancia (De-Esser)", "Tratamiento vocal", "float", 0.25, 0.0, 1.0, 0.05, 2, "", "Atenúa eses excesivas (5-9 kHz)."),
        UIParamDefinition("dereverb_amount", "Reducción de reverberación DSP", "Tratamiento espacial", "float", 0.0, 0.0, 1.0, 0.05, 2, "", "Reduce cola reverberante leve mediante envolvente."),
        UIParamDefinition("stereo_width", "Anchura estéreo", "Tratamiento espacial", "float", 1.0, 0.0, 2.0, 0.05, 2, "", "0 = mono, 1 = original, >1 = más amplio."),
    ]


@dataclass
class _PlaybackState:
    """Estado interno seguro entre hilos para reproducción directa con sounddevice."""

    samples_original: np.ndarray | None = None
    samples_processed: np.ndarray | None = None
    samples_a: np.ndarray | None = None
    sample_rate: int = 48000
    position_frames: int = 0
    playing: bool = False
    loop: bool = False
    bypass_processed: bool = True
    ab_slot: str = "B"
    loudness_match: bool = True
    monitor_volume: float = 0.85
    selection_frames: tuple[int, int] | None = None
    rms_levels: list[float] = field(default_factory=lambda: [0.0, 0.0])
    peak_levels: list[float] = field(default_factory=lambda: [0.0, 0.0])


class AppController(QObject):
    """Coordina la UI, el reproductor de previsualización, el motor y las tareas en segundo plano."""

    _sig_audio_loaded = Signal(object, dict)
    _sig_spectrogram_ready = Signal(object, float, int)
    _sig_load_failed = Signal(str)
    _sig_render_ready = Signal(object)
    _sig_job_progress = Signal(bool, str, float)
    _sig_job_error = Signal(str)

    def __init__(
        self,
        *,
        paths: AppPaths,
        config_manager: ConfigManager,
        job_manager: JobManager,
        restoration_tab: RestorationTab,
        environment_tab: EnvironmentTab,
        preview_player: PreviewPlayer,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths
        self.config_manager = config_manager
        self.job_manager = job_manager
        self.restoration_tab = restoration_tab
        self.environment_tab = environment_tab
        self.preview_player = preview_player

        self._source_path: Path | None = None
        self._source_meta: dict[str, Any] = {}
        self._selected_model_id: str = config_manager.settings.last_selected_model
        self._selected_preset_id: str = "studio_voice"
        self._intensity: int = 75
        self._advanced_values: dict[str, Any] = {p.key: p.default for p in build_dsp_parameter_schema()}
        self._env_enabled: bool = False
        self._env_id: str = "estudio"
        self._ir_path: Path | None = None
        self._acoustic_params: dict[str, float] = {
            "room_size": 45.0,
            "reverb_mix": 28.0,
            "decay_s": 1.2,
            "predelay_ms": 14.0,
            "damping": 45.0,
            "direct_level": 100.0,
            "reflected_level": 40.0,
            "stereo_width": 100.0,
            "apparent_distance": 20.0,
            "pan": 0.0,
        }
        self._layers: list[dict[str, Any]] = []
        self._master_ambience_vol: float = 0.60

        self._playback = _PlaybackState(
            loudness_match=config_manager.settings.loudness_match_preview,
            monitor_volume=config_manager.settings.preview_monitor_volume,
        )
        self._pb_lock = threading.Lock()
        self._sd_stream: Any = None

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._on_playback_tick)

        self._connect_signals()
        self._initialize_ui_data()

    def _connect_signals(self) -> None:
        self._sig_audio_loaded.connect(self._apply_loaded_audio)
        self._sig_spectrogram_ready.connect(self.restoration_tab.spectrogram.set_spectrogram)
        self._sig_load_failed.connect(self._show_error)
        self._sig_render_ready.connect(self._apply_rendered_audio)
        self._sig_job_progress.connect(self.preview_player.set_processing_state)
        self._sig_job_error.connect(self._on_render_error)

        rt = self.restoration_tab
        rt.open_file_requested.connect(self.load_audio_file)
        rt.play_original_requested.connect(self._play_original_shortcut)
        rt.model_selected.connect(self._on_model_selected)
        rt.manage_models_requested.connect(self.open_model_manager)
        rt.preset_selected.connect(self._on_preset_selected)
        rt.save_custom_preset_requested.connect(self._save_custom_preset)
        rt.delete_custom_preset_requested.connect(self._delete_custom_preset)
        rt.intensity_changed.connect(self._on_intensity_changed)
        rt.advanced_param_changed.connect(self._on_advanced_param_changed)
        rt.advanced_reset_requested.connect(self._on_advanced_reset)
        rt.waveform.seek_requested.connect(self.seek)
        rt.spectrogram.seek_requested.connect(self.seek)
        rt.waveform.selection_changed.connect(self._on_selection_changed)
        rt.waveform.selection_cleared.connect(self._on_selection_cleared)

        et = self.environment_tab
        et.environment_enabled_toggled.connect(self._on_env_enabled_toggled)
        et.environment_selected.connect(self._on_env_selected)
        et.acoustic_param_changed.connect(self._on_acoustic_param_changed)
        et.load_local_ir_requested.connect(self._on_load_local_ir)
        et.clear_ir_requested.connect(self._on_clear_ir)
        et.add_local_layer_requested.connect(self._on_add_local_layer)
        et.replace_layer_file_requested.connect(self._on_replace_layer_file)
        et.remove_layer_requested.connect(self._on_remove_layer)
        et.layer_changed.connect(self._on_layer_changed)
        et.layer_solo_toggled.connect(self._on_layer_solo_toggled)
        et.master_ambience_volume_changed.connect(self._on_master_ambience_vol)
        et.manage_resources_requested.connect(self.open_resource_manager)

        pp = self.preview_player
        pp.play_clicked.connect(self.play)
        pp.pause_clicked.connect(self.pause)
        pp.stop_clicked.connect(self.stop)
        pp.seek_requested.connect(self.seek)
        pp.loop_toggled.connect(self._on_loop_toggled)
        pp.bypass_toggled.connect(self._on_bypass_toggled)
        pp.ab_slot_changed.connect(self._on_ab_changed)
        pp.snapshot_a_requested.connect(self._snapshot_current_to_a)
        pp.loudness_match_toggled.connect(self._on_loudness_match_toggled)
        pp.monitor_volume_changed.connect(self._on_monitor_volume_changed)
        pp.clear_selection_requested.connect(self._on_selection_cleared)
        pp.cancel_processing_requested.connect(lambda: self.job_manager.cancel_group("preview_render"))

    def _initialize_ui_data(self) -> None:
        self.preview_player.set_monitor_volume(self.config_manager.settings.preview_monitor_volume)
        self.restoration_tab.set_models_list(self._get_models_catalog(), self._selected_model_id)
        self.restoration_tab.set_presets_list(self._load_all_presets(), self._selected_preset_id)
        self._refresh_advanced_schema()
        self.environment_tab.set_layers(self._layers)

    def _get_models_catalog(self) -> list[dict[str, Any]]:
        try:
            from models.model_registry import get_global_registry  # type: ignore[import-not-found]

            return get_global_registry(self.paths).list_ui_summaries()
        except ImportError:
            return [dict(m) for m in _KNOWN_MODELS_METADATA]

    def _load_all_presets(self) -> list[dict[str, Any]]:
        presets = [dict(p, is_custom=False) for p in _BUILTIN_PRESETS]
        custom_dir = self.paths.custom_presets_dir
        if custom_dir.is_dir():
            for file in sorted(custom_dir.glob("*.json")):
                try:
                    data = json.loads(file.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "name" in data:
                        presets.append(
                            {
                                "preset_id": f"custom:{file.stem}",
                                "name": str(data["name"]),
                                "intensity": int(data.get("intensity", 75)),
                                "params": dict(data.get("params", {})),
                                "is_custom": True,
                                "file_path": str(file),
                            }
                        )
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Preset personalizado corrupto %s: %s", file, exc)
        return presets

    def _refresh_advanced_schema(self) -> None:
        schema = build_dsp_parameter_schema()
        try:
            from models.model_registry import get_global_registry  # type: ignore[import-not-found]

            schema = get_global_registry(self.paths).get_ui_schema_for_model(self._selected_model_id) + schema
        except ImportError:
            pass
        self.restoration_tab.set_advanced_schema(schema, self._advanced_values)

    def load_audio_file(self, file_path: str) -> None:
        """Valida y decodifica un audio en segundo plano sin congelar la interfaz."""
        try:
            validated = validate_audio_input_path(file_path)
        except ZenTralAudioError as exc:
            self._show_error(exc.format_for_user())
            return

        self.stop()
        self._sig_job_progress.emit(True, f"Cargando {validated.name}...", -1.0)

        def _load_worker(ctx: JobContext) -> tuple[np.ndarray, dict[str, Any]]:
            ctx.check_cancelled()
            try:
                from engine.audio_io import load_audio  # type: ignore[import-not-found]

                buf, meta_obj = load_audio(validated, paths=self.paths, config=self.config_manager.settings)
                samples = buf.samples
                meta = {
                    "filename": validated.name,
                    "sample_rate": meta_obj.sample_rate,
                    "channels": meta_obj.channels,
                    "bit_depth": meta_obj.bit_depth,
                    "duration_s": meta_obj.duration_seconds,
                    "path": validated,
                }
            except ImportError:
                try:
                    info = sf.info(str(validated))
                    data, sr = sf.read(str(validated), dtype="float32", always_2d=True)
                except Exception as exc:
                    raise ZenTralAudioError(
                        f"No se pudo decodificar '{validated.name}' en esta etapa.",
                        details="Los formatos comprimidos como MP3/M4A/AAC requieren FFmpeg (disponible en Fase 4 engine/audio_io).",
                    ) from exc
                samples = data.T.copy()
                meta = {
                    "filename": validated.name,
                    "sample_rate": int(sr),
                    "channels": int(samples.shape[0]),
                    "bit_depth": _subtype_to_bit_depth(info.subtype),
                    "duration_s": samples.shape[1] / float(sr),
                    "path": validated,
                }
            ctx.check_cancelled()
            return samples, meta

        self.job_manager.submit(
            f"load_audio:{validated.name}",
            _load_worker,
            group_key="audio_load",
            on_completed=lambda res: self._on_audio_loaded_worker(res[0], res[1]),
            on_failed=lambda exc: self._sig_load_failed.emit(
                exc.format_for_user() if isinstance(exc, ZenTralAudioError) else str(exc)
            ),
            on_cancelled=lambda: self._sig_job_progress.emit(False, "Carga cancelada", 0.0),
        )

    def _on_audio_loaded_worker(self, samples: np.ndarray, meta: dict[str, Any]) -> None:
        self._sig_audio_loaded.emit(samples, meta)
        sr = int(meta["sample_rate"])
        dur = float(meta["duration_s"])
        self.job_manager.submit(
            "compute_spectrogram",
            lambda ctx: compute_spectrogram_image(samples, sr),
            group_key="spectrogram",
            on_completed=lambda img: self._sig_spectrogram_ready.emit(img, dur, sr),
        )

    def _apply_loaded_audio(self, samples: np.ndarray, meta: dict[str, Any]) -> None:
        self._source_path = meta["path"]
        self._source_meta = meta
        self.config_manager.add_recent_file(self._source_path)

        with self._pb_lock:
            self._playback.samples_original = samples
            self._playback.samples_processed = None
            self._playback.samples_a = None
            self._playback.sample_rate = int(meta["sample_rate"])
            self._playback.position_frames = 0
            self._playback.selection_frames = None

        self.restoration_tab.set_metadata(
            filename=meta["filename"],
            duration_s=float(meta["duration_s"]),
            sample_rate=int(meta["sample_rate"]),
            bit_depth=meta["bit_depth"],
            channels=int(meta["channels"]),
        )
        self.restoration_tab.waveform.set_audio(samples, int(meta["sample_rate"]), meta["filename"])
        self.preview_player.set_audio_loaded(float(meta["duration_s"]), enabled=True)
        self.preview_player.set_selection_range(None, None)
        self.preview_player.set_processing_state(False, f"Listo ({meta['filename']})", 1.0)
        self._schedule_preview_render()

    def _schedule_preview_render(self) -> None:
        with self._pb_lock:
            original = self._playback.samples_original
            sr = self._playback.sample_rate
        if original is None:
            return

        try:
            from engine.preview_engine import render_preview_from_state  # type: ignore[import-not-found]
        except ImportError:
            self.preview_player.set_processing_state(False, "Audio cargado (motor DSP/IA pendiente de Fase 4)", 1.0)
            return

        state_snapshot = self.export_pipeline_dict()
        self._sig_job_progress.emit(True, "Renderizando previsualización...", -1.0)
        self.job_manager.submit(
            "preview_render",
            lambda ctx: render_preview_from_state(original, sr, state_snapshot, self.paths, ctx),
            group_key="preview_render",
            resource_key="pipeline_compute",
            on_completed=lambda out: self._sig_render_ready.emit(out),
            on_failed=lambda exc: self._sig_job_error.emit(str(exc)),
            on_cancelled=lambda: None,
        )

    def _apply_rendered_audio(self, rendered: np.ndarray) -> None:
        with self._pb_lock:
            self._playback.samples_processed = rendered
        self.preview_player.set_processing_state(False, "Previsualización actualizada", 1.0)

    def _on_render_error(self, message: str) -> None:
        self.preview_player.set_error_state(f"Error de procesamiento: {message}")

    def export_pipeline_dict(self, *, include_environment: bool | None = None) -> dict[str, Any]:
        """Representación serializable única de la cadena activa (compartida con exportación)."""
        env_flag = self._env_enabled if include_environment is None else include_environment
        return {
            "model_id": self._selected_model_id,
            "preset_id": self._selected_preset_id,
            "intensity": self._intensity,
            "advanced_params": dict(self._advanced_values),
            "environment": {
                "enabled": env_flag,
                "environment_id": self._env_id,
                "ir_path": str(self._ir_path) if self._ir_path else None,
                "acoustics": dict(self._acoustic_params),
                "master_volume": self._master_ambience_vol,
                "layers": [dict(layer) for layer in self._layers],
            },
            "output_safety": {
                "true_peak_ceiling_db": -1.0,
                "loudness_match_preview": self._playback.loudness_match,
            },
        }

    def play(self) -> None:
        with self._pb_lock:
            if self._playback.samples_original is None:
                return
            if self._playback.selection_frames is not None:
                s0, s1 = self._playback.selection_frames
                if not (s0 <= self._playback.position_frames < s1):
                    self._playback.position_frames = s0
            self._playback.playing = True
            sr = self._playback.sample_rate

        if self._sd_stream is None:
            try:
                import sounddevice as sd

                dev_idx = self.config_manager.settings.output_device_index
                self._sd_stream = sd.OutputStream(
                    samplerate=sr,
                    channels=2,
                    dtype="float32",
                    device=dev_idx,
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self._sd_stream.start()
            except Exception as exc:
                with self._pb_lock:
                    self._playback.playing = False
                self._show_error(f"No se pudo abrir el dispositivo de salida de audio:\n{exc}")
                return
        self._ui_timer.start()

    def pause(self) -> None:
        with self._pb_lock:
            self._playback.playing = False

    def stop(self) -> None:
        with self._pb_lock:
            self._playback.playing = False
            self._playback.position_frames = (
                self._playback.selection_frames[0] if self._playback.selection_frames else 0
            )
            pos_s = self._playback.position_frames / float(max(1, self._playback.sample_rate))
        self._ui_timer.stop()
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        self.preview_player.set_position(pos_s)
        self.restoration_tab.waveform.set_position(pos_s)
        self.restoration_tab.spectrogram.set_position(pos_s)
        self.preview_player.level_meter.reset()

    def seek(self, seconds: float) -> None:
        with self._pb_lock:
            if self._playback.samples_original is None:
                return
            total = self._playback.samples_original.shape[1]
            frame = int(max(0, min(total - 1, round(seconds * self._playback.sample_rate))))
            self._playback.position_frames = frame
        self.preview_player.set_position(seconds)
        self.restoration_tab.waveform.set_position(seconds)
        self.restoration_tab.spectrogram.set_position(seconds)

    def _audio_callback(self, outdata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
        with self._pb_lock:
            orig = self._playback.samples_original
            if not self._playback.playing or orig is None:
                outdata.fill(0.0)
                self._playback.rms_levels = [0.0, 0.0]
                self._playback.peak_levels = [0.0, 0.0]
                return

            if not self._playback.bypass_processed:
                source = orig
            elif self._playback.ab_slot == "A" and self._playback.samples_a is not None:
                source = self._playback.samples_a
            elif self._playback.samples_processed is not None:
                source = self._playback.samples_processed
            else:
                source = orig

            total_frames = source.shape[1]
            start_bound, end_bound = self._playback.selection_frames or (0, total_frames)
            end_bound = min(end_bound, total_frames)
            pos = self._playback.position_frames
            vol = self._playback.monitor_volume

            gain_comp = 1.0
            if self._playback.loudness_match and source is not orig:
                rms_orig = float(np.sqrt(np.mean(orig[:, start_bound:end_bound] ** 2)) + 1e-6)
                rms_src = float(np.sqrt(np.mean(source[:, start_bound:end_bound] ** 2)) + 1e-6)
                gain_comp = max(0.35, min(2.0, rms_orig / rms_src))

            written = 0
            out_buf = np.zeros((2, frames), dtype=np.float32)
            while written < frames:
                if pos >= end_bound:
                    if self._playback.loop and end_bound > start_bound:
                        pos = start_bound
                    else:
                        self._playback.playing = False
                        pos = start_bound
                        break
                avail = min(frames - written, end_bound - pos)
                chunk = source[:, pos : pos + avail]
                if chunk.shape[0] == 1:
                    out_buf[0, written : written + avail] = chunk[0]
                    out_buf[1, written : written + avail] = chunk[0]
                else:
                    out_buf[0, written : written + avail] = chunk[0]
                    out_buf[1, written : written + avail] = chunk[1]
                pos += avail
                written += avail

            out_buf *= vol * gain_comp
            np.clip(out_buf, -0.999, 0.999, out=out_buf)
            outdata[:] = out_buf.T
            self._playback.position_frames = pos
            self._playback.rms_levels = [
                float(np.sqrt(np.mean(out_buf[0] ** 2))),
                float(np.sqrt(np.mean(out_buf[1] ** 2))),
            ]
            self._playback.peak_levels = [
                float(np.max(np.abs(out_buf[0]))),
                float(np.max(np.abs(out_buf[1]))),
            ]

    def _on_playback_tick(self) -> None:
        with self._pb_lock:
            playing = self._playback.playing
            pos_s = self._playback.position_frames / float(max(1, self._playback.sample_rate))
            rms = list(self._playback.rms_levels)
            peaks = list(self._playback.peak_levels)
        self.preview_player.set_position(pos_s)
        self.restoration_tab.waveform.set_position(pos_s)
        self.restoration_tab.spectrogram.set_position(pos_s)
        self.preview_player.level_meter.set_levels(rms, peaks)
        if not playing:
            self._ui_timer.stop()
            self.preview_player.level_meter.reset()

    def _play_original_shortcut(self) -> None:
        self.preview_player.set_bypass_processed(False)
        self.play()

    def _on_model_selected(self, model_id: str) -> None:
        self._selected_model_id = model_id
        self.config_manager.settings.last_selected_model = model_id
        self.config_manager.save()
        self._refresh_advanced_schema()
        self._schedule_preview_render()

    def _on_preset_selected(self, preset_id: str) -> None:
        self._selected_preset_id = preset_id
        for p in self._load_all_presets():
            if p["preset_id"] == preset_id:
                self._intensity = int(p.get("intensity", 75))
                self.restoration_tab.set_intensity(self._intensity)
                self._advanced_values.update(p.get("params", {}))
                self.restoration_tab.advanced_panel.set_values(self._advanced_values)
                break
        self._schedule_preview_render()

    def _save_custom_preset(self) -> None:
        name, ok = QInputDialog.getText(self.restoration_tab, "Guardar preset personalizado", "Nombre del preset:")
        if not ok or not name.strip():
            return
        safe = sanitize_filename(name.strip(), fallback="custom_preset")
        self.paths.custom_presets_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name.strip(),
            "model_id": self._selected_model_id,
            "intensity": self._intensity,
            "params": dict(self._advanced_values),
        }
        target = self.paths.custom_presets_dir / f"{safe}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        new_id = f"custom:{safe}"
        self._selected_preset_id = new_id
        self.restoration_tab.set_presets_list(self._load_all_presets(), new_id)

    def _delete_custom_preset(self, preset_id: str) -> None:
        stem = preset_id.removeprefix("custom:")
        target = self.paths.custom_presets_dir / f"{stem}.json"
        if target.is_file():
            target.unlink()
        self._selected_preset_id = "studio_voice"
        self.restoration_tab.set_presets_list(self._load_all_presets(), self._selected_preset_id)

    def _on_intensity_changed(self, value: int) -> None:
        self._intensity = value
        self._schedule_preview_render()

    def _on_advanced_param_changed(self, key: str, value: Any) -> None:
        self._advanced_values[key] = value
        self._schedule_preview_render()

    def _on_advanced_reset(self) -> None:
        self._advanced_values = {p.key: p.default for p in build_dsp_parameter_schema()}
        self.restoration_tab.advanced_panel.set_values(self._advanced_values)
        self._schedule_preview_render()

    def _on_selection_changed(self, start_s: float, end_s: float) -> None:
        with self._pb_lock:
            sr = self._playback.sample_rate
            total = self._playback.samples_original.shape[1] if self._playback.samples_original is not None else 0
            f0 = max(0, min(total, int(round(start_s * sr))))
            f1 = max(f0 + 1, min(total, int(round(end_s * sr)))) if total > 0 else 0
            self._playback.selection_frames = (f0, f1) if f1 > f0 else None
        self.preview_player.set_selection_range(start_s, end_s)

    def _on_selection_cleared(self) -> None:
        with self._pb_lock:
            self._playback.selection_frames = None
        self.restoration_tab.waveform.set_selection(None, None)
        self.preview_player.set_selection_range(None, None)

    def _on_loop_toggled(self, checked: bool) -> None:
        with self._pb_lock:
            self._playback.loop = checked

    def _on_bypass_toggled(self, processed_on: bool) -> None:
        with self._pb_lock:
            self._playback.bypass_processed = processed_on

    def _on_ab_changed(self, slot: str) -> None:
        with self._pb_lock:
            self._playback.ab_slot = slot

    def _snapshot_current_to_a(self) -> None:
        with self._pb_lock:
            if self._playback.samples_processed is not None:
                self._playback.samples_a = self._playback.samples_processed.copy()
            elif self._playback.samples_original is not None:
                self._playback.samples_a = self._playback.samples_original.copy()
        self.preview_player.set_processing_state(False, "Instantánea guardada en A", 1.0)

    def _on_loudness_match_toggled(self, checked: bool) -> None:
        with self._pb_lock:
            self._playback.loudness_match = checked
        self.config_manager.settings.loudness_match_preview = checked
        self.config_manager.save()

    def _on_monitor_volume_changed(self, volume: float) -> None:
        with self._pb_lock:
            self._playback.monitor_volume = volume
        self.config_manager.settings.preview_monitor_volume = volume
        self.config_manager.save()

    def _on_env_enabled_toggled(self, checked: bool) -> None:
        self._env_enabled = checked
        self._schedule_preview_render()

    def _on_env_selected(self, env_id: str) -> None:
        self._env_id = env_id
        self._schedule_preview_render()

    def _on_acoustic_param_changed(self, key: str, value: Any) -> None:
        self._acoustic_params[key] = float(value)
        self._schedule_preview_render()

    def _on_load_local_ir(self, file_path: str) -> None:
        path = Path(file_path)
        self._ir_path = path
        self.environment_tab.set_ir_status(path.name)
        self._schedule_preview_render()

    def _on_clear_ir(self) -> None:
        self._ir_path = None
        self.environment_tab.set_ir_status(None)
        self._schedule_preview_render()

    def _on_add_local_layer(self, file_path: str) -> None:
        p = Path(file_path)
        layer = {
            "layer_id": f"local_{uuid.uuid4().hex[:8]}",
            "name": p.stem,
            "filename": p.name,
            "file_path": str(p),
            "status": "Local disponible",
            "license_info": f"Archivo local del usuario: {p}",
            "enabled": True,
            "volume": 0.35,
            "pan": 0.0,
            "solo": False,
        }
        self._layers.append(layer)
        self.environment_tab.set_layers(self._layers)
        self._schedule_preview_render()

    def _on_replace_layer_file(self, layer_id: str, file_path: str) -> None:
        p = Path(file_path)
        for layer in self._layers:
            if layer["layer_id"] == layer_id:
                layer["file_path"] = str(p)
                layer["filename"] = p.name
                layer["status"] = "Local disponible"
                layer["license_info"] = f"Archivo local del usuario: {p}"
                break
        self.environment_tab.set_layers(self._layers)
        self._schedule_preview_render()

    def _on_remove_layer(self, layer_id: str) -> None:
        self._layers = [layer for layer in self._layers if layer["layer_id"] != layer_id]
        self.environment_tab.set_layers(self._layers)
        self._schedule_preview_render()

    def _on_layer_changed(self, layer_id: str, state: dict[str, Any]) -> None:
        for layer in self._layers:
            if layer["layer_id"] == layer_id:
                layer.update(state)
                break
        self._schedule_preview_render()

    def _on_layer_solo_toggled(self, layer_id: str, checked: bool) -> None:
        for layer in self._layers:
            if layer["layer_id"] == layer_id:
                layer["solo"] = checked

    def _on_master_ambience_vol(self, volume: float) -> None:
        self._master_ambience_vol = volume
        self._schedule_preview_render()

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.paths, self.config_manager, parent=self.restoration_tab.window())
        dlg.exec()

    def open_model_manager(self) -> None:
        dlg = ModelManagerDialog(
            self.paths,
            self._get_models_catalog(),
            parent=self.restoration_tab.window(),
        )
        dlg.refresh_requested.connect(lambda: dlg.populate(self._get_models_catalog()))
        dlg.exec()
        self.restoration_tab.set_models_list(self._get_models_catalog(), self._selected_model_id)

    def open_resource_manager(self) -> None:
        resources: list[dict[str, Any]] = []
        try:
            from environments.ambience_manager import list_all_managed_resources  # type: ignore[import-not-found]

            resources = list_all_managed_resources(self.paths)
        except ImportError:
            pass
        dlg = ResourceManagerDialog(self.paths, resources, parent=self.restoration_tab.window())
        dlg.exec()

    def open_export_dialog(self) -> None:
        if self._source_path is None:
            QMessageBox.information(
                self.restoration_tab.window(),
                "Exportar audio",
                "Carga primero una grabación de audio antes de exportar.",
            )
            return

        default_dir = Path(self.config_manager.settings.last_export_dir or self.paths.exports_dir)
        ffmpeg_ok = self.paths.find_ffmpeg(self.config_manager.settings.ffmpeg_custom_path) is not None
        dlg = ExportDialog(
            default_dir=default_dir,
            source_path=self._source_path,
            source_sample_rate=int(self._source_meta.get("sample_rate", 48000)),
            source_channels=int(self._source_meta.get("channels", 1)),
            ffmpeg_available=ffmpeg_ok,
            environment_active=self._env_enabled,
            parent=self.restoration_tab.window(),
        )
        if dlg.exec() != ExportDialog.DialogCode.Accepted or dlg.export_request is None:
            return

        req: ExportRequest = dlg.export_request
        self.config_manager.settings.last_export_dir = str(req.output_path.parent)
        self.config_manager.save()

        try:
            from engine.export_engine import export_with_pipeline  # type: ignore[import-not-found]
        except ImportError:
            QMessageBox.information(
                self.restoration_tab.window(),
                "Exportación pendiente de Fase 4",
                "La validación de exportación está activa, pero el motor de escritura DSP/FFmpeg (engine/export_engine.py) se integrará en la Fase 4.",
            )
            return

        pipeline_dict = self.export_pipeline_dict(include_environment=req.include_environment)
        with self._pb_lock:
            original = self._playback.samples_original
            sr = self._playback.sample_rate
        if original is None:
            return

        self._sig_job_progress.emit(True, f"Exportando {req.output_path.name}...", 0.0)
        self.job_manager.submit(
            f"export:{req.output_path.name}",
            lambda ctx: export_with_pipeline(original, sr, pipeline_dict, req, self.paths, self.config_manager.settings, ctx),
            group_key="export_audio",
            resource_key="pipeline_compute",
            on_progress=lambda p, msg: self._sig_job_progress.emit(True, msg, p),
            on_completed=lambda path: self._sig_job_progress.emit(False, f"Exportado: {Path(path).name}", 1.0),
            on_failed=lambda exc: self._sig_job_error.emit(str(exc)),
        )

    def shutdown(self) -> None:
        self.stop()
        self.job_manager.cancel_all()

    def _show_error(self, message: str) -> None:
        self.preview_player.set_error_state("Error")
        QMessageBox.warning(self.restoration_tab.window(), "ZenTralAudio", message)


def _subtype_to_bit_depth(subtype: str | None) -> int | None:
    if not subtype:
        return None
    mapping = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}
    return mapping.get(subtype.upper())
