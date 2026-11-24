"""Controlador central integrado: UI <-> Motor <-> Modelos IA <-> Ambientes <-> Exportación."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
from core.validation import validate_audio_input_path
from engine.audio_io import load_audio
from engine.export_engine import export_with_pipeline
from engine.preview_engine import get_shared_pipeline, render_preview_from_state
from environments.ambience_manager import AmbienceManager, list_all_managed_resources
from environments.environment_presets import build_environment_ui_state
from models.model_registry import get_global_registry
from presets.preset_manager import PresetManager

logger = logging.getLogger("zentralaudio.controller")


def build_dsp_parameter_schema() -> list[UIParamDefinition]:
    """Esquema real de parámetros DSP conectados directamente a ProcessingPipeline."""
    return [
        UIParamDefinition("nr_enabled", "Activar reducción de ruido DSP", "Reducción de ruido", "bool", True, tooltip="Aplica reducción espectral complementaria."),
        UIParamDefinition("nr_amount", "Intensidad reducción ruido", "Reducción de ruido", "float", 0.50, 0.0, 1.0, 0.05, 2, "", "Nivel de atenuación del ruido estacionario."),
        UIParamDefinition("nr_smoothing", "Suavizado temporal", "Reducción de ruido", "float", 0.40, 0.0, 1.0, 0.05, 2, "", "Reduce el ruido musical entre tramas."),
        UIParamDefinition("eq_hp_freq", "Filtro paso alto", "Ecualización", "int", 80, 20, 400, 5, 0, "Hz", "Elimina retumbes y ruido subgrave."),
        UIParamDefinition("eq_lp_freq", "Filtro paso bajo", "Ecualización", "int", 18000, 3000, 22000, 250, 0, "Hz", "Recorta siseo ultrasónico o aspereza extrema."),
        UIParamDefinition("eq_mid_gain", "Realce/corte medios (2.5 kHz)", "Ecualización", "float", 0.0, -12.0, 12.0, 0.5, 1, "dB", "Ajusta inteligibilidad en la zona media-alta."),
        UIParamDefinition("gate_enabled", "Activar gate / expansor suave", "Dinámica", "bool", False, tooltip="Atenúa ruido residual en pausas entre frases."),
        UIParamDefinition("gate_threshold", "Umbral gate", "Dinámica", "float", -48.0, -70.0, -20.0, 1.0, 0, "dB", "Nivel bajo el cual actúa el expansor."),
        UIParamDefinition("comp_enabled", "Activar compresor", "Dinámica", "bool", True, tooltip="Controla picos y nivela el volumen vocal."),
        UIParamDefinition("comp_threshold", "Threshold", "Dinámica", "float", -18.0, -45.0, 0.0, 0.5, 1, "dB", "Umbral a partir del cual actúa la compresión."),
        UIParamDefinition("comp_ratio", "Ratio", "Dinámica", "float", 2.5, 1.0, 12.0, 0.2, 1, ":1", "Relación de compresión."),
        UIParamDefinition("comp_attack_ms", "Attack", "Dinámica", "float", 12.0, 1.0, 100.0, 1.0, 1, "ms", "Velocidad de respuesta ante transitorios."),
        UIParamDefinition("comp_release_ms", "Release", "Dinámica", "float", 140.0, 20.0, 600.0, 10.0, 0, "ms", "Velocidad de recuperación del compresor."),
        UIParamDefinition("vocal_presence", "Presencia vocal", "Tratamiento vocal", "float", 1.5, -6.0, 9.0, 0.5, 1, "dB", "Realce controlado alrededor de 3.5 kHz."),
        UIParamDefinition("vocal_clarity", "Claridad / aire", "Tratamiento vocal", "float", 1.0, -6.0, 9.0, 0.5, 1, "dB", "Brillo suave en frecuencias altas de voz."),
        UIParamDefinition("deesser_amount", "Control de sibilancia (De-Esser)", "Tratamiento vocal", "float", 0.25, 0.0, 1.0, 0.05, 2, "", "Atenúa eses excesivas (5-9 kHz)."),
        UIParamDefinition("vocal_saturation", "Saturación armónica suave", "Tratamiento vocal", "float", 0.0, 0.0, 1.0, 0.05, 2, "", "Añade calidez analógica controlada."),
        UIParamDefinition("dereverb_amount", "Reducción de reverberación DSP", "Tratamiento espacial", "float", 0.0, 0.0, 1.0, 0.05, 2, "", "Reduce cola reverberante leve mediante envolvente."),
        UIParamDefinition("stereo_width", "Anchura estéreo", "Tratamiento espacial", "float", 1.0, 0.0, 2.0, 0.05, 2, "", "0 = mono, 1 = original, >1 = más amplio."),
    ]


@dataclass
class _PlaybackState:
    samples_original: np.ndarray | None = None
    samples_processed: np.ndarray | None = None
    samples_a: np.ndarray | None = None
    sample_rate: int = 48000
    position_frames: int = 0
    playing: bool = False
    loop: bool = False
    bypass_processed: bool = True
    ab_slot: str = "B"
    last_source_tag: str = "B"
    pending_crossfade: bool = False
    loudness_match: bool = True
    monitor_volume: float = 0.85
    selection_frames: tuple[int, int] | None = None
    rms_levels: list[float] = field(default_factory=lambda: [0.0, 0.0])
    peak_levels: list[float] = field(default_factory=lambda: [0.0, 0.0])


class AppController(QObject):
    """Coordina la UI, el reproductor de previsualización, el motor, modelos y ambientes."""

    _sig_audio_loaded = Signal(object, dict)
    _sig_spectrogram_ready = Signal(object, float, int)
    _sig_load_failed = Signal(str)
    _sig_render_ready = Signal(object)
    _sig_job_progress = Signal(bool, str, float)
    _sig_job_error = Signal(str)
    _sig_export_done = Signal(str)
    _sig_model_dl_progress = Signal(bool, str, float)
    _sig_model_dl_done = Signal(str)
    _sig_resource_dl_progress = Signal(bool, str, float)
    _sig_resource_dl_done = Signal(str)

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

        self.preset_manager = PresetManager(paths)
        self.model_registry = get_global_registry(paths)
        self.ambience_manager = AmbienceManager(paths)

        self._source_path: Path | None = None
        self._source_meta: dict[str, Any] = {}
        self._selected_model_id: str = config_manager.settings.last_selected_model
        self._selected_preset_id: str = "studio_voice"
        self._intensity: int = 75
        self._advanced_values: dict[str, Any] = {p.key: p.default for p in build_dsp_parameter_schema()}
        self._env_enabled: bool = False
        self._env_id: str = "estudio"
        self._ir_path: Path | None = None
        self._acoustic_params: dict[str, float] = {}
        self._layers: list[dict[str, Any]] = []
        self._master_ambience_vol: float = 0.60

        self._active_model_dlg: ModelManagerDialog | None = None
        self._active_resource_dlg: ResourceManagerDialog | None = None

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
        self._sig_export_done.connect(self._on_export_done)
        self._sig_model_dl_progress.connect(self._on_model_dl_progress)
        self._sig_model_dl_done.connect(self._on_model_dl_done)
        self._sig_resource_dl_progress.connect(self._on_resource_dl_progress)
        self._sig_resource_dl_done.connect(self._on_resource_dl_done)

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
        self.restoration_tab.set_models_list(self.model_registry.list_ui_summaries(), self._selected_model_id)
        self.restoration_tab.set_presets_list(self.preset_manager.list_all(), self._selected_preset_id)
        self._apply_preset_resolution(self._selected_preset_id, trigger_render=False)
        self._refresh_advanced_schema()
        self._load_environment_preset(self._env_id, trigger_render=False)

    def _refresh_advanced_schema(self) -> None:
        model_schema = self.model_registry.get_ui_schema_for_model(self._selected_model_id)
        schema = model_schema + build_dsp_parameter_schema()
        for p in schema:
            self._advanced_values.setdefault(p.key, p.default)
        self.restoration_tab.set_advanced_schema(schema, self._advanced_values)

    def _apply_preset_resolution(self, preset_id: str, *, trigger_render: bool = True) -> None:
        try:
            res = self.preset_manager.resolve_for_model(preset_id, self._selected_model_id)
        except ZenTralAudioError:
            self.restoration_tab.set_preset_notes([])
            return
        self._intensity = res.intensity
        self.restoration_tab.set_intensity(self._intensity)
        self._advanced_values.update(res.params)
        self.restoration_tab.advanced_panel.set_values(self._advanced_values)
        self.restoration_tab.set_preset_notes(res.notes)
        if trigger_render:
            self._schedule_preview_render()

    def _load_environment_preset(self, env_id: str, *, trigger_render: bool = True) -> None:
        acoustics, layers, ir_name = build_environment_ui_state(env_id, self.paths)
        if acoustics:
            self._acoustic_params = acoustics
            self.environment_tab.set_acoustic_values(acoustics)
        self._layers = layers
        self.environment_tab.set_layers(self._layers)
        if self._ir_path is None:
            self.environment_tab.set_ir_status(ir_name)
        if trigger_render:
            self._schedule_preview_render()

    def load_audio_file(self, file_path: str) -> None:
        """Valida y decodifica un archivo de entrada en segundo plano."""
        try:
            validated = validate_audio_input_path(file_path)
        except ZenTralAudioError as exc:
            self._show_error(exc.format_for_user())
            return

        self.stop()
        self._sig_job_progress.emit(True, f"Cargando {validated.name}...", -1.0)

        def _load_worker(ctx: JobContext) -> tuple[np.ndarray, dict[str, Any]]:
            ctx.check_cancelled()
            buf, meta_obj = load_audio(validated, paths=self.paths, config=self.config_manager.settings)
            ctx.check_cancelled()
            adapter = self.model_registry.get_adapter(self._selected_model_id)
            if adapter is not None:
                val = adapter.validate_input(buf)
                for w in val.warnings:
                    logger.info("Aviso de entrada (%s): %s", self._selected_model_id, w)
            meta = {
                "filename": validated.name,
                "sample_rate": meta_obj.sample_rate,
                "channels": meta_obj.channels,
                "bit_depth": meta_obj.bit_depth,
                "duration_s": meta_obj.duration_seconds,
                "path": validated,
            }
            return buf.samples, meta

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

        state_snapshot = self.export_pipeline_dict()
        self._sig_job_progress.emit(True, "Renderizando previsualización...", 0.05)
        self.job_manager.submit(
            "preview_render",
            lambda ctx: render_preview_from_state(original, sr, state_snapshot, self.paths, ctx),
            group_key="preview_render",
            resource_key="pipeline_compute",
            on_progress=lambda p, msg: self._sig_job_progress.emit(True, msg, p),
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
        """Representación serializable única compartida entre previsualización y exportación."""
        env_flag = self._env_enabled if include_environment is None else include_environment
        adv = dict(self._advanced_values)
        adv["device"] = self.config_manager.settings.preferred_device
        return {
            "model_id": self._selected_model_id,
            "preset_id": self._selected_preset_id,
            "intensity": self._intensity,
            "advanced_params": adv,
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

    def _select_playback_buffer(self) -> tuple[np.ndarray, str, float]:
        pb = self._playback
        orig = pb.samples_original
        assert orig is not None
        if not pb.bypass_processed:
            return orig, "orig", 1.0
        if pb.ab_slot == "A" and pb.samples_a is not None:
            src, tag = pb.samples_a, "A"
        elif pb.samples_processed is not None:
            src, tag = pb.samples_processed, "B"
        else:
            return orig, "orig", 1.0

        gain_comp = 1.0
        if pb.loudness_match:
            s0, s1 = pb.selection_frames or (0, min(orig.shape[1], src.shape[1]))
            s1 = max(s0 + 1, min(s1, orig.shape[1], src.shape[1]))
            rms_orig = float(np.sqrt(np.mean(orig[:, s0:s1] ** 2)) + 1e-6)
            rms_src = float(np.sqrt(np.mean(src[:, s0:s1] ** 2)) + 1e-6)
            gain_comp = max(0.35, min(2.0, rms_orig / rms_src))
        return src, tag, gain_comp

    @staticmethod
    def _extract_stereo_slice(buf: np.ndarray, start: int, count: int) -> np.ndarray:
        slice_arr = buf[:, start : start + count]
        if slice_arr.shape[0] == 1:
            return np.repeat(slice_arr, 2, axis=0)
        return slice_arr[:2]

    def _audio_callback(self, outdata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
        with self._pb_lock:
            orig = self._playback.samples_original
            if not self._playback.playing or orig is None:
                outdata.fill(0.0)
                self._playback.rms_levels = [0.0, 0.0]
                self._playback.peak_levels = [0.0, 0.0]
                return

            source, tag, gain_comp = self._select_playback_buffer()
            prev_tag = self._playback.last_source_tag
            do_xfade = self._playback.pending_crossfade and (tag != prev_tag)
            self._playback.pending_crossfade = False
            self._playback.last_source_tag = tag

            total_frames = source.shape[1]
            start_bound, end_bound = self._playback.selection_frames or (0, total_frames)
            end_bound = min(end_bound, total_frames)
            pos = self._playback.position_frames
            vol = self._playback.monitor_volume

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
                chunk = self._extract_stereo_slice(source, pos, avail) * gain_comp
                if do_xfade and written == 0 and avail >= 16:
                    cf = min(avail, 256)
                    if prev_tag == "orig":
                        prev_buf = orig
                    elif p
