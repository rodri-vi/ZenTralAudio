"""Pruebas unitarias de los 14 ambientes, convolución IR, simulación algorítmica y recursos."""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np
import pytest
import soundfile as sf

from core.exceptions import ResourceDownloadError
from core.paths import AppPaths
from engine.audio_types import AudioBuffer
from engine.processing_graph import PipelineConfig
from environments.acoustic_simulator import (
    build_algorithmic_room_ir,
    simulate_acoustic_space,
)
from environments.ambience_manager import AmbienceManager, ResourceStatus
from environments.environment_catalog import EnvironmentCatalog
from environments.environment_engine import apply_environment_stage
from environments.environment_presets import build_environment_ui_state
from tests.conftest import make_sine_buffer


@pytest.mark.unit
@pytest.mark.unit
def test_ambience_layer_solo_sync_does_not_rebuild_or_emit_duplicate_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Solo se sincroniza en sitio y no dispara un segundo evento genérico."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.environment_tab import EnvironmentTab

    app = QApplication.instance() or QApplication([])
    tab = EnvironmentTab()
    tab.set_layers(
        [
            {
                "layer_id": "layer_a",
                "name": "Capa A",
                "enabled": True,
                "volume": 0.35,
                "pan": 0.0,
                "solo": False,
            },
            {
                "layer_id": "layer_b",
                "name": "Capa B",
                "enabled": True,
                "volume": 0.35,
                "pan": 0.0,
                "solo": False,
            },
        ]
    )
    first_widget = tab._layer_widgets["layer_a"]
    solo_events: list[tuple[str, bool]] = []
    state_events: list[tuple[str, dict[str, Any]]] = []
    tab.layer_solo_toggled.connect(lambda layer_id, checked: solo_events.append((layer_id, checked)))
    tab.layer_changed.connect(lambda layer_id, state: state_events.append((layer_id, state)))

    first_widget._solo_btn.click()

    assert solo_events == [("layer_a", True)]
    assert state_events == []
    assert tab._layer_widgets["layer_a"] is first_widget
    assert tab._layer_widgets["layer_a"]._solo_btn.isChecked()

    tab.set_layer_solo_states(
        [
            {"layer_id": "layer_a", "solo": False},
            {"layer_id": "layer_b", "solo": True},
        ]
    )

    assert tab._layer_widgets["layer_a"] is first_widget
    assert not first_widget._solo_btn.isChecked()
    assert tab._layer_widgets["layer_b"]._solo_btn.isChecked()
    # La sincronización programática no vuelve a emitir señales.
    assert solo_events == [("layer_a", True)]
    assert state_events == []

    tab.close()
    _ = app


def test_catalog_contains_all_fourteen_environments() -> None:
    paths = AppPaths.detect(frozen=False)
    catalog = EnvironmentCatalog(paths)

    ids = {e.environment_id for e in catalog.list_all()}

    assert len(ids) == 14
    assert {
        "bosque",
        "playa",
        "calle",
        "wc_bano",
        "oficina",
        "restaurante",
        "habitacion",
        "iglesia",
    } <= ids


@pytest.mark.unit
@pytest.mark.unit
def test_frozen_portable_paths_resolve_bundled_catalogs_and_manifests(
    portable_paths: AppPaths,
) -> None:
    """Los recursos inmutables siguen accesibles aunque app_root solo contenga datos portables."""
    catalog = EnvironmentCatalog(portable_paths)
    assert len(catalog.list_all()) == 14

    acoustics, layers, _ = build_environment_ui_state("bosque", portable_paths)
    assert "room_size" in acoustics
    assert len(layers) == 3

    from models.model_manager import ModelManager
    from presets.preset_manager import PresetManager

    assert PresetManager(portable_paths).list_all()
    assert ModelManager(portable_paths).list_entries()
    assert portable_paths.resolve_resource("manifests", "ambience_manifest.json").is_file()


def test_ir_convolution_with_dirac_impulse_preserves_signal() -> None:
    voice = make_sine_buffer(
        freq_hz=440.0,
        duration_s=0.25,
        sample_rate=48000,
        channels=2,
        amplitude=0.4,
    )

    ir = np.zeros((2, 256), dtype=np.float32)
    ir[:, 0] = 1.0  # Impulso unitario sin retardo ni coloración

    ir_buf = AudioBuffer(samples=ir, sample_rate=48000)

    out = simulate_acoustic_space(
        voice,
        ir_buffer=ir_buf,
        acoustics={
            "reverb_mix": 100.0,
            "predelay_ms": 0.0,
            "damping": 0.0,
            "direct_level": 0.0,
            "reflected_level": 100.0,
            "stereo_width": 100.0,
            "apparent_distance": 0.0,
            "pan": 0.0,
        },
    )

    corr = float(
        np.corrcoef(voice.samples[0], out.samples[0])[0, 1]
    )

    assert corr > 0.999


@pytest.mark.unit
def test_algorithmic_room_ir_and_environment_stage_with_local_layer(
    tmp_path: Path,
    portable_paths: AppPaths,
) -> None:
    ir_short = build_algorithmic_room_ir(
        48000,
        room_size=25.0,
        decay_s=0.3,
        damping=50.0,
    )

    ir_long = build_algorithmic_room_ir(
        48000,
        room_size=85.0,
        decay_s=2.5,
        damping=20.0,
    )

    # La cola de 2.5 s debe conservar más energía tardía que la de 0.3 s.
    tail_short = float(np.sum(ir_short.samples[:, 12000:] ** 2))
    tail_long = float(np.sum(ir_long.samples[:, 12000:] ** 2))

    assert tail_long > tail_short

    layer_wav = tmp_path / "amb_tone.wav"

    amb_buf = make_sine_buffer(
        freq_hz=1200.0,
        duration_s=0.2,
        sample_rate=48000,
        amplitude=0.2,
    )

    sf.write(
        str(layer_wav),
        amb_buf.samples.T,
        48000,
        subtype="FLOAT",
    )

    voice = make_sine_buffer(
        freq_hz=260.0,
        duration_s=0.3,
        sample_rate=48000,
        amplitude=0.4,
    )

    cfg = PipelineConfig(
        environment_enabled=True,
        environment_id="habitacion",
        master_ambience_volume=0.8,
        ambience_layers=[
            {
                "layer_id": "l1",
                "file_path": str(layer_wav),
                "enabled": True,
                "volume": 0.5,
                "pan": 0.0,
                "solo": False,
            }
        ],
    )

    out = apply_environment_stage(
        voice,
        config=cfg,
        paths=portable_paths,
    )

    assert out.num_samples == voice.num_samples
    assert np.all(np.isfinite(out.samples))


@pytest.mark.unit
def test_ambience_manager_consent_and_ui_state(
    portable_paths: AppPaths,
) -> None:
    src_paths = AppPaths.detect(frozen=False)
    mgr = AmbienceManager(src_paths)

    with pytest.raises(ResourceDownloadError):
        mgr.download_resource(
            "amb_bosque_pajaros",
            user_consented=False,
        )

    assert (
        mgr.get_status("amb_bosque_pajaros")
        == ResourceStatus.NOT_DOWNLOADED
    )

    acoustics, layers, _ = build_environment_ui_state(
        "bosque",
        src_paths,
    )

    assert "room_size" in acoustics
    assert len(layers) == 3
