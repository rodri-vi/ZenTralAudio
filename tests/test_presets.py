"""Pruebas unitarias de presets profesionales adaptativos y presets personalizados."""

from __future__ import annotations

import pytest

from core.paths import AppPaths
from presets.preset_manager import PresetManager


@pytest.mark.unit
def test_builtin_presets_present_and_adaptive(portable_paths: AppPaths) -> None:
    src_paths = AppPaths.detect(frozen=False)
    pm = PresetManager(src_paths)
    names = {p["name"] for p in pm.list_all()}
    expected = {
        "Studio Voice",
        "Podcast Pro",
        "Noise Removal",
        "Heavy Noise",
        "Echo Removal",
        "Old Recording",
        "Telephone Voice",
        "Natural Voice",
        "Maximum Recovery",
    }
    assert expected <= names

    # DeepFilterNet3 no tiene dereverb nativo -> el preset Echo Removal debe activar dereverb DSP
    resolved = pm.resolve_for_model("echo_removal", "deepfilternet3")
    assert float(resolved.params.get("dereverb_amount", 0.0)) >= 0.65
    assert "model_vf_mode" not in resolved.params


@pytest.mark.unit
def test_save_and_delete_custom_preset(portable_paths: AppPaths) -> None:
    pm = PresetManager(portable_paths)
    saved = pm.save_custom(
        "Mi Voz Clara",
        intensity=82,
        params={"nr_amount": 0.4, "vocal_presence": 3.0},
        model_id="deepfilternet3",
    )
    assert saved.is_file()
    found = [p for p in pm.list_all() if p["name"] == "Mi Voz Clara"]
    assert len(found) == 1
    assert pm.delete_custom(found[0]["preset_id"]) is True
    assert not saved.exists()