"""Previsualización progresiva, intensidad sin re-convolución, memoria y cambio de dispositivo."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from engine.audio_types import AudioBuffer
from engine.playback_utils import splice_with_crossfade
from engine.processing_graph import PipelineConfig
from engine.processing_pipeline import (
    ProcessingPipeline,
    default_cache_budget_bytes,
    is_dual_mono,
    physical_memory_bytes,
)
from environments import acoustic_simulator
from environments.acoustic_simulator import _ArrayLRU, clear_acoustic_caches


def _voice(seconds: float, sr: int = 22050, seed: int = 0) -> AudioBuffer:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    base = 0.25 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.9 * t) > 0)
    data = np.stack([base + 0.02 * rng.standard_normal(t.size), 0.8 * base + 0.02 * rng.standard_normal(t.size)])
    return AudioBuffer(samples=data.astype(np.float32), sample_rate=sr)


def _config(**overrides: Any) -> PipelineConfig:
    base = {
        "ai_enabled": False,
        "intensity": 70,
        "environment_enabled": True,
        "environment_id": "iglesia",
        "acoustics": {"reverb_mix": 40.0, "decay_s": 1.2},
        "advanced_params": {"nr_amount": 0.5},
    }
    base.update(overrides)
    return PipelineConfig(**base)


@pytest.mark.unit
def test_intensity_change_reuses_convolution_and_matches_full_render(
    monkeypatch: pytest.MonkeyPatch, portable_paths
) -> None:
    clear_acoustic_caches()
    calls = {"n": 0}
    real = acoustic_simulator.oaconvolve

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(acoustic_simulator, "oaconvolve", counting)
    source = _voice(3.0)
    pipe = ProcessingPipeline()
    pipe.render(source, _config(intensity=70), paths=portable_paths)
    first_calls = calls["n"]
    assert first_calls == 4  # original + voz procesada, 2 canales cada una

    out_40 = pipe.render(source, _config(intensity=40), paths=portable_paths)
    out_90 = pipe.render(source, _config(intensity=90), paths=portable_paths)
    assert calls["n"] == first_calls, "Cambiar la intensidad no debe recalcular la convolución"

    # Referencia: un pipeline nuevo sin cachés debe dar el mismo resultado.
    clear_acoustic_caches()
    ref_40 = ProcessingPipeline().render(source, _config(intensity=40), paths=portable_paths)
    np.testing.assert_allclose(out_40.samples, ref_40.samples, atol=2e-5)
    assert np.isfinite(out_90.samples).all()


@pytest.mark.unit
def test_array_lru_respects_budget_but_keeps_latest() -> None:
    cache = _ArrayLRU(max_entries=2, max_bytes=1000)
    a = np.zeros(100, np.float32)  # 400 B
    b = np.zeros(100, np.float32)
    c = np.zeros(400, np.float32)  # 1600 B > presupuesto
    cache.put("a", a)
    cache.put("b", b)
    assert cache.get("a") is a and cache.get("b") is b
    cache.put("c", c)
    assert cache.get("c") is c  # la más reciente se conserva aunque exceda
    assert cache.get("a") is None and cache.get("b") is None
    assert not c.flags.writeable


@pytest.mark.unit
def test_cache_budget_is_proportional_to_ram() -> None:
    budget = default_cache_budget_bytes()
    assert 768 * 1024**2 <= budget <= 4 * 1024**3
    assert budget <= physical_memory_bytes() * 0.25 + 1
    pipe = ProcessingPipeline(memory_budget_bytes=1_000_000_000)
    assert pipe.memory_budget_bytes == 1_000_000_000


@pytest.mark.unit
def test_dual_mono_detection() -> None:
    x = np.random.default_rng(0).standard_normal(500_000).astype(np.float32)
    assert is_dual_mono(np.stack([x, x]))
    y = x.copy()
    y[400_000] += 0.01
    assert not is_dual_mono(np.stack([x, y]))
    assert not is_dual_mono(x[np.newaxis, :])


@pytest.mark.unit
def test_ai_stage_processes_identical_channels_once(monkeypatch: pytest.MonkeyPatch, portable_paths) -> None:
    seen: list[int] = []

    class FakeAdapter:
        def is_ready_for_inference(self) -> bool:
            return True

        def get_capabilities(self):
            return SimpleNamespace(required_sample_rate=22050)

        def process(self, buf, params, **_kw):
            seen.append(buf.channels)
            return AudioBuffer(samples=buf.samples * 0.5, sample_rate=buf.sample_rate)

    registry = SimpleNamespace(get_adapter=lambda _mid: FakeAdapter())
    monkeypatch.setattr("models.model_registry.get_global_registry", lambda _p: registry)
    mono = _voice(1.0).samples[:1]
    source = AudioBuffer(samples=np.repeat(mono, 2, axis=0), sample_rate=22050)
    pipe = ProcessingPipeline()
    out = pipe._run_ai_stage(
        source, _config(ai_enabled=True), src_fp="x", paths=portable_paths, cancel_token=None, progress_cb=None
    )
    assert seen == [1]
    assert out.channels == 2
    np.testing.assert_array_equal(out.samples[0], out.samples[1])
    np.testing.assert_allclose(out.samples[0], mono[0] * 0.5)


@pytest.mark.unit
def test_preview_window_is_close_to_full_render_and_does_not_touch_caches(portable_paths) -> None:
    clear_acoustic_caches()
    source = _voice(20.0)
    sr = source.sample_rate
    pipe = ProcessingPipeline()
    config = _config(intensity=80, advanced_params={"nr_amount": 0.4, "eq_mid_gain": 3.0})
    start, end = 8 * sr, 14 * sr
    window = pipe.render_preview_window(source, config, start, end, paths=portable_paths)
    assert window is not None and window.shape == (2, end - start)
    assert acoustic_simulator.acoustic_cache_bytes() == 0
    assert pipe._voice_cache.get(config.voice_stage_cache_key("x")) is None

    full = pipe.render(source, config, paths=portable_paths).samples[:, start:end]
    err = window - full
    snr = 10 * np.log10(np.sum(full**2) / np.sum(err**2))
    assert snr > 20.0, f"Ventana demasiado distinta del render completo ({snr:.1f} dB)"
    # Con la voz ya cacheada, la ventana no aporta nada.
    assert pipe.render_preview_window(source, config, start, end, paths=portable_paths) is None


@pytest.mark.unit
def test_splice_with_crossfade_has_no_jumps() -> None:
    base = np.full((2, 10_000), 0.5, np.float32)
    window = np.full((2, 3_000), -0.5, np.float32)
    out = splice_with_crossfade(base, window, 4_000, 200)
    assert np.all(out[:, :4_000] == 0.5)
    assert np.all(out[:, 4_200:6_800] == -0.5)
    assert np.all(out[:, 7_000:] == 0.5)
    assert np.max(np.abs(np.diff(out[0]))) <= 1.0 / 199 + 1e-6
    assert base[0, 5_000] == 0.5  # el original no se modifica


@pytest.mark.unit
def test_output_device_change_reopens_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import app_controller
    from app.app_controller import AppController, _PlaybackState

    ctrl = AppController.__new__(AppController)
    ctrl._pb_lock = threading.Lock()
    ctrl._playback = _PlaybackState()
    ctrl._playback.sample_rate = 48000
    ctrl.config_manager = SimpleNamespace(settings=SimpleNamespace(output_device_index=None))
    ctrl._device_watch_timer = SimpleNamespace(stop=lambda: None)
    ctrl._sd_stream = SimpleNamespace(active=True)
    ctrl._stream_default_device = 10
    reasons: list[str] = []
    ctrl.reopen_output_stream = lambda reason: reasons.append(reason)

    monkeypatch.setattr(app_controller, "default_output_device_id", lambda: 10)
    ctrl._check_output_device()
    assert reasons == []

    monkeypatch.setattr(app_controller, "default_output_device_id", lambda: 42)
    ctrl._check_output_device()
    assert reasons and "predeterminado" in reasons[0]

    reasons.clear()
    ctrl._sd_stream = SimpleNamespace(active=False)
    ctrl._check_output_device()
    assert reasons and "detuvo" in reasons[0]


@pytest.mark.unit
def test_core_audio_default_device_query_is_safe() -> None:
    from core.macos_audio import default_output_device_id

    value = default_output_device_id()
    assert value is None or value > 0
