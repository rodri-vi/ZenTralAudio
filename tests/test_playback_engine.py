"""Pruebas del callback de reproducción en tiempo real y de la gestión del stream."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.app_controller import CROSSFADE_FRAMES, AppController, _PlaybackState


class _Flags:
    output_underflow = False

    def __bool__(self) -> bool:
        return False


def _controller(samples: np.ndarray, sample_rate: int = 48000, **state: Any) -> AppController:
    """Instancia mínima con solo el estado que usa el callback (sin Qt ni dispositivos)."""
    ctrl = AppController.__new__(AppController)
    ctrl._pb_lock = threading.Lock()
    ctrl._playback = _PlaybackState(monitor_volume=1.0, loudness_match=False)
    ctrl._playback.samples_original = samples
    ctrl._playback.sample_rate = sample_rate
    ctrl._playback.playing = True
    for key, value in state.items():
        setattr(ctrl._playback, key, value)
    return ctrl


def _pull(ctrl: AppController, frames: int) -> np.ndarray:
    out = np.full((frames, 2), 7.0, dtype=np.float32)
    ctrl._audio_callback(out, frames, None, _Flags())
    return out


def _ramp(channels: int, n: int) -> np.ndarray:
    base = (np.arange(n, dtype=np.float32) / n) * 0.5
    return np.stack([base * (1.0 if c == 0 else -1.0) for c in range(channels)])


@pytest.mark.unit
@pytest.mark.parametrize("channels", [1, 2])
def test_callback_streams_mono_and_stereo_contiguously(channels: int) -> None:
    samples = _ramp(channels, 10_000)
    ctrl = _controller(samples)
    blocks = [_pull(ctrl, 1024) for _ in range(5)]
    played = np.concatenate(blocks)
    expected_right = samples[1 if channels == 2 else 0, :5120]
    np.testing.assert_allclose(played[:, 0], samples[0, :5120], atol=1e-7)
    np.testing.assert_allclose(played[:, 1], expected_right, atol=1e-7)
    assert ctrl._playback.position_frames == 5120


@pytest.mark.unit
def test_callback_stops_at_end_and_zero_fills() -> None:
    samples = _ramp(2, 1500)
    ctrl = _controller(samples)
    _pull(ctrl, 1024)
    out = _pull(ctrl, 1024)
    np.testing.assert_allclose(out[:476, 0], samples[0, 1024:1500], atol=1e-7)
    assert np.all(out[476:] == 0.0)
    assert ctrl._playback.playing is False
    assert np.all(_pull(ctrl, 256) == 0.0)


@pytest.mark.unit
def test_callback_loops_selection_without_gaps() -> None:
    samples = _ramp(2, 4000)
    ctrl = _controller(samples, loop=True, selection_frames=(1000, 1600), position_frames=1000)
    played = np.concatenate([_pull(ctrl, 512) for _ in range(4)])
    expected = np.concatenate([samples[0, 1000:1600]] * 4)[:2048]
    np.testing.assert_allclose(played[:, 0], expected, atol=1e-7)
    assert ctrl._playback.playing is True


@pytest.mark.unit
def test_callback_applies_volume_gain_and_clipping() -> None:
    samples = np.full((2, 2048), 0.8, dtype=np.float32)
    processed = np.full((2, 2048), 0.8, dtype=np.float32)
    ctrl = _controller(samples, samples_processed=processed, gain_comp_processed=2.0, monitor_volume=0.5)
    out = _pull(ctrl, 512)
    np.testing.assert_allclose(out, 0.8, atol=1e-6)  # 0.8 * 2.0 * 0.5
    ctrl._playback.monitor_volume = 1.0
    out = _pull(ctrl, 512)
    assert out.max() <= np.float32(0.999)


@pytest.mark.unit
def test_new_render_during_playback_is_crossfaded_not_switched_abruptly() -> None:
    old = np.full((2, 8192), 0.5, dtype=np.float32)
    new = np.full((2, 8192), -0.5, dtype=np.float32)
    ctrl = _controller(np.zeros((2, 8192), np.float32), samples_processed=old)
    _pull(ctrl, 1024)
    with ctrl._pb_lock:
        ctrl._playback.swap_from = old
        ctrl._playback.swap_gain = 1.0
        ctrl._playback.swap_tag = "B"
        ctrl._playback.samples_processed = new
    out = _pull(ctrl, 1024)
    assert out[0, 0] == pytest.approx(0.5, abs=1e-3)
    assert np.max(np.abs(np.diff(out[:, 0]))) < 1.0 / CROSSFADE_FRAMES * 1.01 + 1e-6
    np.testing.assert_allclose(out[CROSSFADE_FRAMES:, 0], -0.5, atol=1e-6)
    assert ctrl._playback.swap_from is None


@pytest.mark.unit
def test_bypass_toggle_crossfades_between_original_and_processed() -> None:
    orig = np.full((2, 8192), 0.2, dtype=np.float32)
    processed = np.full((2, 8192), 0.6, dtype=np.float32)
    ctrl = _controller(orig, samples_processed=processed)
    _pull(ctrl, 1024)
    ctrl._on_bypass_toggled(False)
    out = _pull(ctrl, 1024)
    assert out[0, 0] == pytest.approx(0.6, abs=1e-3)
    np.testing.assert_allclose(out[CROSSFADE_FRAMES:, 0], 0.2, atol=1e-6)


@pytest.mark.unit
def test_paused_callback_outputs_silence_and_keeps_position() -> None:
    ctrl = _controller(_ramp(2, 4096), position_frames=300)
    ctrl._playback.playing = False
    out = _pull(ctrl, 512)
    assert np.all(out == 0.0)
    assert ctrl._playback.position_frames == 300


@pytest.mark.unit
def test_stream_is_recreated_when_sample_rate_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    import sounddevice as sd

    created: list[dict[str, Any]] = []

    class FakeStream:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)
            self.samplerate = kwargs["samplerate"]
            self.latency = 0.1
            self.active = False
            self.closed = False

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    ctrl = _controller(_ramp(2, 4096), sample_rate=44100)
    ctrl._sd_stream = None
    ctrl.config_manager = SimpleNamespace(settings=SimpleNamespace(output_device_index=None))

    ctrl._ensure_output_stream(44100)
    ctrl._ensure_output_stream(44100)
    assert len(created) == 1
    first = ctrl._sd_stream
    assert created[0]["latency"] >= 0.08

    ctrl._ensure_output_stream(48000)
    assert len(created) == 2
    assert first.closed
    assert ctrl._sd_stream.samplerate == 48000
    assert ctrl._playback.output_latency_s == pytest.approx(0.1)


@pytest.mark.unit
def test_stale_render_results_are_ignored() -> None:
    ctrl = _controller(np.zeros((2, 1024), np.float32))
    ctrl._render_generation = 3
    ctrl._apply_rendered_audio((2, np.ones((2, 1024), np.float32)))
    assert ctrl._playback.samples_processed is None
