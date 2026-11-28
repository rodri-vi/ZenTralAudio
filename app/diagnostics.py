"""Autodiagnóstico de reproducción en tiempo real (``--playback-selftest``).

Abre la interfaz real, carga un audio, lo reproduce por el dispositivo de salida
configurado y mide cada callback de CoreAudio/PortAudio:

- margen hasta el instante en que el bloque debe sonar (``outputBufferDacTime``);
  un margen negativo o casi nulo significa que el bloque llegó tarde (corte);
- underflows que reporta PortAudio;
- duración del propio callback.

Opcionalmente lanza renders de previsualización acústica cada 2 s, como si se
movieran los sliders. Por defecto reproduce a volumen 0 (silencio audible, misma
temporización), para poder ejecutarlo sin molestar.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np


def run_playback_selftest(
    window: Any,
    audio_path: str,
    *,
    seconds: float = 15.0,
    with_renders: bool = True,
    volume: float = 0.0,
) -> dict[str, Any]:
    from PySide6.QtCore import QEventLoop, QTimer

    controller = window.controller
    stats: dict[str, list[float] | int] = {"margins": [], "durations": [], "underflows": 0, "calls": 0}
    original_callback = controller._audio_callback

    def instrumented(outdata, frames, time_info, status) -> None:  # noqa: ANN001
        started = time.perf_counter()
        original_callback(outdata, frames, time_info, status)
        stats["durations"].append(time.perf_counter() - started)  # type: ignore[union-attr]
        stats["calls"] = int(stats["calls"]) + 1
        if status and status.output_underflow:
            stats["underflows"] = int(stats["underflows"]) + 1
        stream = controller._sd_stream
        if stream is not None and time_info is not None:
            try:
                stats["margins"].append(time_info.outputBufferDacTime - stream.time)  # type: ignore[union-attr]
            except Exception:
                pass

    controller._audio_callback = instrumented
    controller._playback.monitor_volume = max(0.0, min(1.0, float(volume)))
    controller.load_audio_file(audio_path)

    loop = QEventLoop()
    state: dict[str, Any] = {"started": None, "renders": 0, "info": {}}

    def tick() -> None:
        pb = controller._playback
        if state["started"] is None:
            if pb.samples_processed is not None:
                controller.play()
                state["started"] = time.time()
            elif time.time() - state.setdefault("waiting", time.time()) > 120:
                loop.quit()
            return
        elapsed = time.time() - state["started"]
        if with_renders and int(elapsed / 2.0) > state["renders"]:
            state["renders"] += 1
            controller._env_enabled = True
            controller._on_acoustic_param_changed("reverb_mix", 20 + (state["renders"] * 13) % 60)
            controller._on_acoustic_param_changed("decay_s", 0.8 + (state["renders"] % 4) * 0.6)
        if elapsed >= seconds:
            stream = controller._sd_stream
            state["info"] = {
                "samplerate": float(getattr(stream, "samplerate", 0.0) or 0.0),
                "latency_ms": round(float(getattr(stream, "latency", 0.0) or 0.0) * 1000, 1),
            }
            controller.stop()
            loop.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(20)
    loop.exec()
    timer.stop()
    controller._audio_callback = original_callback

    margins = np.asarray(stats["margins"][5:], dtype=np.float64)
    durations = np.asarray(stats["durations"][5:], dtype=np.float64)
    late = int((margins < 0.001).sum()) if margins.size else 0
    return {
        "file": audio_path,
        "seconds": seconds,
        "renders_during_playback": state["renders"],
        **state["info"],
        "callbacks": int(stats["calls"]),
        "underflows": int(stats["underflows"]),
        "late_callbacks": late,
        "min_margin_ms": round(float(margins.min()) * 1000, 2) if margins.size else None,
        "median_margin_ms": round(float(np.median(margins)) * 1000, 2) if margins.size else None,
        "max_callback_ms": round(float(durations.max()) * 1000, 3) if durations.size else None,
        "ok": bool(margins.size) and late == 0 and int(stats["underflows"]) == 0,
    }


def format_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
