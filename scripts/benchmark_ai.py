"""Mide el tiempo de los modelos IA a través del pipeline real de la aplicación.

Uso:  ZENTRALAUDIO_MACOS_USER_DIRS=1 .venv/bin/python scripts/benchmark_ai.py [segundos] [modelo...]

Para cada modelo procesa una voz sintética con ruido en dos variantes: estéreo
real (canales distintos) y "dual mono" (mismo canal duplicado), y muestra el
tiempo, el factor sobre tiempo real y si la salida es válida.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths  # noqa: E402
from engine.audio_types import AudioBuffer  # noqa: E402
from engine.processing_graph import PipelineConfig  # noqa: E402
from engine.processing_pipeline import ProcessingPipeline  # noqa: E402
from models.model_registry import get_global_registry  # noqa: E402

ALL_MODELS = ("deepfilternet3", "mossformer2_se_48k", "mossformergan_se_16k", "frcrn_se_16k", "voicefixer")


def voice_like(seconds: float, sr: int = 44100) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(1)
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum((0.3 / k) * np.sin(k * phase) for k in range(1, 10)) * (np.sin(2 * np.pi * 1.3 * t) > -0.2)
    left = (voice + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
    right = (0.9 * voice + 0.03 * rng.standard_normal(t.size)).astype(np.float32)
    return left, right


def main() -> int:
    args = sys.argv[1:]
    seconds = float(args.pop(0)) if args and args[0].replace(".", "", 1).isdigit() else 30.0
    models = args or list(ALL_MODELS)
    paths = AppPaths.detect()
    sr = 44100
    left, right = voice_like(seconds, sr)
    cases = {"estereo": np.stack([left, right]), "dual_mono": np.stack([left, left])}
    for model_id in models:
        for name, samples in cases.items():
            pipeline = ProcessingPipeline()
            config = PipelineConfig(
                model_id=model_id,
                intensity=100,
                advanced_params={"nr_enabled": False, "comp_enabled": False},
            )
            started = time.perf_counter()
            out = pipeline.render(AudioBuffer(samples=samples, sample_rate=sr), config, paths=paths)
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "case": name,
                        "audio_s": seconds,
                        "seconds": round(elapsed, 1),
                        "x_realtime": round(seconds / elapsed, 2),
                        "finite": bool(np.isfinite(out.samples).all()),
                        "shape_ok": out.samples.shape == samples.shape,
                    }
                ),
                flush=True,
            )
        get_global_registry(paths).unload_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
