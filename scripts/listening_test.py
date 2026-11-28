"""Validación auditiva real de la reproducción mediante loopback (BlackHole).

1. Genera un seno de prueba (o usa el archivo indicado).
2. Lanza ZenTralAudio (``--playback-selftest``) reproduciendo por "BlackHole 16ch"
   con renders acústicos cada 2 s, como si se movieran los sliders.
3. Graba la salida del loopback en un proceso independiente.
4. Busca discontinuidades (huecos, saltos, clics) en la grabación.

Requiere BlackHole y permiso de micrófono para la app que ejecuta el script.
Uso:  .venv/bin/python scripts/listening_test.py [segundos] [--app RUTA_A_ZenTralAudio]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "BlackHole 16ch"


def find_device(name: str, *, output: bool) -> int:
    import sounddevice as sd

    key = "max_output_channels" if output else "max_input_channels"
    for index, info in enumerate(sd.query_devices()):
        if info["name"] == name and info[key] >= 2:
            return index
    raise RuntimeError(f"No se encontró el dispositivo '{name}'.")


def analyze(recording: np.ndarray, sr: int, freq: float) -> dict[str, object]:
    x = recording[:, 0].astype(np.float64)
    active = np.flatnonzero(np.abs(x) > 1e-3)
    if active.size == 0:
        return {"error": "grabación en silencio"}
    x = x[active[0] + sr // 10 : active[-1] - sr // 10]
    amp = float(np.percentile(np.abs(x), 99.5))
    d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    expected = amp * (2 * np.pi * freq / sr) ** 2
    events = np.flatnonzero(d2 > expected * 8 + 2e-4)
    groups: list[int] = []
    for e in events:
        if not groups or e - groups[-1] > sr * 0.005:
            groups.append(int(e))
    # Huecos: tramos de >2 ms casi en silencio dentro del tono.
    quiet = np.abs(x) < amp * 0.02
    gaps = 0
    run = 0
    for value in quiet[:: max(1, sr // 4000)]:
        run = run + 1 if value else 0
        if run == 8:
            gaps += 1
    return {
        "seconds_analyzed": round(x.size / sr, 2),
        "amplitude": round(amp, 3),
        "discontinuities": len(groups),
        "first_at_s": [round(g / sr, 3) for g in groups[:10]],
        "silent_gaps": gaps,
    }


def main() -> int:
    args = sys.argv[1:]
    seconds = float(args.pop(0)) if args and args[0].replace(".", "", 1).isdigit() else 20.0
    app = None
    if "--app" in args:
        app = args[args.index("--app") + 1]
    work = Path(tempfile.mkdtemp(prefix="zentral_listen_"))
    sr, freq = 44100, 440.0
    t = np.arange(int(sr * (seconds + 30))) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    source = work / "sine440.wav"
    sf.write(source, np.stack([tone, tone], 1), sr, subtype="PCM_24")

    out_index = find_device(DEVICE, output=True)
    data_dir = work / "data"
    (data_dir / "config").mkdir(parents=True)
    (data_dir / "config" / "settings.json").write_text(
        json.dumps({"schema_version": 1, "output_device_index": out_index, "loudness_match_preview": False}),
        encoding="utf-8",
    )
    recording_path = work / "recording.wav"
    record_seconds = seconds + 25
    recorder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, sounddevice as sd, soundfile as sf\n"
            f"dev = {find_device(DEVICE, output=False)}\n"
            f"x = sd.rec(int({record_seconds} * 48000), samplerate=48000, channels=2, device=dev, blocking=True)\n"
            f"sf.write(r'{recording_path}', x, 48000, subtype='FLOAT')\n",
        ]
    )
    time.sleep(1.0)
    env = dict(os.environ, ZENTRALAUDIO_DATA_DIR=str(data_dir))
    command = [app] if app else [sys.executable, str(ROOT / "main.py")]
    command += ["--playback-selftest", str(source), "--selftest-seconds", str(seconds), "--selftest-volume", "1.0"]
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    report_text = proc.stdout[proc.stdout.find("{") :]
    selftest = json.loads(report_text) if report_text.strip() else {"error": proc.stderr[-800:]}
    recorder.wait()
    recording, rec_sr = sf.read(recording_path, dtype="float32", always_2d=True)
    result = {"selftest": selftest, "recording": analyze(recording, rec_sr, freq)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = selftest.get("ok") and result["recording"].get("discontinuities") == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
