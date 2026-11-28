"""Ejecución aislada de modelos, con worker portable y fallback a .venv de desarrollo."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from core.exceptions import JobCancelledError, ModelInferenceError
from core.job_manager import CancellationToken
from engine.audio_types import AudioBuffer
from models.isolated_runners.runner_protocol import IsolatedRunRequest

logger = logging.getLogger("zentralaudio.isolated_runner")
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_BUNDLED_WORKER_NAME = "ZenTralModelWorker.exe"


def find_bundled_worker(model_dir: Path) -> Path | None:
    """Localiza el worker autocontenido relativo a la carpeta portable."""
    portable_root = model_dir.resolve().parent.parent
    worker_name = _BUNDLED_WORKER_NAME if os.name == "nt" else "ZenTralModelWorker"
    candidates = (
        portable_root / "runtimes" / "ZenTralModelWorker" / worker_name,
        portable_root / "runtimes" / "ZenTralModelWorker" / _BUNDLED_WORKER_NAME,
        portable_root / "runtimes" / "ZenTralModelWorker" / "ZenTralModelWorker",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def find_isolated_python(model_dir: Path) -> Path | None:
    """Busca primero el worker incluido y, después, un .venv aislado de desarrollo."""
    bundled = find_bundled_worker(model_dir)
    if bundled is not None:
        return bundled
    candidates = (
        model_dir / ".venv" / "Scripts" / "python.exe",
        model_dir / ".venv" / "bin" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _portable_worker_environment(weights_dir: Path) -> dict[str, str]:
    """Redirige cachés al portable y bloquea descargas de modelos durante inferencia."""
    portable_root = weights_dir.resolve().parent.parent
    worker_home = portable_root / "cache" / "offline_worker_home"
    hf_home = portable_root / "cache" / "huggingface"
    worker_home.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(worker_home),
            "USERPROFILE": str(worker_home),
            "HF_HOME": str(hf_home),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        }
    )
    return env


def run_in_isolated_subprocess(
    *,
    python_exe: Path,
    worker_script: Path,
    buffer: AudioBuffer,
    model_id: str,
    weights_dir: Path,
    device: str,
    params: dict[str, Any],
    cancel_token: CancellationToken | None = None,
    timeout_seconds: float = 600.0,
) -> AudioBuffer:
    """Intercambia WAV float32 con el worker portable o el intérprete del .venv."""
    if not python_exe.is_file():
        raise ModelInferenceError(f"Runtime aislado no encontrado: {python_exe}")
    # En macOS los ejecutables no llevan extensión .exe. Identificamos el worker
    # empaquetado por su carpeta estable, no por el nombre del binario.
    bundled_worker = python_exe.parent.name.lower() == "zentralmodelworker"
    if not bundled_worker and not worker_script.is_file():
        raise ModelInferenceError(f"Script de inferencia aislada no encontrado: {worker_script}")

    with tempfile.TemporaryDirectory(prefix=f"zentral_iso_{model_id}_") as tmp_dir:
        tmp = Path(tmp_dir)
        in_wav = tmp / "input.wav"
        out_wav = tmp / "output.wav"
        req_json = tmp / "request.json"
        sf.write(str(in_wav), buffer.samples.T, buffer.sample_rate, subtype="FLOAT", format="WAV")
        req = IsolatedRunRequest(
            model_id=model_id,
            input_wav=str(in_wav),
            output_wav=str(out_wav),
            weights_dir=str(weights_dir),
            device=device,
            params=params,
        )
        req_json.write_text(json.dumps(req.to_json_dict(), indent=2), encoding="utf-8")

        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        command = (
            [str(python_exe), str(req_json)]
            if bundled_worker
            else [str(python_exe), str(worker_script), str(req_json)]
        )
        if bundled_worker:
            kwargs["env"] = _portable_worker_environment(weights_dir)
        if os.name == "nt":
            kwargs["creationflags"] = _CREATE_NO_WINDOW

        proc = subprocess.Popen(command, **kwargs)
        deadline = time.monotonic() + timeout_seconds
        while proc.poll() is None:
            if cancel_token and cancel_token.is_cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise JobCancelledError(f"Inferencia aislada de '{model_id}' cancelada.")
            if time.monotonic() > deadline:
                proc.kill()
                raise ModelInferenceError(f"La inferencia aislada de '{model_id}' superó el tiempo límite.")
            time.sleep(0.05)

        stdout, stderr = proc.communicate()
        if proc.returncode != 0 or not out_wav.is_file():
            raise ModelInferenceError(
                f"Fallo en el runner aislado de '{model_id}' (código {proc.returncode}).",
                details=(stderr or stdout or "").strip(),
            )
        out_data, out_sr = sf.read(str(out_wav), dtype="float32", always_2d=True)
        return AudioBuffer(samples=out_data.T, sample_rate=int(out_sr))
