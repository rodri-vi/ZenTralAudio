"""Ejecución aislada de modelos, con worker portable y fallback a .venv de desarrollo."""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
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


SHARED_RUNTIME_DIRNAME = "ai-python"


def shared_runtime_python(data_root: Path) -> Path:
    """Intérprete del runtime IA compartido (clearvoice + voicefixer) de la raíz de datos."""
    base = data_root / "runtimes" / SHARED_RUNTIME_DIRNAME
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_isolated_python(model_dir: Path) -> Path | None:
    """Busca el worker incluido, un .venv por modelo o el runtime IA compartido.

    El runtime compartido (``<datos>/runtimes/ai-python``) evita instalar
    clearvoice/voicefixer en el entorno principal: sus dependencias (streamlit,
    opencv, soundfile 0.12...) chocan con las de la aplicación.
    """
    bundled = find_bundled_worker(model_dir)
    if bundled is not None:
        return bundled
    candidates = (
        model_dir / ".venv" / "Scripts" / "python.exe",
        model_dir / ".venv" / "bin" / "python",
        shared_runtime_python(model_dir.resolve().parent.parent),
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
            # Apple Silicon: ejecutar en CPU las operaciones sin kernel MPS.
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )
    return env


def run_in_isolated_subprocess(
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
    """Intercambia WAV float32 con el worker IA aislado.

    Usa un worker persistente que mantiene el modelo cargado entre llamadas
    (antes se lanzaba un proceso nuevo por cada canal y por cada segmento, y cada
    uno volvía a copiar los pesos y a cargar el modelo). Si el worker no admite
    el modo servidor (p. ej. un worker empaquetado antiguo), se usa una ejecución
    de un solo uso como antes.
    """
    if not python_exe.is_file():
        raise ModelInferenceError(f"Runtime aislado no encontrado: {python_exe}")
    # En macOS los ejecutables no llevan extensión .exe. Identificamos el worker
    # empaquetado por su carpeta estable, no por el nombre del binario.
    bundled_worker = python_exe.parent.name.lower() == "zentralmodelworker"
    if not bundled_worker and not worker_script.is_file():
        raise ModelInferenceError(f"Script de inferencia aislada no encontrado: {worker_script}")

    command = [str(python_exe)] if bundled_worker else [str(python_exe), str(worker_script)]
    with tempfile.TemporaryDirectory(prefix=f"zentral_iso_{model_id}_") as tmp_dir:
        tmp = Path(tmp_dir)
        in_wav = tmp / "input.wav"
        out_wav = tmp / "output.wav"
        sf.write(str(in_wav), buffer.samples.T, buffer.sample_rate, subtype="FLOAT", format="WAV")
        req = IsolatedRunRequest(
            model_id=model_id,
            input_wav=str(in_wav),
            output_wav=str(out_wav),
            weights_dir=str(weights_dir),
            device=device,
            params=params,
        )

        worker = _get_persistent_worker(command, weights_dir)
        if worker is not None:
            try:
                worker.request(req.to_json_dict(), cancel_token=cancel_token, timeout=timeout_seconds)
            except _ServerModeUnavailable:
                _disable_persistent_worker(command)
                worker = None
        if worker is None:
            _run_one_shot(command, req, tmp, weights_dir, model_id, cancel_token, timeout_seconds)

        if not out_wav.is_file():
            raise ModelInferenceError(f"El runner aislado de '{model_id}' no generó audio de salida.")
        out_data, out_sr = sf.read(str(out_wav), dtype="float32", always_2d=True)
        return AudioBuffer(samples=out_data.T, sample_rate=int(out_sr))


def _run_one_shot(
    command: list[str],
    req: IsolatedRunRequest,
    tmp: Path,
    weights_dir: Path,
    model_id: str,
    cancel_token: CancellationToken | None,
    timeout_seconds: float,
) -> None:
    """Modo histórico: un proceso por solicitud."""
    req_json = tmp / "request.json"
    req_json.write_text(json.dumps(req.to_json_dict(), indent=2), encoding="utf-8")
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        # Tanto el worker empaquetado como los runtimes Python aislados se ejecutan
        # sin red y con HOME/cachés dentro de la carpeta de datos de la app.
        "env": _portable_worker_environment(weights_dir),
    }
    if os.name == "nt":
        kwargs["creationflags"] = _CREATE_NO_WINDOW

    proc = subprocess.Popen([*command, str(req_json)], **kwargs)
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
    if proc.returncode != 0 or not Path(req.output_wav).is_file():
        raise ModelInferenceError(
            f"Fallo en el runner aislado de '{model_id}' (código {proc.returncode}).",
            details=(stderr or stdout or "").strip(),
        )


# --------------------------------------------------------------------------------------
# Worker persistente
# --------------------------------------------------------------------------------------

RESPONSE_PREFIX = "@@ZENTRAL "
WORKER_IDLE_SECONDS = 300.0
_STARTUP_TIMEOUT_SECONDS = 180.0


class _ServerModeUnavailable(Exception):
    """El worker no entiende ``--serve``: usar el modo de un solo uso."""


class PersistentWorker:
    """Proceso IA de larga duración que atiende solicitudes JSON por stdin.

    - Mantiene el último modelo cargado (el worker solo conserva uno a la vez para
      no agotar la RAM en equipos de 8 GB).
    - Cancelar o superar el tiempo límite mata el proceso; la siguiente solicitud
      lo vuelve a arrancar.
    - Se cierra solo tras ``WORKER_IDLE_SECONDS`` sin uso y al salir de la app.
    """

    def __init__(self, command: list[str], env: dict[str, str], log_path: Path) -> None:
        self._command = command
        self._env = env
        self._log_path = log_path
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._ready = False
        self._counter = 0
        self._idle_timer: threading.Timer | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_path.is_file() and self._log_path.stat().st_size > 5 * 1024 * 1024:
            self._log_path.unlink(missing_ok=True)
        log_handle = open(self._log_path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": log_handle,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": self._env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = _CREATE_NO_WINDOW
        self._proc = subprocess.Popen([*self._command, "--serve"], **kwargs)
        log_handle.close()
        self._lines = queue.Queue()
        self._ready = False
        proc = self._proc

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._lines.put(line.rstrip("\n"))
            self._lines.put(None)

        threading.Thread(target=_reader, name="ZenTralAIWorkerReader", daemon=True).start()
        logger.info("Worker IA persistente iniciado (pid %s)", proc.pid)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        self._ready = False
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if self.alive and self._proc is not None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=3.0)
                except Exception:
                    pass
            self._kill()

    def _schedule_idle_shutdown(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(WORKER_IDLE_SECONDS, self.close)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _log_tail(self) -> str:
        try:
            return "\n".join(self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        except OSError:
            return ""

    def request(
        self,
        payload: dict[str, Any],
        *,
        cancel_token: CancellationToken | None,
        timeout: float,
    ) -> dict[str, Any]:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if not self.alive:
                self._start()
            assert self._proc is not None and self._proc.stdin is not None
            self._counter += 1
            request_id = f"r{self._counter}"
            try:
                self._proc.stdin.write(json.dumps({**payload, "request_id": request_id}) + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                self._kill()
                raise ModelInferenceError("El worker IA se cerró inesperadamente.", details=str(exc)) from exc

            model_id = str(payload.get("model_id"))
            started = time.monotonic()
            deadline = started + timeout + (0.0 if self._ready else _STARTUP_TIMEOUT_SECONDS)
            try:
                while True:
                    if cancel_token and cancel_token.is_cancelled:
                        self._kill()
                        raise JobCancelledError(f"Inferencia aislada de '{model_id}' cancelada.")
                    if time.monotonic() > deadline:
                        self._kill()
                        raise ModelInferenceError(
                            f"La inferencia aislada de '{model_id}' superó el tiempo límite."
                        )
                    try:
                        line = self._lines.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if line is None:
                        code = self._proc.poll() if self._proc is not None else None
                        tail = self._log_tail()
                        self._kill()
                        if not self._ready and ("--serve" in tail or "request.json" in tail or code == 2):
                            raise _ServerModeUnavailable()
                        raise ModelInferenceError(
                            f"El worker IA terminó inesperadamente (código {code}).", details=tail
                        )
                    if not line.startswith(RESPONSE_PREFIX):
                        continue  # salida informativa de clearvoice/voicefixer
                    message = json.loads(line[len(RESPONSE_PREFIX) :])
                    if message.get("event") == "ready":
                        self._ready = True
                        continue
                    if message.get("request_id") != request_id:
                        continue
                    if not message.get("ok"):
                        raise ModelInferenceError(
                            f"Fallo en el runner aislado de '{model_id}'.",
                            details=str(message.get("error") or ""),
                        )
                    return message
            finally:
                if self.alive:
                    self._schedule_idle_shutdown()


_WORKERS_LOCK = threading.Lock()
_WORKERS: dict[tuple[str, ...], PersistentWorker] = {}
_SERVER_DISABLED: set[tuple[str, ...]] = set()


def _get_persistent_worker(command: list[str], weights_dir: Path) -> PersistentWorker | None:
    if os.environ.get("ZENTRALAUDIO_ONE_SHOT_WORKER") == "1":
        return None
    key = tuple(command)
    with _WORKERS_LOCK:
        if key in _SERVER_DISABLED:
            return None
        worker = _WORKERS.get(key)
        if worker is None:
            data_root = weights_dir.resolve().parent.parent
            worker = PersistentWorker(
                command,
                _portable_worker_environment(weights_dir),
                data_root / "cache" / "ai_worker.log",
            )
            _WORKERS[key] = worker
        return worker


def _disable_persistent_worker(command: list[str]) -> None:
    logger.warning("El worker IA no admite modo persistente; se usará un proceso por solicitud.")
    with _WORKERS_LOCK:
        _SERVER_DISABLED.add(tuple(command))
        worker = _WORKERS.pop(tuple(command), None)
    if worker is not None:
        worker.close()


def shutdown_isolated_workers() -> None:
    """Cierra todos los workers IA persistentes (al salir de la app)."""
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.close()


atexit.register(shutdown_isolated_workers)
