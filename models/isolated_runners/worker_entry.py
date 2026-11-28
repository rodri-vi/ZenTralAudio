"""Worker autónomo para inferencia ClearVoice y VoiceFixer dentro del portable Offline AI."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _check_runtime() -> int:
    """Comprueba imports críticos sin iniciar inferencia ni acceder a Internet."""
    import importlib.util

    import torch
    import torchaudio
    import torchvision
    import librosa
    import soundfile
    import clearvoice
    import torchlibrosa

    # ``import voicefixer`` descarga ~600 MB de pesos desde Zenodo si no están en
    # ~/.cache/voicefixer. La comprobación solo verifica que el paquete existe.
    if importlib.util.find_spec("voicefixer") is None:
        raise ImportError("voicefixer no está instalado en el runtime")

    print(
        json.dumps(
            {
                "status": "ok",
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "torchvision": torchvision.__version__,
                "clearvoice": getattr(clearvoice, "__version__", "installed"),
                "voicefixer": "installed",
                "librosa": librosa.__version__,
                "soundfile": soundfile.__version__,
                "torchlibrosa": getattr(torchlibrosa, "__version__", "installed"),
            },
            ensure_ascii=True,
        )
    )
    return 0


def _link_or_copy(source: Path, target: Path) -> None:
    """Enlaza el checkpoint (sin duplicar cientos de MB) o lo copia si no es posible."""
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


# --------------------------------------------------------------------------------------
# Modelos cargados (el modo servidor los reutiliza entre solicitudes)
# --------------------------------------------------------------------------------------

# Modelos que en Apple Silicon son más rápidos en CPU que en la GPU (MPS).
# Medido en un M1 con 8 s de audio: FRCRN 15.6 s en CPU frente a 40 s en MPS;
# MossFormerGAN 23 s en MPS (con fallback) frente a 56 s en CPU.
_CPU_PREFERRED_ON_MPS = {"frcrn_se_16k"}
_CLEARVOICE_NAMES = {
    "mossformer2_se_48k": "MossFormer2_SE_48K",
    "mossformergan_se_16k": "MossFormerGAN_SE_16K",
    "frcrn_se_16k": "FRCRN_SE_16K",
}
# Solo se mantiene un modelo cargado: en un Mac de 8 GB dos modelos a la vez
# (más la propia app) provocan swap.
_LOADED: dict[str, Any] = {"key": None, "model": None, "workdir": None}
_ORIGINAL_MPS_AVAILABLE: Any = None


def _configure_torch_device(model_id: str) -> None:
    """Elige el dispositivo de ClearVoice, que usa MPS automáticamente si existe.

    ``PYTORCH_ENABLE_MPS_FALLBACK`` (fijado por el lanzador antes de importar
    torch) permite ejecutar en CPU las operaciones sin kernel MPS, como
    ``aten::angle`` en MossFormerGAN. ``ZENTRALAUDIO_FORCE_CPU=1`` desactiva MPS.
    Se restaura el comportamiento original antes de cada modelo para que un
    proceso persistente pueda alternar entre modelos CPU y MPS.
    """
    global _ORIGINAL_MPS_AVAILABLE
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return
    if _ORIGINAL_MPS_AVAILABLE is None:
        _ORIGINAL_MPS_AVAILABLE = mps.is_available
    mps.is_available = _ORIGINAL_MPS_AVAILABLE  # type: ignore[assignment]
    if os.environ.get("ZENTRALAUDIO_FORCE_CPU") == "1" or model_id in _CPU_PREFERRED_ON_MPS:
        mps.is_available = lambda: False  # type: ignore[assignment]


def _configure_threads() -> None:
    """Usa solo los núcleos de rendimiento en Apple Silicon.

    Medido en un M1 (4 P + 4 E): FRCRN procesa 10 s en 12,2 s con 4 hilos frente
    a 13,8 s con 8; los núcleos de eficiencia frenan las operaciones paralelas.
    """
    if os.environ.get("OMP_NUM_THREADS"):
        return
    try:
        import subprocess

        import torch

        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=2
        )
        perf_cores = int(out.stdout.strip() or 0)
        if perf_cores > 0:
            torch.set_num_threads(perf_cores)
    except Exception:
        pass


def _unload_current() -> None:
    import gc

    workdir = _LOADED.get("workdir")
    _LOADED.update({"key": None, "model": None, "workdir": None})
    gc.collect()
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


def _link_weights(weights_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for item in weights_dir.iterdir():
        if item.is_file() and not item.name.startswith("."):
            _link_or_copy(item, target_dir / item.name)
            linked += 1
    return linked


def _load_clearvoice(model_id: str, weights_dir: Path) -> tuple[Any, Path]:
    key = f"clearvoice:{model_id}:{weights_dir}"
    if _LOADED["key"] == key:
        return _LOADED["model"], Path(_LOADED["workdir"])
    _unload_current()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _configure_torch_device(model_id)
    from clearvoice import ClearVoice  # type: ignore[import-not-found]

    import tempfile

    cv_name = _CLEARVOICE_NAMES[model_id]
    workdir = Path(tempfile.mkdtemp(prefix=f"zentral_{model_id}_"))
    if _link_weights(weights_dir, workdir / "checkpoints" / cv_name) == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        raise FileNotFoundError(f"No hay archivos de pesos locales de {cv_name} en {weights_dir}")
    previous_cwd = os.getcwd()
    os.chdir(str(workdir))
    try:
        model = ClearVoice(task="speech_enhancement", model_names=[cv_name])
    finally:
        os.chdir(previous_cwd)
    _LOADED.update({"key": key, "model": model, "workdir": str(workdir)})
    return model, workdir


def _run_clearvoice(req: dict[str, Any]) -> None:
    import soundfile as sf

    model_id = str(req["model_id"])
    model, workdir = _load_clearvoice(model_id, Path(req["weights_dir"]))

    # ClearVoice lee con pydub, que solo entiende WAV PCM entero sin FFmpeg, y
    # escala dividiendo por 2**15 salvo que detecte valores mayores. Se le
    # entrega por tanto PCM de 16 bits (su formato nativo) en lugar de float32.
    source, source_sr = sf.read(req["input_wav"], dtype="float32", always_2d=True)
    pcm_input = Path(req["output_wav"]).parent / "input_pcm16.wav"
    sf.write(str(pcm_input), source.clip(-1.0, 1.0), source_sr, subtype="PCM_16")

    previous_cwd = os.getcwd()
    os.chdir(str(workdir))
    try:
        output_wav = model(input_path=str(pcm_input), online_write=False)
    finally:
        os.chdir(previous_cwd)
    sample_rate = 48000 if model_id == "mossformer2_se_48k" else 16000
    sf.write(
        req["output_wav"],
        output_wav.T if output_wav.ndim == 2 else output_wav,
        sample_rate,
        subtype="FLOAT",
    )


def _load_voicefixer(weights_dir: Path) -> Any:
    key = f"voicefixer:{weights_dir}"
    if _LOADED["key"] == key:
        return _LOADED["model"]
    _unload_current()
    vf_ckpt = weights_dir / "vf.ckpt"
    vocoder_ckpt = weights_dir / "model.ckpt-1490000_trimed.pt"
    if not vf_ckpt.is_file() or not vocoder_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoints de VoiceFixer no encontrados en {weights_dir}")

    # Path.home() se redirige a cache/offline_worker_home desde el lanzador.
    cache_root = Path.home() / ".cache" / "voicefixer"
    am_dir = cache_root / "analysis_module" / "checkpoints"
    voc_dir = cache_root / "synthesis_module" / "44100"
    am_dir.mkdir(parents=True, exist_ok=True)
    voc_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(vf_ckpt, am_dir / "vf.ckpt")
    _link_or_copy(vocoder_ckpt, voc_dir / "model.ckpt-1490000_trimed.pt")

    # Importar solo después de colocar los pesos locales: el paquete descarga sus
    # checkpoints al importarse si no los encuentra en ~/.cache/voicefixer.
    from voicefixer import VoiceFixer  # type: ignore[import-not-found]

    model = VoiceFixer()
    _LOADED.update({"key": key, "model": model, "workdir": None})
    return model


def _run_voicefixer(req: dict[str, Any]) -> None:
    model = _load_voicefixer(Path(req["weights_dir"]))
    use_cuda = req.get("device") == "cuda"
    mode = int((req.get("params") or {}).get("model_vf_mode", 0))
    model.restore(input=req["input_wav"], output=req["output_wav"], cuda=use_cuda, mode=mode)


def _handle(req: dict[str, Any]) -> None:
    model_id = str(req["model_id"])
    if model_id == "voicefixer":
        _run_voicefixer(req)
    elif model_id in _CLEARVOICE_NAMES:
        _run_clearvoice(req)
    else:
        raise ValueError(f"Modelo no soportado en worker_entry: {model_id}")


_RESPONSE_PREFIX = "@@ZENTRAL "


def _respond(message: dict[str, Any]) -> None:
    stream = sys.__stdout__ or sys.stdout
    stream.write(_RESPONSE_PREFIX + json.dumps(message, ensure_ascii=True) + "\n")
    stream.flush()


def _serve() -> int:
    """Modo persistente: una solicitud JSON por línea en stdin, respuestas en stdout."""
    import traceback

    _configure_threads()
    _respond({"event": "ready", "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _respond({"ok": False, "error": f"JSON no válido: {exc}"})
            continue
        if req.get("cmd") == "exit":
            break
        request_id = req.get("request_id")
        try:
            _handle(req)
            _respond({"request_id": request_id, "ok": True})
        except Exception:
            _respond({"request_id": request_id, "ok": False, "error": traceback.format_exc()[-4000:]})
    _unload_current()
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--check-runtime":
        return _check_runtime()
    if len(sys.argv) == 2 and sys.argv[1] == "--serve":
        return _serve()
    if len(sys.argv) < 2:
        print("Uso: ZenTralModelWorker <request.json> | --serve | --check-runtime", file=sys.stderr)
        return 2

    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    _configure_threads()
    try:
        _handle(req)
    finally:
        _unload_current()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
