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


def _run_voicefixer(req: dict[str, Any]) -> None:
    weights_dir = Path(req["weights_dir"])
    vf_ckpt = weights_dir / "vf.ckpt"
    vocoder_ckpt = weights_dir / "model.ckpt-1490000_trimed.pt"
    if not vf_ckpt.is_file() or not vocoder_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoints de VoiceFixer no encontrados en {weights_dir}")

    # Path.home() se redirige a cache/offline_worker_home desde el launcher portable.
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

    use_cuda = req.get("device") == "cuda"
    mode = int((req.get("params") or {}).get("model_vf_mode", 0))
    vf = VoiceFixer()
    vf.restore(input=req["input_wav"], output=req["output_wav"], cuda=use_cuda, mode=mode)


# Modelos que en Apple Silicon son más rápidos en CPU que en la GPU (MPS).
# Medido en un M1 con 8 s de audio: FRCRN 15.6 s en CPU frente a 40 s en MPS;
# MossFormerGAN 23 s en MPS (con fallback) frente a 56 s en CPU.
_CPU_PREFERRED_ON_MPS = {"frcrn_se_16k"}


def _configure_torch_device(model_id: str) -> None:
    """Elige el dispositivo de ClearVoice, que usa MPS automáticamente si existe.

    ``PYTORCH_ENABLE_MPS_FALLBACK`` (fijado por el lanzador antes de importar
    torch) permite ejecutar en CPU las operaciones sin kernel MPS, como
    ``aten::angle`` en MossFormerGAN. ``ZENTRALAUDIO_FORCE_CPU=1`` desactiva MPS.
    """
    import torch

    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return
    if os.environ.get("ZENTRALAUDIO_FORCE_CPU") == "1" or model_id in _CPU_PREFERRED_ON_MPS:
        mps.is_available = lambda: False  # type: ignore[assignment]


def _run_clearvoice(req: dict[str, Any]) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _configure_torch_device(str(req["model_id"]))
    import soundfile as sf
    from clearvoice import ClearVoice  # type: ignore[import-not-found]

    model_map = {
        "mossformer2_se_48k": "MossFormer2_SE_48K",
        "mossformergan_se_16k": "MossFormerGAN_SE_16K",
        "frcrn_se_16k": "FRCRN_SE_16K",
    }
    cv_name = model_map[req["model_id"]]
    weights_dir = Path(req["weights_dir"])
    work_root = Path(req["output_wav"]).parent
    work_ckpt_dir = work_root / "checkpoints" / cv_name
    work_ckpt_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in weights_dir.iterdir():
        if item.is_file() and not item.name.startswith("."):
            target = work_ckpt_dir / item.name
            if not target.exists():
                shutil.copy2(item, target)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No hay archivos de pesos locales de {cv_name} en {weights_dir}")

    # ClearVoice lee con pydub, que solo entiende WAV PCM entero sin FFmpeg, y
    # escala dividiendo por 2**15 salvo que detecte valores mayores. Se le
    # entrega por tanto PCM de 16 bits (su formato nativo) en lugar de float32.
    source, source_sr = sf.read(req["input_wav"], dtype="float32", always_2d=True)
    pcm_input = work_root / "input_pcm16.wav"
    sf.write(str(pcm_input), source.clip(-1.0, 1.0), source_sr, subtype="PCM_16")

    previous_cwd = os.getcwd()
    os.chdir(str(work_root))
    try:
        cv = ClearVoice(task="speech_enhancement", model_names=[cv_name])
        output_wav = cv(input_path=str(pcm_input), online_write=False)
        sample_rate = 48000 if cv_name == "MossFormer2_SE_48K" else 16000
        sf.write(
            req["output_wav"],
            output_wav.T if output_wav.ndim == 2 else output_wav,
            sample_rate,
            subtype="FLOAT",
        )
    finally:
        os.chdir(previous_cwd)


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--check-runtime":
        return _check_runtime()
    if len(sys.argv) < 2:
        print("Uso: ZenTralModelWorker.exe <request.json> | --check-runtime", file=sys.stderr)
        return 2

    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    model_id = str(req["model_id"])
    if model_id == "voicefixer":
        _run_voicefixer(req)
    elif model_id in ("mossformer2_se_48k", "mossformergan_se_16k", "frcrn_se_16k"):
        _run_clearvoice(req)
    else:
        raise ValueError(f"Modelo no soportado en worker_entry: {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
