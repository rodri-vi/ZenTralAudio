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
    import torch
    import torchaudio
    import torchvision
    import librosa
    import soundfile
    import clearvoice
    import voicefixer
    import torchlibrosa

    print(
        json.dumps(
            {
                "status": "ok",
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "torchvision": torchvision.__version__,
                "clearvoice": getattr(clearvoice, "__version__", "installed"),
                "voicefixer": getattr(voicefixer, "__version__", "installed"),
                "librosa": librosa.__version__,
                "soundfile": soundfile.__version__,
                "torchlibrosa": getattr(torchlibrosa, "__version__", "installed"),
            },
            ensure_ascii=True,
        )
    )
    return 0


def _run_voicefixer(req: dict[str, Any]) -> None:
    from voicefixer import VoiceFixer  # type: ignore[import-not-found]

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
    if not (am_dir / "vf.ckpt").is_file():
        shutil.copy2(vf_ckpt, am_dir / "vf.ckpt")
    if not (voc_dir / "model.ckpt-1490000_trimed.pt").is_file():
        shutil.copy2(vocoder_ckpt, voc_dir / "model.ckpt-1490000_trimed.pt")

    use_cuda = req.get("device") == "cuda"
    mode = int((req.get("params") or {}).get("model_vf_mode", 0))
    vf = VoiceFixer()
    vf.restore(input=req["input_wav"], output=req["output_wav"], cuda=use_cuda, mode=mode)


def _run_clearvoice(req: dict[str, Any]) -> None:
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

    previous_cwd = os.getcwd()
    os.chdir(str(work_root))
    try:
        cv = ClearVoice(task="speech_enhancement", model_names=[cv_name])
        output_wav = cv(input_path=req["input_wav"], online_write=False)
        sf.read(req["input_wav"], dtype="float32", always_2d=True)
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
