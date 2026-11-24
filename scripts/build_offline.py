"""Construye ZenTralAudio Offline AI con pesos locales y worker autocontenido."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths
from models.model_manager import ModelManager, compute_sha256

MODEL_IDS = (
    "deepfilternet3",
    "mossformer2_se_48k",
    "mossformergan_se_16k",
    "frcrn_se_16k",
    "voicefixer",
)
WORKER_MODULES = (
    "clearvoice",
    "voicefixer",
    "torchlibrosa",
    "rotary_embedding_torch",
    "yamlargparse",
    "python_speech_features",
    "scenedetect",
    "soundfile",
    "librosa",
)


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("Ejecutando:", " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=str(cwd), env=env)
    if proc.returncode:
        raise RuntimeError(f"El comando terminó con código {proc.returncode}: {' '.join(command)}")


def _download_all_models(paths: AppPaths) -> dict[str, Any]:
    manager = ModelManager(paths)
    inventory: dict[str, Any] = {"schema_version": 1, "models": []}
    for model_id in MODEL_IDS:
        entry = manager.get_entry(model_id)
        if entry is None or not entry.download_supported or not entry.files:
            raise RuntimeError(f"No hay manifiesto de descarga verificable para {model_id}.")

        if not manager.is_weights_present(model_id):
            print(f"Descargando pesos de {entry.display_name}...", flush=True)
            last_reported = [-5]

            def progress(fraction: float, message: str) -> None:
                percent = int(max(0.0, min(1.0, fraction)) * 100)
                if percent >= last_reported[0] + 5 or percent == 100:
                    print(f"[{percent:3d}%] {message}", flush=True)
                    last_reported[0] = percent

            manager.download_model(model_id, user_consented=True, progress_cb=progress)
        else:
            print(f"Pesos ya presentes: {entry.display_name}", flush=True)

        files = []
        model_dir = manager.model_dir(model_id)
        for spec in entry.files:
            file_path = model_dir / spec.filename
            if not file_path.is_file() or file_path.stat().st_size <= 0:
                raise FileNotFoundError(f"Falta el peso esperado de {model_id}: {file_path}")
            actual = compute_sha256(file_path)
            if spec.sha256 and actual != spec.sha256.lower():
                raise RuntimeError(f"SHA-256 oficial incorrecto para {file_path.name}: {actual}")
            files.append(
                {
                    "filename": spec.filename,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": actual,
                    "official_sha256": spec.sha256,
                }
            )
        inventory["models"].append(
            {
                "model_id": model_id,
                "display_name": entry.display_name,
                "source_url": entry.source_url,
                "license_code": entry.license_code,
                "license_weights": entry.license_weights,
                "attribution_required": entry.attribution_required,
                "files": files,
            }
        )
    return inventory


def _build_worker(dist_dir: Path) -> Path:
    for module in WORKER_MODULES:
        try:
            __import__(module)
        except Exception as exc:
            raise RuntimeError(f"Falta el módulo requerido para el worker offline '{module}': {exc}") from exc

    import torch
    import torchaudio
    import torchvision

    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"Versión inesperada de Torch: {torch.__version__}")
    if torchaudio.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"Versión inesperada de Torchaudio: {torchaudio.__version__}")
    if torchvision.__version__.split("+")[0] != "0.20.1":
        raise RuntimeError(f"Versión inesperada de Torchvision: {torchvision.__version__}")

    worker_dist = dist_dir / "offline_runtimes"
    worker_work = ROOT / "build" / "offline_worker"
    worker_spec = ROOT / "build" / "offline_worker_specs"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name",
        "ZenTralModelWorker",
        "--distpath",
        str(worker_dist),
        "--workpath",
        str(worker_work),
        "--specpath",
        str(worker_spec),
        "--paths",
        str(ROOT),
    ]
    for module in WORKER_MODULES:
        command.extend(["--collect-all", module])
    command.append(str(ROOT / "models" / "isolated_runners" / "worker_entry.py"))
    _run(command)

    worker_dir = worker_dist / "ZenTralModelWorker"
    worker_exe = worker_dir / "ZenTralModelWorker.exe"
    if not worker_exe.is_file():
        raise FileNotFoundError(f"PyInstaller no generó el worker portable: {worker_exe}")
    return worker_dir


def _copy_model_assets(paths: AppPaths, portable_dir: Path) -> None:
    target_root = portable_dir / "models_data"
    target_root.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_IDS:
        source = paths.models_dir / model_id
        target = target_root / model_id
        if not source.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de pesos descargados: {source}")
        shutil.copytree(source, target, dirs_exist_ok=True)


def _check_relocated_portable(portable_dir: Path) -> None:
    worker = portable_dir / "runtimes" / "ZenTralModelWorker" / "ZenTralModelWorker.exe"
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    _run([str(worker), "--check-runtime"], cwd=portable_dir, env=env)

    # La copia a otra ruta comprueba que ningún .venv conserva rutas de la máquina de CI.
    with tempfile.TemporaryDirectory(prefix="zentral_offline_relocated_") as tmp:
        relocated = Path(tmp) / "ZenTralAudio_Portable"
        shutil.copytree(portable_dir, relocated)
        exe = relocated / "ZenTralAudio.exe"
        _run([str(exe), "--check-offline"], cwd=relocated, env=env)


def _create_portable_zip(portable_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    store_suffixes = {".pt", ".pth", ".ckpt", ".zip", ".exe", ".dll", ".pyd", ".bin"}
    with zipfile.ZipFile(output_path, "w", allowZip64=True) as archive:
        files = sorted(path for path in portable_dir.rglob("*") if path.is_file())
        total = len(files)
        for index, path in enumerate(files, start=1):
            relative = Path("ZenTralAudio_Portable") / path.relative_to(portable_dir)
            compression = zipfile.ZIP_STORED if path.suffix.lower() in store_suffixes else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=str(relative), compress_type=compression, compresslevel=1)
            if index % 250 == 0 or index == total:
                print(f"Empaquetando archivos: {index}/{total}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye la edición ZenTralAudio Offline AI.")
    parser.add_argument("--bundle-ffmpeg", action="store_true")
    args = parser.parse_args()

    paths = AppPaths.detect()
    paths.ensure_writable_directories()
    inventory = _download_all_models(paths)

    dist_dir = ROOT / "dist"
    command = [sys.executable, str(ROOT / "scripts" / "build_windows.py")]
    if args.bundle_ffmpeg:
        command.append("--bundle-ffmpeg")
    command.append("--smoke-test")
    _run(command)

    portable_dir = dist_dir / "ZenTralAudio_Portable"
    if not (portable_dir / "ZenTralAudio.exe").is_file():
        raise FileNotFoundError("Falta ZenTralAudio.exe en el portable base.")

    worker_dir = _build_worker(dist_dir)
    runtime_target = portable_dir / "runtimes" / "ZenTralModelWorker"
    runtime_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(worker_dir, runtime_target, dirs_exist_ok=True)

    _copy_model_assets(paths, portable_dir)
    shutil.copy2(ROOT / "LICENSES_AI.md", portable_dir / "LICENSES_AI.md")
    (portable_dir / "MODELS_OFFLINE_MANIFEST.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (portable_dir / "LEEME_OFFLINE_AI.txt").write_text(
        "ZenTralAudio Offline AI para Windows\n"
        "=====================================\n"
        "1. Extrae toda la carpeta ZenTralAudio_Portable.\n"
        "2. Ejecuta ZenTralAudio.exe.\n"
        "3. Los cinco modelos incluidos usan pesos locales y no descargan modelos durante inferencia.\n"
        "4. Conserva juntas las carpetas _internal, models_data, runtimes, bin y el ejecutable.\n"
        "5. Reserva al menos 5 GB libres para extraer y usar esta edición.\n",
        encoding="utf-8",
    )

    _check_relocated_portable(portable_dir)

    # Evita confundir el ZIP Offline AI con el ZIP ligero de la edición base.
    (dist_dir / "ZenTralAudio-Windows-Portable.zip").unlink(missing_ok=True)
    output_path = dist_dir / "ZenTralAudio-Offline-AI.zip"
    _create_portable_zip(portable_dir, output_path)
    print(f"Paquete Offline AI generado: {output_path}", flush=True)
    print(f"Tamaño: {output_path.stat().st_size / (1024 * 1024):.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
