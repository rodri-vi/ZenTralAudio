"""Construye una edición de prueba Offline AI con DeepFilterNet3 completamente local."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths
from models.model_manager import ModelManager, compute_sha256

MODEL_ID = "deepfilternet3"


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("Ejecutando:", " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=str(cwd), env=env)
    if proc.returncode:
        raise RuntimeError(f"El comando terminó con código {proc.returncode}: {' '.join(command)}")


def _prepare_model(paths: AppPaths) -> dict[str, Any]:
    manager = ModelManager(paths)
    entry = manager.get_entry(MODEL_ID)
    if entry is None or not entry.download_supported or not entry.files:
        raise RuntimeError("DeepFilterNet3 no tiene una fuente de descarga válida en el manifiesto.")

    if not manager.is_weights_present(MODEL_ID):
        print("Descargando pesos de DeepFilterNet3...", flush=True)
        manager.download_model(
            MODEL_ID,
            user_consented=True,
            progress_cb=lambda fraction, message: print(
                f"[{int(fraction * 100):3d}%] {message}", flush=True
            ),
        )

    model_dir = manager.model_dir(MODEL_ID)
    files = []
    for spec in entry.files:
        file_path = model_dir / spec.filename
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise FileNotFoundError(f"Falta el peso local: {file_path}")
        digest = compute_sha256(file_path)
        if spec.sha256 and digest != spec.sha256.lower():
            raise RuntimeError(f"SHA-256 oficial incorrecto para {file_path.name}.")
        files.append(
            {
                "filename": spec.filename,
                "size_bytes": file_path.stat().st_size,
                "sha256": digest,
                "official_sha256": spec.sha256,
            }
        )

    return {
        "schema_version": 1,
        "edition": "partial",
        "models": [
            {
                "model_id": MODEL_ID,
                "display_name": entry.display_name,
                "source_url": entry.source_url,
                "license_code": entry.license_code,
                "license_weights": entry.license_weights,
                "attribution_required": entry.attribution_required,
                "files": files,
            }
        ],
    }


def _make_zip(portable_dir: Path, output_path: Path) -> None:
    output_path.unlink(missing_ok=True)
    stored = {".pt", ".pth", ".ckpt", ".zip", ".exe", ".dll", ".pyd", ".bin"}
    with zipfile.ZipFile(output_path, "w", allowZip64=True) as archive:
        paths = sorted(item for item in portable_dir.rglob("*") if item.is_file())
        for index, item in enumerate(paths, start=1):
            relative = Path("ZenTralAudio_Portable") / item.relative_to(portable_dir)
            compression = zipfile.ZIP_STORED if item.suffix.lower() in stored else zipfile.ZIP_DEFLATED
            archive.write(item, arcname=str(relative), compress_type=compression, compresslevel=1)
            if index % 300 == 0 or index == len(paths):
                print(f"Empaquetando: {index}/{len(paths)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye ZenTralAudio Offline AI — prueba parcial.")
    parser.add_argument("--bundle-ffmpeg", action="store_true")
    args = parser.parse_args()

    paths = AppPaths.detect()
    paths.ensure_writable_directories()
    inventory = _prepare_model(paths)

    build_command = [sys.executable, str(ROOT / "scripts" / "build_windows.py")]
    if args.bundle_ffmpeg:
        build_command.append("--bundle-ffmpeg")
    build_command.append("--smoke-test")
    _run(build_command)

    dist_dir = ROOT / "dist"
    portable_dir = dist_dir / "ZenTralAudio_Portable"
    target = portable_dir / "models_data" / MODEL_ID
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(paths.models_dir / MODEL_ID, target, dirs_exist_ok=True)

    (portable_dir / "MODELS_OFFLINE_MANIFEST.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(ROOT / "LICENSES_AI.md", portable_dir / "LICENSES_AI.md")
    (portable_dir / "LEEME_OFFLINE_AI_PRUEBA.txt").write_text(
        "ZenTralAudio Offline AI — PRUEBA PARCIAL\n"
        "=========================================\n"
        "Esta edición de prueba incluye DeepFilterNet3 y sus pesos locales.\n"
        "Los otros modelos IA no están incluidos en esta edición de prueba.\n"
        "Extrae la carpeta completa y ejecuta ZenTralAudio.exe.\n"
        "La prueba automática comprueba el arranque portable y una inferencia real de 2 segundos.\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    exe = portable_dir / "ZenTralAudio.exe"
    _run([str(exe), "--check-offline"], cwd=portable_dir, env=env)
    _run(
        [str(exe), "--smoke-test-model", MODEL_ID],
        cwd=portable_dir,
        env=env,
    )

    output_path = dist_dir / "ZenTralAudio-Offline-AI-Partial.zip"
    _make_zip(portable_dir, output_path)
    print(f"Paquete de prueba parcial generado: {output_path}", flush=True)
    print(f"Tamaño: {output_path.stat().st_size / (1024 * 1024):.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
