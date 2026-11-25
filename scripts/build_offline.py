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
    worker_name = "ZenTralModelWorker.exe" if os.name == "nt" else "ZenTralModelWorker"
    worker_exe = worker_dir / worker_name
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
    worker_name = "ZenTralModelWorker.exe" if os.name == "nt" else "ZenTralModelWorker"
    worker = portable_dir / "runtimes" / "ZenTralModelWorker" / worker_name
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
        exe = (
            relocated / "ZenTralAudio.exe"
            if os.name == "nt"
            else relocated / "ZenTralAudio.app" / "Contents" / "MacOS" / "ZenTralAudio"
        )
        _run([str(exe), "--check-offline"], cwd=relocated, env=env)


def _create_portable_zip(portable_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    if sys.platform == "darwin":
        # ditto conserva la estructura del .app, permisos y enlaces simbólicos de macOS.
        _run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(portable_dir), str(output_path)]
        )
        return
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



def _split_release_asset_if_needed(output_path: Path, *, max_part_bytes: int = 1_800_000_000) -> list[Path]:
    """Divide el ZIP en partes inferiores a 2 GB para subirlas como assets de GitHub Release."""
    if output_path.stat().st_size <= max_part_bytes:
        return [output_path]

    parts: list[Path] = []
    part_size = 0
    part_path: Path | None = None
    part_index = 0
    with output_path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            if part_path is None or part_size + len(chunk) > max_part_bytes:
                part_index += 1
                part_size = 0
                part_path = output_path.with_name(f"{output_path.name}.part{part_index:02d}")
                part_path.unlink(missing_ok=True)
                parts.append(part_path)
            with part_path.open("ab") as target:
                target.write(chunk)
            part_size += len(chunk)

    output_path.unlink()
    print(
        "ZIP dividido para publicación como Release: "
        + ", ".join(f"{p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)" for p in parts),
        flush=True,
    )
    return parts


def _write_release_notes(dist_dir: Path, assets: list[Path]) -> Path:
    """Prepara instrucciones de descarga, unión opcional de partes y extracción."""
    is_macos = sys.platform == "darwin"
    platform_label = "macOS Apple Silicon" if is_macos else "Windows x64"
    zip_name = (
        "ZenTralAudio-Offline-AI-macOS-Apple-Silicon.zip"
        if is_macos else "ZenTralAudio-Offline-AI.zip"
    )
    notes = [
        f"# ZenTralAudio Offline AI — {platform_label}",
        "",
        "Edición portable con cinco modelos: DeepFilterNet3, MossFormer2_SE_48K, "
        "MossFormerGAN_SE_16K, FRCRN_SE_16K y VoiceFixer.",
        "",
        "## Instalación",
        "",
    ]
    if len(assets) == 1 and assets[0].suffix.lower() == ".zip":
        notes.extend([
            f"1. Descarga {zip_name} desde esta Release.",
            "2. Extrae el ZIP completo en una carpeta local.",
            (
                "3. Abre la carpeta ZenTralAudio_Portable y haz doble clic en ZenTralAudio.app."
                if is_macos else
                "3. Abre ZenTralAudio_Portable y ejecuta ZenTralAudio.exe."
            ),
        ])
    else:
        names = [p.name for p in assets]
        joined_names = " ".join(f'"{name}"' for name in names)
        if is_macos:
            command = f"cat {joined_names} > \"{zip_name}\""
            notes.extend([
                "1. Descarga TODAS las partes en la misma carpeta.",
                "2. Abre Terminal en esa carpeta y ejecuta:",
                "",
                f"    {command}",
                "",
                "3. Extrae el ZIP resultante y abre ZenTralAudio.app.",
            ])
        else:
            command = " + ".join(names) + " ZenTralAudio-Offline-AI.zip"
            notes.extend([
                "1. Descarga TODAS las partes adjuntas a esta Release en la misma carpeta.",
                "2. Abre CMD en esa carpeta y ejecuta este comando para reconstruir el ZIP:",
                "",
                "    copy /b " + command,
                "",
                "3. Extrae ZenTralAudio-Offline-AI.zip.",
                "4. Abre ZenTralAudio_Portable y ejecuta ZenTralAudio.exe.",
            ])
    notes.extend([
        "",
        "## Requisitos y notas",
        "",
        (
            "- macOS 13 o posterior y Apple Silicon (M1 o posterior)."
            if is_macos else "- Windows 10/11 de 64 bits."
        ),
        "- Reserva al menos 5 GB libres para extraer y trabajar con la carpeta portable.",
        "- Conserva juntas la aplicación y las carpetas models_data, runtimes y bin.",
        "- La inferencia se configura inicialmente en CPU para priorizar compatibilidad.",
        "- GTCRN no está incluido porque sus pesos y runtime todavía no están verificados.",
        "- Consulta LICENSES_AI.md para licencias y atribuciones, incluida CC-BY-4.0 para VoiceFixer.",
        "",
        "La comprobación automática valida los pesos, las importaciones del runtime IA y el arranque portable desde otra ruta.",
    ])
    release_notes = dist_dir / "RELEASE_NOTES_OFFLINE_AI.md"
    release_notes.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return release_notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye la edición ZenTralAudio Offline AI.")
    parser.add_argument("--bundle-ffmpeg", action="store_true")
    args = parser.parse_args()

    paths = AppPaths.detect()
    paths.ensure_writable_directories()
    inventory = _download_all_models(paths)

    dist_dir = ROOT / "dist"
    builder = "build_windows.py" if os.name == "nt" else "build_macos.py"
    command = [sys.executable, str(ROOT / "scripts" / builder)]
    if args.bundle_ffmpeg:
        command.append("--bundle-ffmpeg")
    command.append("--smoke-test")
    _run(command)

    portable_dir = dist_dir / "ZenTralAudio_Portable"
    app_executable = (
        portable_dir / "ZenTralAudio.exe"
        if os.name == "nt"
        else portable_dir / "ZenTralAudio.app" / "Contents" / "MacOS" / "ZenTralAudio"
    )
    if not app_executable.is_file():
        raise FileNotFoundError(f"No se encontró el ejecutable principal: {app_executable}")

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
    readme_lines = (
        [
            "ZenTralAudio Offline AI para macOS Apple Silicon",
            "================================================",
            "1. Extrae la carpeta ZenTralAudio_Portable.",
            "2. Abre ZenTralAudio.app.",
            "3. Los cinco modelos incluidos usan pesos locales.",
            "4. Conserva la aplicación junto a models_data, runtimes y bin.",
            "5. Si macOS bloquea la primera apertura, usa clic derecho > Abrir.",
        ]
        if sys.platform == "darwin"
        else [
            "ZenTralAudio Offline AI para Windows",
            "=====================================",
            "1. Extrae toda la carpeta ZenTralAudio_Portable.",
            "2. Ejecuta ZenTralAudio.exe.",
            "3. Los cinco modelos incluidos usan pesos locales y no descargan modelos durante inferencia.",
            "4. Conserva juntas las carpetas _internal, models_data, runtimes, bin y el ejecutable.",
            "5. Reserva al menos 5 GB libres para extraer y usar esta edición.",
        ]
    )
    (portable_dir / "LEEME_OFFLINE_AI.txt").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    _check_relocated_portable(portable_dir)

    # Evita confundir el paquete Offline AI con el ZIP de la edición ligera.
    if os.name == "nt":
        (dist_dir / "ZenTralAudio-Windows-Portable.zip").unlink(missing_ok=True)
        output_path = dist_dir / "ZenTralAudio-Offline-AI.zip"
    else:
        output_path = dist_dir / "ZenTralAudio-Offline-AI-macOS-Apple-Silicon.zip"
    _create_portable_zip(portable_dir, output_path)
    assets = _split_release_asset_if_needed(output_path)
    _write_release_notes(dist_dir, assets)
    for asset in assets:
        print(f"Asset Offline AI preparado: {asset.name} ({asset.stat().st_size / (1024 * 1024):.1f} MB)", flush=True)
    print(f"Total publicado en {len(assets)} archivo(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
