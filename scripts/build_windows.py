"""Compila ZenTralAudio con PyInstaller (--onedir), valida el .exe y genera el ZIP portable."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths  # noqa: E402

PORTABLE_SUBDIRS = (
    "models_data",
    "ambience_data",
    "cache",
    "projects",
    "exports",
    "config",
    "logs",
    "presets/custom",
    "bin",
)


def _run(cmd: list[str], cwd: Path, *, capture: bool = False) -> None:
    print("Ejecutando:", " ".join(cmd))
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout:
            print(proc.stdout)
    else:
        proc = subprocess.run(cmd, cwd=str(cwd))

    if proc.returncode != 0:
        raise RuntimeError(f"El comando falló con código {proc.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Empaquetador portable Windows de ZenTralAudio")
    parser.add_argument("--bundle-ffmpeg", action="store_true", help="Copia ffmpeg/ffprobe reales a bin/.")
    parser.add_argument("--smoke-test", action="store_true", help="Ejecuta ZenTralAudio.exe --check antes de comprimir.")
    args = parser.parse_args()

    spec_path = ROOT / "packaging" / "ZenTralAudio.spec"
    dist_dir = ROOT / "dist"
    pyinstaller_out = dist_dir / "ZenTralAudio"
    portable_dir = dist_dir / "ZenTralAudio_Portable"
    zip_base = dist_dir / "ZenTralAudio-Windows-Portable"

    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    if pyinstaller_out.exists():
        shutil.rmtree(pyinstaller_out)

    _run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)], cwd=ROOT)

    exe_in = pyinstaller_out / ("ZenTralAudio.exe" if os.name == "nt" else "ZenTralAudio")
    if not exe_in.is_file():
        raise FileNotFoundError(f"No se generó el ejecutable esperado: {exe_in}")

    shutil.move(str(pyinstaller_out), str(portable_dir))

    for sub in PORTABLE_SUBDIRS:
        (portable_dir / sub).mkdir(parents=True, exist_ok=True)

    if args.bundle_ffmpeg:
        app_paths = AppPaths.detect()
        ffmpeg = app_paths.find_ffmpeg()
        ffprobe = app_paths.find_ffprobe()
        if ffmpeg and ffmpeg.is_file():
            if os.name == "nt" and ffmpeg.stat().st_size < 1_000_000:
                raise RuntimeError(
                    f"El ejecutable detectado parece un wrapper/shim y no el binario FFmpeg completo: {ffmpeg}"
                )
            shutil.copy2(ffmpeg, portable_dir / "bin" / ffmpeg.name)
            print(f"Incluido FFmpeg portable: {ffmpeg}")
        if ffprobe and ffprobe.is_file():
            shutil.copy2(ffprobe, portable_dir / "bin" / ffprobe.name)
            print(f"Incluido FFprobe portable: {ffprobe}")

    readme_portable = portable_dir / "LEEME.txt"
    readme_portable.write_text(
        "ZenTralAudio Portable para Windows\n"
        "===================================\n"
        "1. Ejecuta ZenTralAudio.exe (no requiere instalador).\n"
        "2. Puedes copiar esta carpeta completa a otra ubicación o memoria USB.\n"
        "3. Los pesos de modelos IA y recursos ambientales se guardan en models_data/ y ambience_data/.\n",
        encoding="utf-8",
    )

    exe_final = portable_dir / exe_in.name
    if args.smoke_test:
        print(f"Ejecutando smoke test sobre {exe_final} --check ...")
        _run([str(exe_final), "--check"], cwd=portable_dir, capture=True)
        cfg = portable_dir / "config" / "settings.json"
        log = portable_dir / "logs" / "zentralaudio.log"
        if not cfg.is_file() or not log.is_file():
            raise RuntimeError("El smoke test del ejecutable no creó config/settings.json o logs/zentralaudio.log.")
        cfg.unlink(missing_ok=True)
        log.unlink(missing_ok=True)
        print("Smoke test del ejecutable superado correctamente.")

    zip_file = Path(f"{zip_base}.zip")
    zip_file.unlink(missing_ok=True)
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=str(dist_dir), base_dir="ZenTralAudio_Portable")
    print(f"Paquete portable generado: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
