"""Construye la carpeta portable ZenTralAudio para macOS Apple Silicon."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _run(command: list[str], *, cwd: Path) -> None:
    print("Ejecutando:", " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=str(cwd))
    if proc.returncode:
        raise RuntimeError(f"El comando falló con código {proc.returncode}: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compila ZenTralAudio para macOS Apple Silicon.")
    parser.add_argument("--bundle-ffmpeg", action="store_true", help="Incluye el binario FFmpeg autónomo.")
    parser.add_argument("--smoke-test", action="store_true", help="Comprueba el arranque del ejecutable.")
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise RuntimeError("La compilación macOS debe ejecutarse en macOS.")
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError(
            f"Se requiere un runner Apple Silicon arm64; arquitectura detectada: {platform.machine()}."
        )

    dist_dir = ROOT / "dist"
    app_bundle = dist_dir / "ZenTralAudio.app"
    portable_dir = dist_dir / "ZenTralAudio_Portable"
    spec_path = ROOT / "packaging" / "ZenTralAudio-macos.spec"

    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    if app_bundle.exists():
        shutil.rmtree(app_bundle)
    dist_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(spec_path),
        ],
        cwd=ROOT,
    )

    app_executable = app_bundle / "Contents" / "MacOS" / "ZenTralAudio"
    if not app_executable.is_file():
        raise FileNotFoundError(f"PyInstaller no generó la aplicación esperada: {app_executable}")

    portable_dir.mkdir(parents=True)
    shutil.move(str(app_bundle), str(portable_dir / "ZenTralAudio.app"))
    for subdir in PORTABLE_SUBDIRS:
        (portable_dir / subdir).mkdir(parents=True, exist_ok=True)

    # imageio-ffmpeg aporta un binario macOS arm64 autónomo, sin depender de Homebrew.
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Falta imageio-ffmpeg. Instala las dependencias de requirements-macos.txt."
        ) from exc
    ffmpeg_source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not ffmpeg_source.is_file():
        raise FileNotFoundError(f"No se encontró el binario FFmpeg: {ffmpeg_source}")
    ffmpeg_target = portable_dir / "bin" / "ffmpeg"
    shutil.copy2(ffmpeg_source, ffmpeg_target)
    ffmpeg_target.chmod(0o755)

    if args.smoke_test:
        final_executable = portable_dir / "ZenTralAudio.app" / "Contents" / "MacOS" / "ZenTralAudio"
        _run([str(final_executable), "--check"], cwd=portable_dir)

    (portable_dir / "LEEME.txt").write_text(
        "ZenTralAudio para macOS Apple Silicon\n"
        "=====================================\n"
        "1. Conserva ZenTralAudio.app dentro de esta carpeta portable.\n"
        "2. Abre ZenTralAudio.app con doble clic.\n"
        "3. Mantén junto a la aplicación models_data, runtimes y bin.\n"
        "4. FFmpeg está incluido para importar y exportar formatos compatibles.\n"
        "5. Si macOS bloquea la primera apertura, usa clic derecho > Abrir.\n",
        encoding="utf-8",
    )
    print(f"Carpeta portable macOS creada: {portable_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
