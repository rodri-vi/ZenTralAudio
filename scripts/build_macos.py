"""Construye ZenTralAudio.app para macOS (Apple Silicon arm64 o Intel x86_64).

Uso:
    .venv/bin/python scripts/build_macos.py                 # dist/ZenTralAudio.app
    .venv/bin/python scripts/build_macos.py --smoke-test    # + pruebas del ejecutable
    .venv/bin/python scripts/build_macos.py --install       # + copia a ~/Applications
    .venv/bin/python scripts/build_macos.py --portable      # carpeta portable Offline AI

La .app es autónoma: incluye Python, PySide6, NumPy/SciPy, PyTorch, DeepFilterNet,
FFmpeg y los recursos. Los datos del usuario (modelos, ajustes, logs, caché y el
runtime IA opcional) se guardan en ~/Library/... y nunca dentro del bundle.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILD_DIR = ROOT / "build" / "macos"
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


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("Ejecutando:", " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=str(cwd), env=env)
    if proc.returncode:
        raise RuntimeError(f"El comando falló con código {proc.returncode}: {' '.join(command)}")


def build_icns(svg_path: Path, out_dir: Path) -> Path | None:
    """Rasteriza el SVG del proyecto a todos los tamaños de icono y crea el .icns."""
    if shutil.which("iconutil") is None:
        print("iconutil no disponible; la app usará el icono genérico.", flush=True)
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(str(svg_path))
    iconset = out_dir / "ZenTralAudio.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = size * scale
            image = QImage(px, px, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Margen estándar de los iconos de macOS (~10 %).
            margin = px * 0.1
            renderer.render(painter, QRectF(margin, margin, px - 2 * margin, px - 2 * margin))
            painter.end()
            suffix = "@2x" if scale == 2 else ""
            image.save(str(iconset / f"icon_{size}x{size}{suffix}.png"))
    icns = out_dir / "ZenTralAudio.icns"
    _run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)])
    shutil.rmtree(iconset, ignore_errors=True)
    return icns


def stage_ffmpeg(out_dir: Path) -> Path | None:
    """Copia el FFmpeg estático de imageio-ffmpeg con el nombre que busca la app."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("imageio-ffmpeg no instalado: M4A/AAC no estarán disponibles sin FFmpeg.", flush=True)
        return None
    source = Path(imageio_ffmpeg.get_ffmpeg_exe())
    target_dir = out_dir / "ffmpeg_stage"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "ffmpeg"
    shutil.copy2(source, target)
    target.chmod(0o755)
    return target


def smoke_test(app_bundle: Path) -> None:
    """Ejecuta el binario empaquetado fuera del entorno de desarrollo."""
    exe = app_bundle / "Contents" / "MacOS" / "ZenTralAudio"
    clean_env = {
        "HOME": os.environ["HOME"],
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",  # como al abrir desde Finder
        "LANG": os.environ.get("LANG", "es_ES.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    _run([str(exe), "--check"], cwd=Path("/"), env=clean_env)
    _run([str(exe), "--check-environment"], cwd=Path("/"), env=clean_env)
    dfn = (
        Path.home() / "Library" / "Application Support" / "ZenTralAudio" / "models_data"
        / "deepfilternet3" / "DeepFilterNet3.zip"
    )
    if dfn.is_file():
        _run([str(exe), "--smoke-test-model", "deepfilternet3"], cwd=Path("/"), env=clean_env)
    else:
        print("DeepFilterNet3 no instalado en Application Support: se omite la inferencia.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compila ZenTralAudio.app para macOS.")
    parser.add_argument("--smoke-test", action="store_true", help="Prueba el ejecutable empaquetado.")
    parser.add_argument("--install", action="store_true", help="Copia la app a ~/Applications.")
    parser.add_argument("--portable", action="store_true", help="Crea dist/ZenTralAudio_Portable.")
    parser.add_argument("--bundle-ffmpeg", action="store_true", help="(compatibilidad) FFmpeg siempre se incluye.")
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise RuntimeError("La compilación macOS debe ejecutarse en macOS.")
    arch = platform.machine().lower()
    if arch not in {"arm64", "x86_64"}:
        raise RuntimeError(f"Arquitectura no soportada: {arch}")
    print(f"Compilando para macOS {platform.mac_ver()[0]} ({arch})", flush=True)

    dist_dir = ROOT / "dist"
    app_bundle = dist_dir / "ZenTralAudio.app"
    if app_bundle.exists():
        shutil.rmtree(app_bundle)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    icns = build_icns(ROOT / "resources" / "icons" / "zentralaudio.svg", BUILD_DIR)
    ffmpeg = stage_ffmpeg(BUILD_DIR)

    env = os.environ.copy()
    env["ZENTRALAUDIO_TARGET_ARCH"] = arch
    if icns is not None:
        env["ZENTRALAUDIO_ICNS"] = str(icns)
    if ffmpeg is not None:
        env["ZENTRALAUDIO_FFMPEG"] = str(ffmpeg)
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--workpath",
            str(BUILD_DIR / "work"),
            "--distpath",
            str(dist_dir),
            str(ROOT / "packaging" / "ZenTralAudio-macos.spec"),
        ],
        env=env,
    )

    exe = app_bundle / "Contents" / "MacOS" / "ZenTralAudio"
    if not exe.is_file():
        raise FileNotFoundError(f"PyInstaller no generó la aplicación esperada: {exe}")
    # Firma ad-hoc de todo el bundle (obligatoria en Apple Silicon) y verificación.
    _run(["codesign", "--force", "--deep", "--sign", "-", str(app_bundle)])
    _run(["codesign", "--verify", "--deep", "--strict", str(app_bundle)])
    shutil.rmtree(dist_dir / "ZenTralAudio", ignore_errors=True)  # carpeta intermedia COLLECT

    if args.smoke_test:
        smoke_test(app_bundle)

    if args.portable:
        portable_dir = dist_dir / "ZenTralAudio_Portable"
        if portable_dir.exists():
            shutil.rmtree(portable_dir)
        portable_dir.mkdir(parents=True)
        shutil.copytree(app_bundle, portable_dir / "ZenTralAudio.app", symlinks=True)
        for subdir in PORTABLE_SUBDIRS:
            (portable_dir / subdir).mkdir(parents=True, exist_ok=True)
        if ffmpeg is not None:
            shutil.copy2(ffmpeg, portable_dir / "bin" / "ffmpeg")
        (portable_dir / "LEEME.txt").write_text(
            "ZenTralAudio para macOS (modo portable)\n"
            "=======================================\n"
            "1. Conserva ZenTralAudio.app dentro de esta carpeta.\n"
            "2. Abre ZenTralAudio.app con doble clic.\n"
            "3. Los datos se guardan junto a la app (models_data, config, logs...).\n"
            "4. Si macOS bloquea la primera apertura, usa clic derecho > Abrir.\n",
            encoding="utf-8",
        )
        print(f"Carpeta portable macOS creada: {portable_dir}", flush=True)

    if args.install:
        target = Path.home() / "Applications" / "ZenTralAudio.app"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        _run(["ditto", str(app_bundle), str(target)])
        print(f"Instalada en {target}", flush=True)

    size = subprocess.run(["du", "-sh", str(app_bundle)], capture_output=True, text=True).stdout.split()[0]
    print(f"Aplicación creada: {app_bundle} ({size})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
