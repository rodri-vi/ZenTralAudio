# -*- mode: python ; coding: utf-8 -*-
"""Especificación PyInstaller para ZenTralAudio.app (macOS arm64 o x86_64).

Se invoca desde scripts/build_macos.py, que prepara el icono .icns y FFmpeg y
fija la arquitectura real de la máquina (ZENTRALAUDIO_TARGET_ARCH).
"""

import importlib.util
import os
import platform
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
TARGET_ARCH = os.environ.get("ZENTRALAUDIO_TARGET_ARCH") or platform.machine()
ICNS = os.environ.get("ZENTRALAUDIO_ICNS") or None
FFMPEG = os.environ.get("ZENTRALAUDIO_FFMPEG")

datas = [
    (str(ROOT / "resources"), "resources"),
    (str(ROOT / "presets" / "restoration"), "presets/restoration"),
    (str(ROOT / "presets" / "environments"), "presets/environments"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "LICENSES_AI.md"), "."),
    # Script del runtime IA aislado (se ejecuta con el intérprete del runtime, no
    # con el de la app, por eso debe existir como archivo real dentro del bundle).
    (str(ROOT / "models" / "isolated_runners" / "worker_entry.py"), "models/isolated_runners"),
]
binaries = []
if FFMPEG:
    binaries.append((FFMPEG, "bin"))
datas += collect_data_files("sounddevice")
datas += collect_data_files("soundfile")

hiddenimports = [
    "soxr",
    "sounddevice",
    "soundfile",
    "torch",
    "torchaudio",
    "scipy.special._cdflib",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "scipy.fft",
    "numba",
    "imageio_ffmpeg",
]
hiddenimports += collect_submodules("soxr")
hiddenimports += collect_submodules("core")
hiddenimports += collect_submodules("engine")
hiddenimports += collect_submodules("models")
hiddenimports += collect_submodules("environments")
hiddenimports += collect_submodules("presets")
hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("scipy.signal")
hiddenimports += collect_submodules("scipy.ndimage")

# DeepFilterNet solo se instala en el entorno de compilación Offline AI.
if importlib.util.find_spec("df") is not None:
    datas += collect_data_files("df")
    hiddenimports += collect_submodules("df")
    hiddenimports += ["libdf"]

excludes = [
    "matplotlib",
    "tkinter",
    "pytest",
    "pytestqt",
    "IPython",
    "jupyter",
    "notebook",
    "PyInstaller",
    "ruff",
    "tests",
    "clearvoice",
    "voicefixer",
    "streamlit",
    "cv2",
    "pandas",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZenTralAudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=TARGET_ARCH,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ZenTralAudio",
)

app = BUNDLE(
    coll,
    name="ZenTralAudio.app",
    icon=ICNS,
    bundle_identifier="com.zentralaudio.desktop",
    info_plist={
        "CFBundleName": "ZenTralAudio",
        "CFBundleDisplayName": "ZenTralAudio",
        "CFBundleIdentifier": "com.zentralaudio.desktop",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.music",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "ZenTralAudio - MIT License",
        # La interfaz usa su propio tema oscuro.
        "NSRequiresAquaSystemAppearance": False,
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Audio",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": [
                    "public.audio",
                    "com.microsoft.waveform-audio",
                    "public.mp3",
                    "org.xiph.flac",
                    "public.aiff-audio",
                    "public.mpeg-4-audio",
                ],
            }
        ],
    },
)
