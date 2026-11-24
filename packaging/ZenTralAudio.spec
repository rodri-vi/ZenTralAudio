# -*- mode: python ; coding: utf-8 -*-
"""Especificación PyInstaller --onedir para generar la versión portable de ZenTralAudio."""

from pathlib import Path
import importlib.util
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "resources"), "resources"),
    (str(ROOT / "presets" / "restoration"), "presets/restoration"),
    (str(ROOT / "presets" / "environments"), "presets/environments"),
    (str(ROOT / "LICENSE"), "."),
]
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

# DeepFilterNet se instala solo en el entorno de compilación de Offline AI.
# La edición ligera continúa siendo compilable sin el paquete opcional.
if importlib.util.find_spec("df") is not None:
    datas += collect_data_files("df")
    hiddenimports += collect_submodules("df")
    hiddenimports += ["libdf"]

excludes = [
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
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
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(ROOT / "packaging" / "windows" / "version_info.txt"),
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
