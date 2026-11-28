"""Pruebas unitarias de detección de rutas portables en Python y modo PyInstaller."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.paths import AppPaths


@pytest.mark.unit
def test_detect_source_mode() -> None:
    paths = AppPaths.detect(frozen=False)
    assert paths.is_frozen is False
    assert (paths.app_root / "core" / "paths.py").is_file()


@pytest.mark.unit
def test_detect_frozen_portable_mode(tmp_path: Path) -> None:
    usb_dir = tmp_path / "USB_Drive" / "ZenTralAudio_Portable"
    usb_dir.mkdir(parents=True)
    exe = usb_dir / "ZenTralAudio.exe"
    exe.write_bytes(b"")
    internal = usb_dir / "_internal"
    internal.mkdir()

    paths = AppPaths.detect(frozen=True, executable_path=exe, meipass_path=internal)
    paths.ensure_writable_directories()

    assert paths.is_frozen is True
    assert paths.app_root == usb_dir.resolve()
    assert paths.models_dir == usb_dir.resolve() / "models_data"
    assert paths.logs_dir.is_dir()
    assert paths.config_dir.is_dir()
    assert paths.exports_dir.is_dir()



@pytest.mark.unit
def test_detect_macos_app_bundle_keeps_portable_data_outside_app(tmp_path: Path) -> None:
    portable_dir = tmp_path / "ZenTralAudio_Portable"
    app_bundle = portable_dir / "ZenTralAudio.app"
    macos_dir = app_bundle / "Contents" / "MacOS"
    resources_dir = app_bundle / "Contents" / "Resources"
    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)
    (resources_dir / "resources").mkdir()
    exe = macos_dir / "ZenTralAudio"
    exe.write_bytes(b"")

    paths = AppPaths.detect(
        frozen=True,
        executable_path=exe,
        meipass_path=app_bundle / "Contents" / "Frameworks",
    )

    assert paths.app_root == portable_dir.resolve()
    assert paths.bundle_root == resources_dir.resolve()
    assert paths.models_dir == portable_dir.resolve() / "models_data"
    assert paths.models_dir != app_bundle / "Contents" / "MacOS" / "models_data"
    assert paths.resolve_resource("missing.file").parent == resources_dir / "resources"
