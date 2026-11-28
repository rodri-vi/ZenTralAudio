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



def _fake_macos_app(root: Path) -> tuple[Path, Path, Path]:
    app_bundle = root / "ZenTralAudio.app"
    macos_dir = app_bundle / "Contents" / "MacOS"
    resources_dir = app_bundle / "Contents" / "Resources"
    macos_dir.mkdir(parents=True)
    (resources_dir / "resources").mkdir(parents=True)
    exe = macos_dir / "ZenTralAudio"
    exe.write_bytes(b"")
    return app_bundle, resources_dir, exe


@pytest.mark.unit
def test_detect_macos_app_bundle_keeps_portable_data_outside_app(tmp_path: Path) -> None:
    portable_dir = tmp_path / "ZenTralAudio_Portable"
    app_bundle, resources_dir, exe = _fake_macos_app(portable_dir)
    # La edición Offline AI incluye models_data junto a la .app => modo portable.
    (portable_dir / "models_data").mkdir()

    paths = AppPaths.detect(
        frozen=True,
        executable_path=exe,
        meipass_path=app_bundle / "Contents" / "Frameworks",
        platform="darwin",
        user_home=tmp_path / "home",
        environ={},
    )

    assert paths.layout == "portable"
    assert paths.app_root == portable_dir.resolve()
    assert paths.bundle_root == resources_dir.resolve()
    assert paths.models_dir == portable_dir.resolve() / "models_data"
    assert paths.models_dir != app_bundle / "Contents" / "MacOS" / "models_data"
    assert paths.resolve_resource("missing.file").parent == resources_dir / "resources"


@pytest.mark.unit
def test_detect_macos_app_uses_standard_user_folders(tmp_path: Path) -> None:
    applications = tmp_path / "Applications"
    app_bundle, resources_dir, exe = _fake_macos_app(applications)
    home = tmp_path / "home"

    paths = AppPaths.detect(
        frozen=True,
        executable_path=exe,
        meipass_path=app_bundle / "Contents" / "Frameworks",
        platform="darwin",
        user_home=home,
        environ={},
    )
    paths.ensure_writable_directories()

    support = home / "Library" / "Application Support" / "ZenTralAudio"
    assert paths.layout == "macos-user"
    assert paths.data_root == support
    assert paths.models_dir == support / "models_data"
    assert paths.config_dir == support / "config"
    assert paths.custom_presets_dir == support / "presets" / "custom"
    assert paths.cache_dir == home / "Library" / "Caches" / "ZenTralAudio"
    assert paths.logs_dir == home / "Library" / "Logs" / "ZenTralAudio"
    assert paths.exports_dir == home / "Music" / "ZenTralAudio"
    assert paths.bundle_root == resources_dir.resolve()
    # Nada modificable dentro del bundle (puede estar en /Applications y firmado).
    for directory in (paths.models_dir, paths.config_dir, paths.logs_dir, paths.cache_dir):
        assert app_bundle not in directory.parents
        assert directory.is_dir()


@pytest.mark.unit
def test_source_mode_can_share_macos_user_folders_and_custom_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    shared = AppPaths.detect(
        frozen=False, platform="darwin", user_home=home, environ={"ZENTRALAUDIO_MACOS_USER_DIRS": "1"}
    )
    assert shared.layout == "macos-user"
    assert shared.models_dir == home / "Library" / "Application Support" / "ZenTralAudio" / "models_data"

    custom = AppPaths.detect(
        frozen=False, platform="darwin", user_home=home, environ={"ZENTRALAUDIO_DATA_DIR": str(tmp_path / "data")}
    )
    assert custom.layout == "custom"
    assert custom.config_dir == (tmp_path / "data").resolve() / "config"

    windows_like = AppPaths.detect(frozen=False, platform="win32", user_home=home, environ={})
    assert windows_like.layout == "portable"
    assert windows_like.models_dir == windows_like.app_root / "models_data"
