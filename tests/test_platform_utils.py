"""Integración con el sistema operativo (Finder/Explorador) sin comandos de Linux en macOS."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import platform_utils


@pytest.mark.unit
def test_open_in_file_manager_uses_native_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class FakeDesktop:
        @staticmethod
        def openUrl(_url) -> bool:  # noqa: N802
            return False

    import PySide6.QtGui as qtgui

    monkeypatch.setattr(qtgui, "QDesktopServices", FakeDesktop)
    monkeypatch.setattr(platform_utils.subprocess, "Popen", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_utils, "IS_MACOS", True)

    target = tmp_path / "carpeta nueva"
    assert platform_utils.open_in_file_manager(target)
    assert target.is_dir()
    assert calls == [["open", str(target)]]


@pytest.mark.unit
def test_compute_device_label_is_never_empty() -> None:
    assert platform_utils.compute_device_label()
