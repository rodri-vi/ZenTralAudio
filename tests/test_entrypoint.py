"""Pruebas del punto de entrada y modo de comprobación portable."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.configuration import ConfigManager
from core.paths import AppPaths
from main import run_startup_check


def test_startup_check_creates_config_and_log(
    portable_paths: AppPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_setup_logging(logs_dir: Path, *, debug: bool = False) -> Path:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "zentralaudio.log"
        log_path.write_text("startup check\n", encoding="utf-8")
        return log_path

    monkeypatch.setattr("main.setup_logging", fake_setup_logging)

    assert run_startup_check(portable_paths) == 0
    assert (portable_paths.config_dir / "settings.json").is_file()
    assert (portable_paths.logs_dir / "zentralaudio.log").is_file()
    assert ConfigManager(portable_paths.config_dir).load().schema_version == 1


def test_environment_report_lists_models_and_kernels(
    portable_paths: AppPaths,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from main import run_environment_report

    assert run_environment_report(portable_paths) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["failures"] == []
    assert report["numba_kernels"] in (True, False)
    assert "deepfilternet3" in report["models"]
    assert report["paths"]["models_dir"] == str(portable_paths.models_dir)


def test_file_open_events_are_queued_until_window_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS entrega 'Abrir con...' como QFileOpenEvent, a veces antes de crear la ventana."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QFileOpenEvent
    from PySide6.QtWidgets import QApplication

    from main import _FileOpenFilter

    app = QApplication.instance() or QApplication([])
    file_filter = _FileOpenFilter(app)
    opened: list[str] = []
    event = QFileOpenEvent(QUrl.fromLocalFile("/tmp/voz.wav"))
    assert file_filter.eventFilter(app, event) is True
    file_filter.attach(opened.append)
    assert opened == ["/tmp/voz.wav"]
    file_filter.eventFilter(app, QFileOpenEvent(QUrl.fromLocalFile("/tmp/otra.mp3")))
    assert opened[-1] == "/tmp/otra.mp3"
