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
