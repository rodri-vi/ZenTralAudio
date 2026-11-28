"""Pruebas de los divisores redimensionables y la distribución inicial de la interfaz."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_restoration_tab_has_resizable_work_areas_and_waveform_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplitter

    from app.restoration_tab import RestorationTab

    app = QApplication.instance() or QApplication([])
    tab = RestorationTab()

    assert isinstance(tab._workspace_splitter, QSplitter)
    assert tab._workspace_splitter.orientation() == Qt.Orientation.Vertical
    assert tab._workspace_splitter.count() == 3
    assert tab._view_mode_combo.currentData() == "waveform"
    assert tab.advanced_panel.isHidden()

    tab._adv_toggle_btn.setChecked(True)
    assert not tab.advanced_panel.isHidden()
    assert tab.advanced_panel.minimumHeight() >= 200

    tab.close()
    app.processEvents()


@pytest.mark.unit
def test_environment_tab_has_resizable_header_acoustics_and_mixer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplitter

    from app.environment_tab import EnvironmentTab

    app = QApplication.instance() or QApplication([])
    tab = EnvironmentTab()

    assert isinstance(tab._workspace_splitter, QSplitter)
    assert tab._workspace_splitter.orientation() == Qt.Orientation.Vertical
    assert tab._workspace_splitter.count() == 2
    assert isinstance(tab._panels_splitter, QSplitter)
    assert tab._panels_splitter.orientation() == Qt.Orientation.Horizontal
    assert tab._panels_splitter.count() == 2

    tab.close()
    app.processEvents()
