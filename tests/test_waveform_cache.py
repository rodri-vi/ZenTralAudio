"""La forma de onda se dibuja una vez y el cursor se repinta sin recalcularla."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_waveform_pixmap_is_cached_between_cursor_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.waveform_widget import WaveformWidget

    app = QApplication.instance() or QApplication([])
    widget = WaveformWidget()
    widget.resize(800, 160)
    widget.show()
    samples = (np.random.default_rng(0).standard_normal((2, 48000 * 30)) * 0.3).astype(np.float32)
    widget.set_audio(samples, 48000, "test")

    builds = {"n": 0}
    original_build = widget._build_wave_cache

    def counting_build():
        builds["n"] += 1
        return original_build()

    monkeypatch.setattr(widget, "_build_wave_cache", counting_build)
    for second in range(20):
        widget.set_position(float(second))
        widget.repaint()
    app.processEvents()
    assert builds["n"] == 1

    widget.resize(600, 160)
    widget.repaint()
    assert builds["n"] == 2
    image = widget.grab().toImage()
    assert not image.isNull()
