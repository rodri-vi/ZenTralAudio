from __future__ import annotations

import numpy as np

from engine.playback_utils import calculate_loudness_compensation


def test_loudness_compensation_matches_rms_and_clamps_gain() -> None:
    original = np.full((2, 128), 0.5, dtype=np.float32)
    quieter = np.full((2, 128), 0.25, dtype=np.float32)
    louder = np.full((2, 128), 1.0, dtype=np.float32)

    assert calculate_loudness_compensation(original, quieter) == 2.0
    assert calculate_loudness_compensation(original, louder) == 0.5
    assert calculate_loudness_compensation(original, np.zeros_like(original)) == 2.0


def test_loudness_compensation_uses_selected_fragment() -> None:
    original = np.concatenate(
        [np.full((2, 64), 0.5, dtype=np.float32), np.full((2, 64), 0.1, dtype=np.float32)],
        axis=1,
    )
    processed = np.concatenate(
        [np.full((2, 64), 0.25, dtype=np.float32), np.full((2, 64), 0.1, dtype=np.float32)],
        axis=1,
    )

    gain = calculate_loudness_compensation(original, processed, (0, 64))

    assert gain == 2.0


def test_loudness_compensation_handles_identity_and_empty_selection() -> None:
    original = np.ones((2, 128), dtype=np.float32)

    assert calculate_loudness_compensation(original, original) == 1.0
    assert calculate_loudness_compensation(original, np.zeros_like(original), (24, 24)) == 1.0
