"""Utilidades STFT por bloques con memoria acotada y overlap-add vectorizado.

Las versiones originales de la reducción de ruido y del dereverb construían todas
las tramas de una vez (``np.stack`` sobre listas Python) y acumulaban el
overlap-add trama a trama. En una grabación de 10 minutos eso son ~100.000
iteraciones Python por canal y matrices complejas de cientos de MB.

Aquí las tramas se procesan por bloques mediante vistas de ventana deslizante y
la FFT de ``scipy.fft`` (multihilo y sin GIL). El resultado es numéricamente
equivalente al algoritmo original salvo redondeo en coma flotante.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sp_fft

DEFAULT_CHUNK_FRAMES = 4096


def pad_for_frames(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, int]:
    """Rellena la señal para que cada muestra original tenga solapamiento completo.

    Devuelve ``(x_pad, lead)``. Se añaden ``n_fft - hop`` ceros al principio y al
    final: sin ellos, las primeras y últimas muestras se normalizaban dividiendo
    por una suma de ventanas casi nula, lo que generaba un clic de gran amplitud
    al inicio del audio procesado. El final se completa además hasta una trama
    entera.
    """
    x = x.astype(np.float32, copy=False)
    lead = n_fft - hop
    n = x.shape[0] + 2 * lead
    tail = lead + ((hop - ((n - n_fft) % hop)) % hop if n >= n_fft else (n_fft - n))
    return np.pad(x, (lead, tail), mode="constant"), lead


def frame_count(total: int, n_fft: int, hop: int) -> int:
    return (total - n_fft) // hop + 1


def iter_spectra(
    x_pad: np.ndarray,
    window: np.ndarray,
    hop: int,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> Iterator[tuple[int, np.ndarray]]:
    """Genera (índice_primera_trama, espectro_rfft) por bloques de tramas."""
    n_fft = window.shape[0]
    frames_view = sliding_window_view(x_pad, n_fft)[::hop]
    total_frames = frames_view.shape[0]
    for first in range(0, total_frames, chunk_frames):
        block = frames_view[first : first + chunk_frames] * window
        yield first, sp_fft.rfft(block, axis=1, workers=-1)


def overlap_add_chunk(
    y: np.ndarray,
    frames: np.ndarray,
    first_frame: int,
    hop: int,
) -> None:
    """Suma tramas ya enventanadas en ``y`` cuando n_fft es múltiplo de hop."""
    count, n_fft = frames.shape
    ratio = n_fft // hop
    start = first_frame * hop
    span = (count + ratio - 1) * hop
    target = y[start : start + span].reshape(count + ratio - 1, hop)
    parts = frames.reshape(count, ratio, hop)
    for k in range(ratio):
        target[k : k + count] += parts[:, k, :]


def overlap_add_norm(total: int, window_sq: np.ndarray, hop: int, frames: int) -> np.ndarray:
    """Suma de ventanas al cuadrado para normalizar la reconstrucción."""
    n_fft = window_sq.shape[0]
    ratio = n_fft // hop
    norm = np.zeros(total, dtype=np.float32)
    blocks = norm[: (frames + ratio - 1) * hop].reshape(frames + ratio - 1, hop)
    quarters = window_sq.reshape(ratio, hop)
    for k in range(ratio):
        blocks[k : k + frames] += quarters[k]
    return norm


def inverse_chunk(spec: np.ndarray, n_fft: int) -> np.ndarray:
    return sp_fft.irfft(spec, n=n_fft, axis=1, workers=-1).astype(np.float32, copy=False)
