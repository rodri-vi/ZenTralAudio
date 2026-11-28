"""Procesado por segmentos solapados para modelos cuyo consumo de memoria crece con la duración.

Medido en un M1 (pico de memoria del proceso de inferencia):
- MossFormerGAN_SE_16K: 2 s → 1,7 GB, 4 s → 2,8 GB, 10 s → 7,6 GB (agota un Mac de 8 GB).
- VoiceFixer: 10 s → 3,8 GB.
- FRCRN_SE_16K: 10 s → 1,6 GB (ClearVoice lo procesaría de una vez hasta 120 s).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.job_manager import CancellationToken


def process_in_chunks(
    signal: np.ndarray,
    sample_rate: int,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    infer: Callable[[np.ndarray], np.ndarray],
    cancel_token: CancellationToken | None = None,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Aplica ``infer`` (mono → mono) por segmentos y los une con fundidos lineales.

    Cada segmento se pondera con una rampa en las zonas solapadas y el resultado
    se normaliza por la suma de pesos, de modo que las uniones no cambian el nivel
    ni producen clics. Las señales cortas se procesan de una vez.
    """
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    total = x.shape[0]
    chunk = max(int(sample_rate * 0.5), int(round(chunk_seconds * sample_rate)))
    overlap = max(0, min(chunk // 3, int(round(overlap_seconds * sample_rate))))
    if total <= chunk:
        out = np.asarray(infer(x), dtype=np.float32).reshape(-1)
        return _fit_length(out, total)

    acc = np.zeros(total, dtype=np.float32)
    weights = np.zeros(total, dtype=np.float32)
    step = chunk - overlap
    pos = 0
    while pos < total:
        if cancel_token:
            cancel_token.raise_if_cancelled()
        end = min(total, pos + chunk)
        # El último segmento se alarga hacia atrás para no procesar un trozo diminuto.
        if end == total and end - pos < chunk and total >= chunk:
            pos = total - chunk
        segment = x[pos:end]
        seg_out = _fit_length(np.asarray(infer(segment), dtype=np.float32).reshape(-1), segment.shape[0])
        w = np.ones(segment.shape[0], dtype=np.float32)
        if overlap > 8:
            if pos > 0:
                w[:overlap] = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            if end < total:
                w[-overlap:] = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        acc[pos:end] += seg_out * w
        weights[pos:end] += w
        if progress:
            progress(end / float(total))
        if end >= total:
            break
        pos += step
    return acc / np.maximum(weights, 1e-6)


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.shape[0] >= length:
        return samples[:length]
    return np.pad(samples, (0, length - samples.shape[0]))
