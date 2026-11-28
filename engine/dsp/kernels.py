"""Núcleos recursivos compilados (Numba) con respaldo en Python puro.

Algunas etapas DSP (limitador y compresor) usan recurrencias no lineales que
NumPy no puede vectorizar. Ejecutadas en Python puro tardan varios segundos en
grabaciones largas y, peor aún, retienen el GIL durante todo ese tiempo, lo que
deja sin CPU al callback de audio de tiempo real y produce cortes audibles.

Numba ya es una dependencia transitiva de ``librosa``. Si no está disponible
(por ejemplo, en un entorno mínimo), se usa la implementación de referencia en
Python con exactamente el mismo algoritmo.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger("zentralaudio.dsp.kernels")

_LOCK = threading.Lock()
_LOADED = False
_NUMBA_LIMITER = None
_NUMBA_ATTACK_RELEASE = None


def _python_limiter_smoothing(desired_gain: np.ndarray, rel_alpha: float) -> np.ndarray:
    smooth_gain = np.empty_like(desired_gain)
    cur = 1.0
    for idx in range(desired_gain.shape[0]):
        g = float(desired_gain[idx])
        if g < cur:
            cur = 0.4 * cur + 0.6 * g
        else:
            cur = rel_alpha * cur + (1.0 - rel_alpha) * g
        smooth_gain[idx] = min(cur, g)
    return smooth_gain


def _python_attack_release(target_db: np.ndarray, att_coeff: float, rel_coeff: float) -> np.ndarray:
    out = np.empty_like(target_db)
    cur = 0.0
    for idx in range(target_db.shape[0]):
        target = float(target_db[idx])
        coeff = att_coeff if target < cur else rel_coeff
        cur = coeff * cur + (1.0 - coeff) * target
        out[idx] = cur
    return out


def _load_numba() -> None:
    global _LOADED, _NUMBA_LIMITER, _NUMBA_ATTACK_RELEASE
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        if os.environ.get("ZENTRALAUDIO_DISABLE_NUMBA") == "1":
            _LOADED = True
            return
        try:
            from numba import njit

            _NUMBA_LIMITER = njit(cache=False, nogil=True, fastmath=False)(_python_limiter_smoothing)
            _NUMBA_ATTACK_RELEASE = njit(cache=False, nogil=True, fastmath=False)(
                _python_attack_release
            )
            # Compilar ahora con firmas float64 evita la latencia de JIT en el primer render.
            _NUMBA_LIMITER(np.ones(4, dtype=np.float64), 0.5)
            _NUMBA_ATTACK_RELEASE(np.zeros(4, dtype=np.float64), 0.5, 0.5)
        except Exception as exc:  # pragma: no cover - depende del entorno
            logger.warning("Numba no disponible; se usarán núcleos DSP en Python: %s", exc)
            _NUMBA_LIMITER = None
            _NUMBA_ATTACK_RELEASE = None
        _LOADED = True


def warm_up() -> bool:
    """Compila los núcleos por adelantado. Devuelve True si Numba está activo."""
    _load_numba()
    return _NUMBA_LIMITER is not None


def limiter_gain_smoothing(desired_gain: np.ndarray, rel_alpha: float) -> np.ndarray:
    """Suavizado ataque rápido / liberación exponencial del limitador de seguridad."""
    _load_numba()
    data = np.ascontiguousarray(desired_gain, dtype=np.float64)
    if _NUMBA_LIMITER is not None:
        return _NUMBA_LIMITER(data, float(rel_alpha))
    return _python_limiter_smoothing(data, float(rel_alpha))


def attack_release_smoothing(target_db: np.ndarray, att_coeff: float, rel_coeff: float) -> np.ndarray:
    """Seguidor asimétrico attack/release del compresor (en dB)."""
    _load_numba()
    data = np.ascontiguousarray(target_db, dtype=np.float64)
    if _NUMBA_ATTACK_RELEASE is not None:
        return _NUMBA_ATTACK_RELEASE(data, float(att_coeff), float(rel_coeff))
    return _python_attack_release(data, float(att_coeff), float(rel_coeff))
