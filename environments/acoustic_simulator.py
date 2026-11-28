"""Simulacion acustica mediante convolucion IR real y sintesis geometrica algoritmica."""

from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from scipy.signal import butter, oaconvolve, sosfilt

from engine.audio_mixer import apply_pan
from engine.audio_types import AudioBuffer
from engine.dsp.spatial import apply_stereo_width


def _apply_air_damping(samples: np.ndarray, sample_rate: int, damping_pct: float) -> np.ndarray:
    """Atenúa progresivamente agudos en la cola reflejada según el parámetro damping (0..100)."""
    d = max(0.0, min(100.0, float(damping_pct))) / 100.0
    if d <= 0.02 or samples.shape[1] < 16:
        return samples
    cutoff = max(1200.0, (sample_rate * 0.45) * (1.0 - 0.82 * d))
    nyq = sample_rate * 0.5
    if cutoff >= nyq * 0.94:
        return samples
    sos = butter(2, cutoff / nyq, btype="lowpass", output="sos")
    return np.asarray(sosfilt(sos, samples, axis=1), dtype=np.float32)


def _apply_predelay(samples: np.ndarray, sample_rate: int, predelay_ms: float) -> np.ndarray:
    delay_frames = max(0, int(round((max(0.0, float(predelay_ms)) / 1000.0) * sample_rate)))
    if delay_frames == 0:
        return samples
    n = samples.shape[1]
    if delay_frames >= n:
        return np.zeros_like(samples, dtype=np.float32)
    out = np.zeros_like(samples, dtype=np.float32)
    out[:, delay_frames:] = samples[:, : n - delay_frames]
    return out


@lru_cache(maxsize=12)
def _build_algorithmic_room_ir_cached(
    sample_rate: int,
    *,
    room_size: float = 45.0,
    decay_s: float = 1.2,
    damping: float = 45.0,
) -> np.ndarray:
    """Genera una respuesta acústica determinista de sala (reflexiones tempranas + cola difusa).

    No pretende sustituir una IR real medida, sino ofrecer una simulación geométrica
    paramétrica real y reproducible cuando no se ha cargado un archivo IR externo.
    """
    rt60 = max(0.10, min(6.0, float(decay_s)))
    size_norm = max(0.05, min(1.0, float(room_size) / 100.0))
    ir_len = max(256, min(sample_rate * 5, int(round(rt60 * sample_rate * 1.15))))
    ir = np.zeros((2, ir_len), dtype=np.float32)

    # 1. Reflexiones tempranas con tiempos primos escalados por tamaño aparente del espacio
    early_base_ms = np.array([7.3, 11.7, 17.9, 23.3, 31.1, 41.9, 53.7, 67.3], dtype=np.float64)
    early_times_ms = early_base_ms * (0.4 + 1.1 * size_norm)
    for idx, t_ms in enumerate(early_times_ms):
        pos = int(round((t_ms / 1000.0) * sample_rate))
        if 0 < pos < ir_len:
            amp = 0.55 * (0.78 ** idx)
            # Decorrelación izquierda/derecha realista
            ir[0, pos] += amp * (1.0 if idx % 2 == 0 else 0.65)
            ir[1, pos] += amp * (0.65 if idx % 2 == 0 else 1.0)

    # 2. Cola difusa densa determinista (suma de pulsos escalonados + envolvente RT60)
    rng = np.random.default_rng(seed=int(round(size_norm * 1000 + rt60 * 100)))
    t_sec = np.arange(ir_len, dtype=np.float64) / float(sample_rate)
    decay_env = np.exp(-6.907755 * t_sec / rt60)
    tail_onset = int(round((early_times_ms[1] / 1000.0) * sample_rate))
    tail = rng.standard_normal((2, ir_len)).astype(np.float64) * decay_env[np.newaxis, :] * 0.25
    if tail_onset > 0:
        ramp = np.minimum(1.0, np.arange(ir_len, dtype=np.float64) / float(max(1, tail_onset)))
        tail *= ramp[np.newaxis, :]

    ir_raw = (ir.astype(np.float64) + tail).astype(np.float32)
    ir_damped = _apply_air_damping(ir_raw, sample_rate, damping)

    # Normalizar energía total para mantener coherencia de nivel
    energy = float(np.sqrt(np.mean(np.sum(ir_damped ** 2, axis=1))))
    if energy > 1e-6:
        ir_damped /= energy
    return ir_damped


def build_algorithmic_room_ir(
    sample_rate: int,
    *,
    room_size: float = 45.0,
    decay_s: float = 1.2,
    damping: float = 45.0,
) -> AudioBuffer:
    """Obtiene la IR paramétrica desde una caché LRU y devuelve una copia segura."""
    cached = _build_algorithmic_room_ir_cached(
        int(sample_rate),
        room_size=float(room_size),
        decay_s=float(decay_s),
        damping=float(damping),
    )
    return AudioBuffer(samples=cached.copy(), sample_rate=int(sample_rate))


_WET_CACHE_LOCK = threading.Lock()
_WET_CACHE_KEY: str | None = None
_WET_CACHE_VALUE: np.ndarray | None = None
_DIRECT_CACHE_KEY: str | None = None
_DIRECT_CACHE_VALUE: np.ndarray | None = None
_CONVOLUTION_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ZenTralConv")


def clear_acoustic_caches() -> None:
    """Libera las señales húmeda y directa retenidas para reutilización."""
    global _WET_CACHE_KEY, _WET_CACHE_VALUE, _DIRECT_CACHE_KEY, _DIRECT_CACHE_VALUE
    with _WET_CACHE_LOCK:
        _WET_CACHE_KEY = None
        _WET_CACHE_VALUE = None
        _DIRECT_CACHE_KEY = None
        _DIRECT_CACHE_VALUE = None


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _cached_wet_convolution(
    audio_key: str,
    ir_key: str,
    audio: np.ndarray,
    impulse: np.ndarray,
) -> np.ndarray:
    """Reutiliza un único wet render sin retener copias de los WAV como claves LRU.

    Devuelve un array de solo lectura compartido con la caché: los consumidores
    deben crear nuevos arrays en lugar de modificarlo (se fuerza con ``write=False``).
    Los dos canales se convolucionan en paralelo; la FFT de SciPy libera el GIL.
    """
    global _WET_CACHE_KEY, _WET_CACHE_VALUE
    key = f"{audio_key}:{ir_key}:{audio.shape}:{impulse.shape}"
    with _WET_CACHE_LOCK:
        if _WET_CACHE_KEY == key and _WET_CACHE_VALUE is not None:
            return _WET_CACHE_VALUE

    n = audio.shape[1]
    channels = audio.shape[0]

    def convolve_channel(channel: int) -> np.ndarray:
        conv = oaconvolve(audio[channel], impulse[channel], mode="full")[:n]
        return conv.astype(np.float32, copy=False)

    out = np.empty((channels, n), dtype=np.float32)
    for channel, result in enumerate(_CONVOLUTION_POOL.map(convolve_channel, range(channels))):
        out[channel] = result

    _readonly(out)
    with _WET_CACHE_LOCK:
        _WET_CACHE_KEY = key
        _WET_CACHE_VALUE = out
    return out


def _cached_direct_damping(token: str, samples: np.ndarray, sample_rate: int, damping_pct: float) -> np.ndarray:
    """Absorción aérea de la voz directa; solo depende de la voz y de la distancia."""
    global _DIRECT_CACHE_KEY, _DIRECT_CACHE_VALUE
    key = f"{token}:{samples.shape}:{sample_rate}:{damping_pct:.6f}"
    with _WET_CACHE_LOCK:
        if _DIRECT_CACHE_KEY == key and _DIRECT_CACHE_VALUE is not None:
            return _DIRECT_CACHE_VALUE
    damped = _apply_air_damping(samples, sample_rate, damping_pct)
    if damped is samples:
        return samples
    _readonly(damped)
    with _WET_CACHE_LOCK:
        _DIRECT_CACHE_KEY = key
        _DIRECT_CACHE_VALUE = damped
    return damped


def _content_key(samples: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(samples, dtype=np.float32)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=12).hexdigest()


def simulate_acoustic_space(
    voice: AudioBuffer,
    *,
    ir_buffer: AudioBuffer | None,
    acoustics: dict[str, float],
    cache_token: str | None = None,
) -> AudioBuffer:
    """Aplica convolución IR o simulación algorítmica más distancia aparente, anchura y pan.

    ``cache_token`` identifica de forma determinista el contenido de ``voice``
    (lo proporciona el pipeline a partir de sus claves de caché). Si se omite, se
    calcula un hash del contenido. Con él se reutilizan la señal húmeda (la
    convolución, la parte cara) y la voz directa filtrada cuando solo cambian
    mezcla, niveles, predelay, anchura o panorámica.
    """
    sr = voice.sample_rate
    room_size = float(acoustics.get("room_size", 45.0))
    reverb_mix = max(0.0, min(1.0, float(acoustics.get("reverb_mix", 28.0)) / 100.0))
    decay_s = float(acoustics.get("decay_s", 1.2))
    predelay_ms = float(acoustics.get("predelay_ms", 14.0))
    damping = float(acoustics.get("damping", 45.0))
    direct_level = max(0.0, min(1.5, float(acoustics.get("direct_level", 100.0)) / 100.0))
    reflected_level = max(0.0, min(1.5, float(acoustics.get("reflected_level", 40.0)) / 100.0))
    stereo_width = max(0.0, min(1.5, float(acoustics.get("stereo_width", 100.0)) / 100.0))
    distance = max(0.0, min(1.0, float(acoustics.get("apparent_distance", 20.0)) / 100.0))
    pan = max(-1.0, min(1.0, float(acoustics.get("pan", 0.0)) / 100.0))

    # Convertir voz a estéreo para imagen espacial consistente
    voice_st = voice.to_stereo()
    voice_samples = np.ascontiguousarray(voice_st.samples, dtype=np.float32)
    n = voice_samples.shape[1]
    token = cache_token or _content_key(voice_samples)

    # Simulación de absorción aérea en voz directa según distancia aparente
    if distance > 0.05:
        direct_sig = _cached_direct_damping(token, voice_samples, sr, distance * 65.0)
    else:
        direct_sig = voice_samples

    # Nivel efectivo directo vs reflejado ajustado por distancia aparente
    eff_direct = direct_level * (1.0 - 0.45 * distance)
    eff_reflected = reflected_level * (1.0 + 0.65 * distance)

    if reverb_mix <= 0.005 or eff_reflected <= 0.005:
        out = AudioBuffer(samples=direct_sig * np.float32(eff_direct), sample_rate=sr)
        out = apply_stereo_width(out, width=stereo_width)
        return apply_pan(out, pan) if abs(pan) > 1e-4 else out

    if ir_buffer is not None:
        ir = ir_buffer.to_stereo()
        ir_samples = _apply_air_damping(ir.samples, sr, damping * 0.6)
    else:
        ir = build_algorithmic_room_ir(sr, room_size=room_size, decay_s=decay_s, damping=damping)
        ir_samples = ir.samples

    # Overlap-add es más eficiente que una FFT global para voz larga + IR de varios segundos.
    ir_contiguous = np.ascontiguousarray(ir_samples, dtype=np.float32)
    wet = _cached_wet_convolution(token, _content_key(ir_contiguous), voice_samples, ir_contiguous)

    # Mezcla dry/wet equivalente a dry_wet_mix(directo, húmedo con predelay) seguida de
    # la compensación de nivel, calculada en un único array de salida.
    compensation = 1.0 / max(0.8, math.sqrt(1.0 - reverb_mix * 0.25))
    dry_gain = np.float32(eff_direct * (1.0 - reverb_mix) * compensation)
    wet_gain = np.float32(eff_reflected * reverb_mix * compensation)
    mixed = np.multiply(direct_sig, dry_gain, dtype=np.float32)
    delay_frames = max(0, int(round((max(0.0, predelay_ms) / 1000.0) * sr)))
    if delay_frames < n:
        scratch = np.multiply(wet[:, : n - delay_frames], wet_gain, dtype=np.float32)
        mixed[:, delay_frames:] += scratch
        del scratch

    if abs(stereo_width - 1.0) >= 1e-4:
        _stereo_width_inplace(mixed, stereo_width)
    if abs(pan) > 1e-4:
        angle = (pan + 1.0) * (math.pi / 4.0)
        mixed[0] *= np.float32(math.cos(angle) * math.sqrt(2.0))
        mixed[1] *= np.float32(math.sin(angle) * math.sqrt(2.0))
    return AudioBuffer(samples=mixed, sample_rate=sr)


def _stereo_width_inplace(samples: np.ndarray, width: float) -> None:
    """Matriz Mid/Side idéntica a ``apply_stereo_width`` sin copias completas extra."""
    w = np.float32(max(0.0, min(2.0, float(width))))
    left = samples[0]
    right = samples[1]
    mid = (left + right) * np.float32(0.5)
    side = (left - right) * np.float32(0.5 * w)
    np.add(mid, side, out=left)
    np.subtract(mid, side, out=right)
