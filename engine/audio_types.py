"""Estructuras inmutables y búferes internos normalizados en float32."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.exceptions import ValidationError


@dataclass(frozen=True)
class AudioMetadata:
    """Metadatos técnicos reales de un archivo de audio cargado o exportado."""

    file_path: str | None
    sample_rate: int
    channels: int
    bit_depth: int | None
    duration_seconds: float
    num_samples: int
    codec_name: str


@dataclass
class AudioBuffer:
    """Representación interna única de audio: float32 con forma (channels, num_samples)."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim != 2:
            raise ValidationError(f"Forma de audio inválida: {arr.shape}; se esperaba 1D o 2D.")
        if self.sample_rate <= 0:
            raise ValidationError(f"Frecuencia de muestreo inválida: {self.sample_rate}")
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        self.samples = np.ascontiguousarray(arr, dtype=np.float32)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / float(self.sample_rate)

    def copy(self) -> AudioBuffer:
        return AudioBuffer(samples=self.samples.copy(), sample_rate=self.sample_rate)

    def to_mono(self) -> AudioBuffer:
        if self.channels == 1:
            return self.copy()
        mono = np.mean(self.samples, axis=0, keepdims=True, dtype=np.float32)
        return AudioBuffer(samples=mono, sample_rate=self.sample_rate)

    def to_stereo(self) -> AudioBuffer:
        if self.channels == 2:
            return self.copy()
        if self.channels == 1:
            stereo = np.repeat(self.samples, 2, axis=0)
            return AudioBuffer(samples=stereo, sample_rate=self.sample_rate)
        return AudioBuffer(samples=self.samples[:2].copy(), sample_rate=self.sample_rate)

    def match_channels(self, target_channels: int) -> AudioBuffer:
        if target_channels == 1:
            return self.to_mono()
        if target_channels == 2:
            return self.to_stereo()
        raise ValidationError(f"Número de canales no soportado: {target_channels}")
