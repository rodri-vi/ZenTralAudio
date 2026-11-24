"""Carga, normalizacion energetica y validacion de respuestas impulsionales (IR)."""

<<<<<<< HEAD
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.configuration import AppSettings
from core.exceptions import AudioIOError
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_resampler import resample_audio
from engine.audio_types import AudioBuffer


class ImpulseResponseManager:
    """Prepara respuestas impulsionales reales para convolución segura sin picos extremos."""

    def __init__(self, paths: AppPaths, settings: AppSettings | None = None) -> None:
        self.paths = paths
        self.settings = settings

    def load_ir(self, ir_path: str | Path, target_sample_rate: int) -> AudioBuffer:
        buf, _ = load_audio(ir_path, paths=self.paths, config=self.settings)
        if buf.num_samples == 0:
            raise AudioIOError(f"La respuesta impulsional está vacía: {ir_path}")

        if buf.sample_rate != target_sample_rate:
            buf = resample_audio(buf, target_sample_rate)

        # Limitar longitud máxima razonable de IR a 8 s para evitar bloqueos de memoria
        max_len = target_sample_rate * 8
        samples = buf.samples[:, :max_len].astype(np.float32, copy=True)

        # Normalizar energía L2 promedio por canal para que la convolución conserve sonoridad coherente
        energy = float(np.sqrt(np.mean(np.sum(samples ** 2, axis=1))))
        if energy > 1e-6:
            samples /= energy

        return AudioBuffer(samples=samples, sample_rate=target_sample_rate)
=======
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.configuration import AppSettings
from core.exceptions import AudioIOError
from core.paths import AppPaths
from engine.audio_io import load_audio
from engine.audio_resampler import resample_audio
from engine.audio_types import AudioBuffer


class ImpulseResponseManager:
    """Prepara respuestas impulsionales reales para convolución segura sin picos extremos."""

    def __init__(self, paths: AppPaths, settings: AppSettings | None = None) -> None:
        self.paths = paths
        self.settings = settings

    def load_ir(self, ir_path: str | Path, target_sample_rate: int) -> AudioBuffer:
        buf, _ = load_audio(ir_path, paths=self.paths, config=self.settings)
        if buf.num_samples == 0:
            raise AudioIOError(f"La respuesta impulsional está vacía: {ir_path}")

        if buf.sample_rate != target_sample_rate:
            buf = resample_audio(buf, target_sample_rate)

        # Limitar longitud máxima razonable de IR a 8 s para evitar bloqueos de memoria
        max_len = target_sample_rate * 8
        samples = buf.samples[:, :max_len].astype(np.float32, copy=True)

        # Normalizar energía L2 promedio por canal para que la convolución conserve sonoridad coherente
        energy = float(np.sqrt(np.mean(np.sum(samples ** 2, axis=1))))
        if energy > 1e-6:
            samples /= energy

        return AudioBuffer(samples=samples, sample_rate=target_sample_rate)

>>>>>>> 2f6beded7ad89e5c8d2dfbb7461aba8cf002655a