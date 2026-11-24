"""Mezclador alineado dry/wet, panorámica de potencia constante y mezcla de capas ambientales."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from engine.audio_resampler import resample_audio
from engine.audio_types import AudioBuffer


def dry_wet_mix(dry: AudioBuffer, wet: AudioBuffer, wet_ratio: float) -> AudioBuffer:
    """Mezcla lineal compensada entre señal original (dry) y procesada (wet)."""
    ratio = max(0.0, min(1.0, float(wet_ratio)))
    if ratio <= 0.0:
        return dry.copy()

    aligned_wet = wet
    if aligned_wet.sample_rate != dry.sample_rate or aligned_wet.num_samples != dry.num_samples:
        aligned_wet = resample_audio(aligned_wet, dry.sample_rate, target_length=dry.num_samples)

    target_ch = max(dry.channels, aligned_wet.channels)
    dry_m = dry.match_channels(target_ch)
    wet_m = aligned_wet.match_channels(target_ch)

    if ratio >= 1.0:
        return wet_m

    mixed = (1.0 - ratio) * dry_m.samples + ratio * wet_m.samples
    return AudioBuffer(samples=mixed, sample_rate=dry.sample_rate)


def apply_pan(buffer: AudioBuffer, pan: float) -> AudioBuffer:
    """Aplica panorámica estéreo (-1.0 izquierda .. 0.0 centro .. +1.0 derecha)."""
    p = max(-1.0, min(1.0, float(pan)))
    stereo = buffer.to_stereo()
    angle = (p + 1.0) * (math.pi / 4.0)
    gain_l = math.cos(angle) * math.sqrt(2.0)
    gain_r = math.sin(angle) * math.sqrt(2.0)

    out = stereo.samples.copy()
    if abs(p) > 1e-4:
        out[0] *= gain_l
        out[1] *= gain_r
    return AudioBuffer(samples=out, sample_rate=stereo.sample_rate)


def tile_with_crossfade(layer: AudioBuffer, target_samples: int, crossfade_samples: int = 2048) -> AudioBuffer:
    """Repite suavemente una grabación ambiental para cubrir la duración objetivo sin clics."""
    if layer.num_samples >= target_samples:
        return AudioBuffer(samples=layer.samples[:, :target_samples].copy(), sample_rate=layer.sample_rate)

    src = layer.samples
    src_len = src.shape[1]
    cf = min(crossfade_samples, max(0, src_len // 4))
    out = np.zeros((src.shape[0], target_samples), dtype=np.float32)

    if cf < 16:
        reps = (target_samples // src_len) + 1
        tiled = np.tile(src, (1, reps))[:, :target_samples]
        return AudioBuffer(samples=tiled, sample_rate=layer.sample_rate)

    step = src_len - cf
    pos = 0
    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)[np.newaxis, :]
    fade_out = 1.0 - fade_in

    first = True
    while pos < target_samples:
        remain = target_samples - pos
        take = min(src_len, remain)
        chunk = src[:, :take].copy()
        if not first and take >= cf:
            chunk[:, :cf] *= fade_in
            out[:, pos : pos + cf] *= fade_out
        out[:, pos : pos + take] += chunk
        pos += step
        first = False

    return AudioBuffer(samples=out, sample_rate=layer.sample_rate)


def mix_ambience_layers(
    voice: AudioBuffer,
    layer_items: list[tuple[AudioBuffer, dict[str, Any]]],
    *,
    master_volume: float = 0.6,
) -> AudioBuffer:
    """Combina capas ambientales con solo, volumen, pan y margen seguro sobre la voz."""
    master = max(0.0, min(1.5, float(master_volume)))
    any_solo = any(bool(cfg.get("solo", False)) for _, cfg in layer_items)

    needs_stereo = any(
        buf.channels > 1 or abs(float(cfg.get("pan", 0.0))) > 1e-4 for buf, cfg in layer_items
    )
    voice_out = voice.to_stereo() if needs_stereo else voice.copy()
    target_ch = voice_out.channels
    target_sr = voice_out.sample_rate
    target_len = voice_out.num_samples

    amb_mix = np.zeros((target_ch, target_len), dtype=np.float32)
    active_count = 0

    for raw_buf, cfg in layer_items:
        enabled = bool(cfg.get("enabled", True))
        is_solo = bool(cfg.get("solo", False))
        if not enabled or (any_solo and not is_solo):
            continue

        vol = max(0.0, min(1.5, float(cfg.get("volume", 0.35))))
        pan = max(-1.0, min(1.0, float(cfg.get("pan", 0.0))))

        aligned = raw_buf
        if aligned.sample_rate != target_sr:
            aligned = resample_audio(aligned, target_sr)
        looped = tile_with_crossfade(aligned, target_len)
        panned = apply_pan(looped, pan) if target_ch == 2 else looped.to_mono()
        amb_mix += panned.samples * vol
        active_count += 1

    if active_count > 1:
        amb_mix /= math.sqrt(float(active_count)) * 0.85

    combined = voice_out.samples + amb_mix * master
    return AudioBuffer(samples=combined, sample_rate=target_sr)
