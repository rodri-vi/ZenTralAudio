"""Lectura, validación, decodificación y escritura segura (SoundFile + FFmpeg)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from core.configuration import AppSettings
from core.exceptions import (
    AudioFormatUnsupportedError,
    AudioIOError,
    FFmpegNotFoundError,
    JobCancelledError,
)
from core.job_manager import CancellationToken
from core.paths import AppPaths
from core.validation import validate_audio_input_path
from engine.audio_types import AudioBuffer, AudioMetadata

logger = logging.getLogger("zentralaudio.audio_io")

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_SOUNDFILE_DIRECT_EXTS = frozenset({".wav", ".flac", ".aiff", ".aif", ".ogg"})


def _subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    return kwargs


def _run_subprocess_cancellable(
    command: list[str],
    *,
    cancel_token: CancellationToken | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Ejecuta FFmpeg/FFprobe permitiendo terminar el proceso al cancelar."""
    if cancel_token is None:
        return subprocess.run(command, **kwargs)

    cancel_token.raise_if_cancelled()
    process = subprocess.Popen(command, **kwargs)
    try:
        while True:
            if cancel_token.is_cancelled:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                cancel_token.raise_if_cancelled()

            try:
                stdout, stderr = process.communicate(timeout=0.1)
                cancel_token.raise_if_cancelled()
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise


def _read_soundfile_cancellable(
    file_path: str | Path,
    *,
    cancel_token: CancellationToken | None = None,
) -> tuple[np.ndarray, Any]:
    """Lee audio por bloques para que una carga grande se pueda cancelar."""
    info = sf.info(str(file_path))
    if info.frames <= 0 or info.channels <= 0 or info.samplerate <= 0:
        raise AudioFormatUnsupportedError(f"El archivo '{Path(file_path).name}' no contiene audio válido.")

    samples = np.empty((int(info.frames), int(info.channels)), dtype=np.float32)
    offset = 0
    with sf.SoundFile(str(file_path), mode="r") as source:
        while offset < info.frames:
            if cancel_token:
                cancel_token.raise_if_cancelled()
            block = source.read(
                frames=min(65536, int(info.frames) - offset),
                dtype="float32",
                always_2d=True,
            )
            if block.size == 0:
                break
            end = min(int(info.frames), offset + int(block.shape[0]))
            samples[offset:end] = block[: end - offset]
            offset = end

    if cancel_token:
        cancel_token.raise_if_cancelled()
    if offset == 0:
        raise AudioFormatUnsupportedError(f"El archivo '{Path(file_path).name}' está vacío.")
    return samples[:offset], info


def _subtype_to_bit_depth(subtype: str | None) -> int | None:
    if not subtype:
        return None
    mapping = {
        "PCM_S8": 8,
        "PCM_U8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    return mapping.get(subtype.upper())


def _bit_depth_to_subtype(bit_depth: int, export_format: str) -> str:
    fmt = export_format.lower()
    if fmt == "flac":
        return "PCM_16" if bit_depth == 16 else "PCM_24"
    if bit_depth == 16:
        return "PCM_16"
    if bit_depth == 24:
        return "PCM_24"
    return "FLOAT"


def probe_audio_info(
    path: str | Path,
    *,
    paths: AppPaths | None = None,
    config: AppSettings | None = None,
    cancel_token: CancellationToken | None = None,
) -> AudioMetadata:
    """Inspecciona un archivo sin cargarlo entero en memoria cuando es posible."""
    if cancel_token:
        cancel_token.raise_if_cancelled()
    resolved = validate_audio_input_path(path)
    if resolved.suffix.lower() in _SOUNDFILE_DIRECT_EXTS:
        try:
            info = sf.info(str(resolved))
            if info.frames <= 0 or info.samplerate <= 0:
                raise AudioFormatUnsupportedError(f"El archivo '{resolved.name}' no contiene tramas válidas.")
            return AudioMetadata(
                file_path=str(resolved),
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                bit_depth=_subtype_to_bit_depth(info.subtype),
                duration_seconds=float(info.duration),
                num_samples=int(info.frames),
                codec_name=str(info.format),
            )
        except Exception as exc:
            logger.debug("sf.info falló para %s, intentando ffprobe: %s", resolved.name, exc)

    app_paths = paths or AppPaths.detect()
    custom = config.ffmpeg_custom_path if config else None
    ffprobe = app_paths.find_ffprobe(custom)
    if ffprobe is None:
        raise FFmpegNotFoundError(
            f"No se pudo inspeccionar '{resolved.name}' sin FFmpeg/FFprobe.",
            details="Coloca ffmpeg.exe y ffprobe.exe en la carpeta 'bin/' o en el PATH.",
        )

    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_raw_sample,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(resolved),
    ]
    proc = _run_subprocess_cancellable(
        cmd, cancel_token=cancel_token, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", **_subprocess_kwargs(),
    )
    if proc.returncode != 0:
        raise AudioFormatUnsupportedError(
            f"No se pudo leer la cabecera de audio de '{resolved.name}'.",
            details=proc.stderr.strip(),
        )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioFormatUnsupportedError(f"El archivo '{resolved.name}' no contiene ninguna pista de audio.")
    st = streams[0]
    sr = int(st.get("sample_rate") or 0)
    ch = int(st.get("channels") or 0)
    dur = float(st.get("duration") or (data.get("format") or {}).get("duration") or 0.0)
    raw_bits = st.get("bits_per_raw_sample")
    bit_depth = int(raw_bits) if raw_bits and str(raw_bits).isdigit() else None
    if sr <= 0 or ch <= 0:
        raise AudioFormatUnsupportedError(f"Metadatos de audio inválidos en '{resolved.name}'.")

    return AudioMetadata(
        file_path=str(resolved),
        sample_rate=sr,
        channels=ch,
        bit_depth=bit_depth,
        duration_seconds=dur,
        num_samples=int(round(dur * sr)),
        codec_name=str(st.get("codec_name") or resolved.suffix.lstrip(".")),
    )


def load_audio(
    path: str | Path,
    *,
    paths: AppPaths | None = None,
    config: AppSettings | None = None,
    cancel_token: CancellationToken | None = None,
) -> tuple[AudioBuffer, AudioMetadata]:
    """Carga y decodifica un archivo de audio en float32 conservando su frecuencia original."""
    if cancel_token:
        cancel_token.raise_if_cancelled()
    resolved = validate_audio_input_path(path)
    ext = resolved.suffix.lower()

    if ext in _SOUNDFILE_DIRECT_EXTS:
        try:
            samples, info = _read_soundfile_cancellable(resolved, cancel_token=cancel_token)
            if samples.size == 0:
                raise AudioFormatUnsupportedError(f"El archivo '{resolved.name}' no contiene audio.")
            buf = AudioBuffer(samples=samples.T, sample_rate=int(info.samplerate))
            meta = AudioMetadata(
                file_path=str(resolved),
                sample_rate=buf.sample_rate,
                channels=buf.channels,
                bit_depth=_subtype_to_bit_depth(info.subtype),
                duration_seconds=buf.duration_seconds,
                num_samples=buf.num_samples,
                codec_name=str(info.format),
            )
            if cancel_token:
                cancel_token.raise_if_cancelled()
            return buf, meta
        except (AudioFormatUnsupportedError, JobCancelledError):
            raise
        except Exception as exc:
            logger.debug("Lectura directa SoundFile falló para %s (%s); probando FFmpeg.", resolved.name, exc)

    return _decode_with_ffmpeg(resolved, paths=paths, config=config, cancel_token=cancel_token)


def _decode_with_ffmpeg(
    resolved: Path,
    *,
    paths: AppPaths | None,
    config: AppSettings | None,
    cancel_token: CancellationToken | None = None,
) -> tuple[AudioBuffer, AudioMetadata]:
    if cancel_token:
        cancel_token.raise_if_cancelled()
    app_paths = paths or AppPaths.detect()
    custom = config.ffmpeg_custom_path if config else None
    ffmpeg = app_paths.find_ffmpeg(custom)
    if ffmpeg is None:
        raise FFmpegNotFoundError(
            f"Se requiere FFmpeg para decodificar el formato '{resolved.suffix}'.",
            details="Coloca ffmpeg.exe en 'bin/' o configúralo en Ajustes.",
        )

    # Intentar obtener metadatos originales primero si ffprobe está disponible
    probed: AudioMetadata | None = None
    try:
        probed = probe_audio_info(
            resolved, paths=app_paths, config=config, cancel_token=cancel_token
        )
    except JobCancelledError:
        raise
    except Exception:
        probed = None

    with tempfile.TemporaryDirectory(prefix="zentral_dec_") as tmp_dir:
        tmp_wav = Path(tmp_dir) / "decoded.wav"
        cmd = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(resolved),
            "-vn",
            "-c:a",
            "pcm_f32le",
            str(tmp_wav),
        ]
        proc = _run_subprocess_cancellable(
            cmd, cancel_token=cancel_token, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", **_subprocess_kwargs(),
        )
        if proc.returncode != 0 or not tmp_wav.is_file():
            raise AudioFormatUnsupportedError(
                f"FFmpeg no pudo decodificar '{resolved.name}'.",
                details=proc.stderr.strip(),
            )

        data, decoded_info = _read_soundfile_cancellable(tmp_wav, cancel_token=cancel_token)
        if data.size == 0:
            raise AudioFormatUnsupportedError(f"El archivo '{resolved.name}' decodificado está vacío.")

        buf = AudioBuffer(samples=data.T, sample_rate=int(decoded_info.samplerate))
        meta = AudioMetadata(
            file_path=str(resolved),
            sample_rate=buf.sample_rate,
            channels=buf.channels,
            bit_depth=probed.bit_depth if probed else None,
            duration_seconds=buf.duration_seconds,
            num_samples=buf.num_samples,
            codec_name=probed.codec_name if probed else resolved.suffix.lstrip("."),
        )
        if cancel_token:
            cancel_token.raise_if_cancelled()
        return buf, meta


def write_audio(
    buffer: AudioBuffer,
    output_path: Path,
    *,
    export_format: str,
    bit_depth: int = 24,
    mp3_bitrate_kbps: int = 320,
    paths: AppPaths | None = None,
    config: AppSettings | None = None,
    cancel_token: CancellationToken | None = None,
) -> AudioMetadata:
    """Escribe el audio de forma atómica y verifica la integridad básica del archivo final."""
    fmt = export_format.lower().strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = output_path.with_name(f".{output_path.stem}.tmp.{fmt}")

    try:
        if cancel_token:
            cancel_token.raise_if_cancelled()

        safe_samples = np.clip(buffer.samples, -1.0, 1.0).astype(np.float32, copy=False)

        if fmt in ("wav", "flac"):
            subtype = _bit_depth_to_subtype(bit_depth, fmt)
            sf_format = "WAV" if fmt == "wav" else "FLAC"
            sf.write(
                str(tmp_target),
                safe_samples.T,
                buffer.sample_rate,
                subtype=subtype,
                format=sf_format,
            )
        elif fmt == "mp3":
            app_paths = paths or AppPaths.detect()
            custom = config.ffmpeg_custom_path if config else None
            ffmpeg = app_paths.find_ffmpeg(custom)
            if ffmpeg is None:
                raise FFmpegNotFoundError(
                    "Se requiere FFmpeg para exportar en formato MP3.",
                    details="Configura ffmpeg.exe en Ajustes o exporta en WAV/FLAC.",
                )
            with tempfile.TemporaryDirectory(prefix="zentral_enc_") as tmp_dir:
                wav_in = Path(tmp_dir) / "stage.wav"
                sf.write(str(wav_in), safe_samples.T, buffer.sample_rate, subtype="FLOAT", format="WAV")
                if cancel_token:
                    cancel_token.raise_if_cancelled()
                cmd = [
                    str(ffmpeg),
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(wav_in),
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    f"{int(mp3_bitrate_kbps)}k",
                    str(tmp_target),
                ]
                proc = _run_subprocess_cancellable(
                    cmd, cancel_token=cancel_token, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", **_subprocess_kwargs(),
                )
                if proc.returncode != 0 or not tmp_target.is_file():
                    raise AudioIOError(
                        "Error al codificar el archivo MP3 con FFmpeg.",
                        details=proc.stderr.strip(),
                    )
        else:
            raise AudioFormatUnsupportedError(f"Formato de salida no soportado: '{export_format}'")

        if cancel_token:
            cancel_token.raise_if_cancelled()

        os.replace(tmp_target, output_path)
        # Verificación post-escritura
        verified_buf, verified_meta = load_audio(
            output_path, paths=paths, config=config, cancel_token=cancel_token
        )
        if verified_buf.num_samples == 0:
            raise AudioIOError(f"El archivo exportado resultó vacío: {output_path}")
        return verified_meta
    except BaseException:
        tmp_target.unlink(missing_ok=True)
        raise
