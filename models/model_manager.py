"""Administrador seguro de descargas, verificación SHA-256 y extracción de pesos IA."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from core.exceptions import IntegrityCheckError, ResourceDownloadError
from core.job_manager import CancellationToken
from core.paths import AppPaths

logger = logging.getLogger("zentralaudio.model_manager")


@dataclass(frozen=True)
class ModelFileSpec:
    filename: str
    url: str
    sha256: str | None
    approx_size_mb: int


@dataclass(frozen=True)
class ModelManifestEntry:
    model_id: str
    display_name: str
    required_sample_rate: int
    license_code: str
    license_weights: str
    attribution_required: bool
    source_url: str
    download_supported: bool
    archive_type: str
    files: tuple[ModelFileSpec, ...]
    runtime_requirement: str


@dataclass(frozen=True)
class DownloadResult:
    model_id: str
    target_dir: Path
    verified_by_sha256: bool
    verification_note: str


def compute_sha256(file_path: Path) -> str:
    """Calcula el digest SHA-256 hexadecimal de un archivo por bloques."""
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrae un ZIP validando rutas contra ZipSlip."""
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest_resolved / member.filename).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise IntegrityCheckError(
                    f"Archivo comprimido inseguro detectado (ruta fuera de destino): {member.filename}"
                )
        zf.extractall(dest_resolved)


class ModelManager:
    """Gestiona manifiesto, descargas con consentimiento y reanudación e integridad."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._entries = self._load_manifest()

    def _load_manifest(self) -> dict[str, ModelManifestEntry]:
        manifest_path = self.paths.resolve_resource("manifests", "models_manifest.json")
        if not manifest_path.is_file():
            logger.warning("Manifiesto de modelos no encontrado en %s", manifest_path)
            return {}
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries: dict[str, ModelManifestEntry] = {}
        for item in raw.get("models", []):
            files = tuple(
                ModelFileSpec(
                    filename=str(f["filename"]),
                    url=str(f["url"]),
                    sha256=str(f["sha256"]).lower() if f.get("sha256") else None,
                    approx_size_mb=int(f.get("approx_size_mb", 0)),
                )
                for f in item.get("files", [])
            )
            entry = ModelManifestEntry(
                model_id=str(item["model_id"]),
                display_name=str(item["display_name"]),
                required_sample_rate=int(item["required_sample_rate"]),
                license_code=str(item.get("license_code", "")),
                license_weights=str(item.get("license_weights", "")),
                attribution_required=bool(item.get("attribution_required", False)),
                source_url=str(item.get("source_url", "")),
                download_supported=bool(item.get("download_supported", False)),
                archive_type=str(item.get("archive_type", "none")),
                files=files,
                runtime_requirement=str(item.get("runtime_requirement", "")),
            )
            entries[entry.model_id] = entry
        return entries

    def get_entry(self, model_id: str) -> ModelManifestEntry | None:
        return self._entries.get(model_id)

    def list_entries(self) -> list[ModelManifestEntry]:
        return list(self._entries.values())

    def model_dir(self, model_id: str) -> Path:
        return self.paths.models_dir / model_id

    def has_incomplete_files(self, model_id: str) -> bool:
        d = self.model_dir(model_id)
        return d.is_dir() and any(d.glob("*.part"))

    def is_weights_present(self, model_id: str) -> bool:
        entry = self.get_entry(model_id)
        d = self.model_dir(model_id)
        if entry is None or not entry.files or not d.is_dir():
            return False
        if entry.archive_type == "zip":
            # Para ZIP extraídos (p. ej. DeepFilterNet3), verificar marcador de extracción completada
            return (d / ".download_verified.json").is_file()
        return all((d / f.filename).is_file() and (d / f.filename).stat().st_size > 0 for f in entry.files)

    def download_model(
        self,
        model_id: str,
        *,
        user_consented: bool,
        progress_cb: Callable[[float, str], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> DownloadResult:
        """Descarga los pesos del modelo exclusivamente tras consentimiento expreso del usuario."""
        if not user_consented:
            raise ResourceDownloadError(
                "Descarga bloqueada: se requiere consentimiento explícito del usuario antes de descargar pesos."
            )

        entry = self.get_entry(model_id)
        if entry is None or not entry.download_supported or not entry.files:
            raise ResourceDownloadError(
                f"El modelo '{model_id}' no dispone de una fuente de descarga verificada.",
                details=entry.runtime_requirement if entry else None,
            )

        target_dir = self.model_dir(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        all_hashed = True
        total_files = len(entry.files)

        for idx, file_spec in enumerate(entry.files):
            if cancel_token:
                cancel_token.raise_if_cancelled()

            dest_file = target_dir / file_spec.filename
            part_file = target_dir / f"{file_spec.filename}.part"

            if not dest_file.is_file() or dest_file.stat().st_size == 0:
                self._download_single_file(
                    url=file_spec.url,
                    dest_file=dest_file,
                    part_file=part_file,
                    label=f"{entry.display_name}: {file_spec.filename}",
                    file_index=idx,
                    total_files=total_files,
                    progress_cb=progress_cb,
                    cancel_token=cancel_token,
                )

            if file_spec.sha256:
                if progress_cb:
                    progress_cb((idx + 0.95) / total_files, f"Verificando SHA-256 de {file_spec.filename}...")
                actual = compute_sha256(dest_file)
                if actual != file_spec.sha256.lower():
                    dest_file.unlink(missing_ok=True)
                    raise IntegrityCheckError(
                        f"El hash SHA-256 de '{file_spec.filename}' no coincide con el oficial.",
                        details=f"Esperado: {file_spec.sha256}\nObtenido: {actual}",
                    )
            else:
                all_hashed = False
                logger.warning(
                    "Archivo '%s' de '%s' no posee SHA-256 oficial en origen; validado por tamaño > 0.",
                    file_spec.filename,
                    model_id,
                )

            if entry.archive_type == "zip" and dest_file.suffix.lower() == ".zip":
                if progress_cb:
                    progress_cb(0.98, f"Extrayendo {file_spec.filename}...")
                _safe_extract_zip(dest_file, target_dir)

        note = (
            "Integridad verificada mediante SHA-256 oficial."
            if all_hashed
            else "Descarga completada; alguno(s) de los archivos no disponen de hash SHA-256 oficial publicado."
        )
        marker = target_dir / ".download_verified.json"
        marker.write_text(
            json.dumps({"model_id": model_id, "verified_by_sha256": all_hashed, "note": note}, indent=2),
            encoding="utf-8",
        )
        if progress_cb:
            progress_cb(1.0, f"{entry.display_name} listo")
        return DownloadResult(
            model_id=model_id,
            target_dir=target_dir,
            verified_by_sha256=all_hashed,
            verification_note=note,
        )

    def _download_single_file(
        self,
        *,
        url: str,
        dest_file: Path,
        part_file: Path,
        label: str,
        file_index: int,
        total_files: int,
        progress_cb: Callable[[float, str], None] | None,
        cancel_token: CancellationToken | None,
    ) -> None:
        existing_bytes = part_file.stat().st_size if part_file.is_file() else 0
        headers: dict[str, str] = {"User-Agent": "ZenTralAudio/0.1"}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as resp:
                if resp.status_code == 416 and existing_bytes > 0:
                    part_file.replace(dest_file)
                    return
                resp.raise_for_status()

                resuming = resp.status_code == 206 and existing_bytes > 0
                mode = "ab" if resuming else "wb"
                downloaded = existing_bytes if resuming else 0
                content_len = int(resp.headers.get("Content-Length") or 0)
                total_bytes = (existing_bytes + content_len) if resuming and content_len > 0 else content_len

                with part_file.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        if cancel_token:
                            cancel_token.raise_if_cancelled()
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            file_frac = (downloaded / float(total_bytes)) if total_bytes > 0 else 0.5
                            overall = (file_index + min(0.94, file_frac)) / float(total_files)
                            mb = downloaded / (1024 * 1024)
                            progress_cb(overall, f"Descargando {label} ({mb:.1f} MB)...")
        except requests.RequestException as exc:
            raise ResourceDownloadError(
                f"Error de descarga para '{dest_file.name}'. Puedes reintentar para reanudar si el servidor lo permite.",
                details=str(exc),
            ) from exc

        if part_file.stat().st_size == 0:
            part_file.unlink(missing_ok=True)
            raise ResourceDownloadError(f"El archivo descargado resultó vacío: {dest_file.name}")
        part_file.replace(dest_file)

    def delete_model(self, model_id: str) -> None:
        """Elimina todos los pesos descargados de un modelo en models_data/."""
        d = self.model_dir(model_id)
        if d.is_dir():
            shutil.rmtree(d)
