"""Gestion transparente de recursos ambientales e IRs: estado, licencias y descarga con consentimiento."""

<<<<<<< HEAD
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import requests

from core.exceptions import IntegrityCheckError, ResourceDownloadError
from core.job_manager import CancellationToken
from core.paths import AppPaths
from models.model_manager import compute_sha256

logger = logging.getLogger("zentralaudio.environments.ambience_manager")


class ResourceStatus(str, Enum):
    AVAILABLE = "available"
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"

    @property
    def label(self) -> str:
        mapping = {
            ResourceStatus.AVAILABLE: "Disponible",
            ResourceStatus.NOT_DOWNLOADED: "No descargado",
            ResourceStatus.DOWNLOADING: "Descargando",
            ResourceStatus.ERROR: "Error",
            ResourceStatus.INCOMPATIBLE: "Incompatible",
        }
        return mapping[self]


@dataclass(frozen=True)
class ManagedResource:
    id: str
    kind: str  # "ambience" | "ir"
    environment_id: str
    name: str
    filename: str
    source: str
    author: str
    license: str
    source_url: str
    download_url: str | None
    sha256: str | None
    restrictions: str


class AmbienceManager:
    """Administra el catálogo de capas ambientales e IRs en ambience_data/."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._resources = self._load_manifests()

    def _load_manifests(self) -> dict[str, ManagedResource]:
        items: dict[str, ManagedResource] = {}

        amb_path = self.paths.resolve_resource("manifests", "ambience_manifest.json")
        if amb_path.is_file():
            raw = json.loads(amb_path.read_text(encoding="utf-8"))
            for r in raw.get("resources", []):
                res = ManagedResource(
                    id=str(r["id"]),
                    kind="ambience",
                    environment_id=str(r.get("environment_id", "general")),
                    name=str(r.get("name", r["id"])),
                    filename=str(r["filename"]),
                    source=str(r.get("source", "")),
                    author=str(r.get("author", "")),
                    license=str(r.get("license", "")),
                    source_url=str(r.get("source_url", "")),
                    download_url=r.get("download_url"),
                    sha256=str(r["sha256"]).lower() if r.get("sha256") else None,
                    restrictions=str(r.get("restrictions", "")),
                )
                items[res.id] = res

        ir_path = self.paths.resolve_resource("manifests", "ir_manifest.json")
        if ir_path.is_file():
            raw_ir = json.loads(ir_path.read_text(encoding="utf-8"))
            for r in raw_ir.get("impulse_responses", []):
                res = ManagedResource(
                    id=str(r["id"]),
                    kind="ir",
                    environment_id=str(r.get("environment_id", "general")),
                    name=str(r.get("name", r["id"])),
                    filename=str(r["filename"]),
                    source=str(r.get("source", "")),
                    author=str(r.get("author", "")),
                    license=str(r.get("license", "")),
                    source_url=str(r.get("source_url", "")),
                    download_url=r.get("download_url"),
                    sha256=str(r["sha256"]).lower() if r.get("sha256") else None,
                    restrictions=str(r.get("restrictions", "")),
                )
                items[res.id] = res

        return items

    def get_resource(self, resource_id: str) -> ManagedResource | None:
        return self._resources.get(resource_id)

    def resolve_local_path(self, resource: ManagedResource) -> Path:
        if resource.kind == "ir":
            return self.paths.ambience_dir / "ir" / resource.filename
        return self.paths.ambience_dir / resource.environment_id / resource.filename

    def get_status(self, resource_id: str) -> ResourceStatus:
        res = self._resources.get(resource_id)
        if res is None:
            return ResourceStatus.INCOMPATIBLE
        local = self.resolve_local_path(res)
        part = local.with_name(f"{local.name}.part")
        if part.is_file():
            return ResourceStatus.ERROR
        if local.is_file() and local.stat().st_size > 0:
            return ResourceStatus.AVAILABLE
        return ResourceStatus.NOT_DOWNLOADED

    def list_summaries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for res in self._resources.values():
            st = self.get_status(res.id)
            kind_lbl = f"{'IR' if res.kind == 'ir' else 'Ambiente'} ({res.environment_id})"
            out.append(
                {
                    "id": res.id,
                    "name": res.name,
                    "kind": res.kind,
                    "kind_label": kind_lbl,
                    "source_author": f"{res.source} — {res.author}",
                    "license": res.license,
                    "source_url": res.source_url,
                    "download_url": res.download_url,
                    "status_code": st.value,
                    "status_label": st.label,
                    "restrictions": res.restrictions,
                    "local_path": str(self.resolve_local_path(res)),
                }
            )
        return out

    def download_resource(
        self,
        resource_id: str,
        *,
        user_consented: bool,
        progress_cb: Callable[[float, str], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        if not user_consented:
            raise ResourceDownloadError(
                "Descarga rechazada: se requiere consentimiento expreso del usuario."
            )
        res = self.get_resource(resource_id)
        if res is None or not res.download_url:
            raise ResourceDownloadError(
                f"El recurso '{resource_id}' no tiene URL directa configurada.",
                details=res.restrictions if res else None,
            )

        dest = self.resolve_local_path(res)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(f"{dest.name}.part")
        existing = part.stat().st_size if part.is_file() else 0
        headers = {"User-Agent": "ZenTralAudio/0.1"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            with requests.get(res.download_url, headers=headers, stream=True, timeout=(15, 60)) as resp:
                resp.raise_for_status()
                resuming = resp.status_code == 206 and existing > 0
                mode = "ab" if resuming else "wb"
                downloaded = existing if resuming else 0
                total_len = int(resp.headers.get("Content-Length") or 0)
                total = (existing + total_len) if resuming and total_len > 0 else total_len
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=128 * 1024):
                        if cancel_token:
                            cancel_token.raise_if_cancelled()
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            frac = (downloaded / float(total)) if total > 0 else 0.5
                            progress_cb(min(0.95, frac), f"Descargando {res.name}...")
        except requests.RequestException as exc:
            raise ResourceDownloadError(f"Error descargando '{res.name}'.", details=str(exc)) from exc

        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            raise ResourceDownloadError(f"El recurso '{res.name}' resultó vacío.")

        if res.sha256:
            actual = compute_sha256(part)
            if actual != res.sha256.lower():
                part.unlink(missing_ok=True)
                raise IntegrityCheckError(f"SHA-256 inválido para '{res.filename}'.")
        else:
            logger.info("Recurso '%s' guardado sin hash SHA-256 oficial en origen.", res.filename)

        part.replace(dest)
        if progress_cb:
            progress_cb(1.0, f"{res.name} listo")
        return dest

    def delete_resource(self, resource_id: str) -> None:
        res = self.get_resource(resource_id)
        if res is None:
            return
        local = self.resolve_local_path(res)
        part = local.with_name(f"{local.name}.part")
        local.unlink(missing_ok=True)
        part.unlink(missing_ok=True)


def list_all_managed_resources(paths: AppPaths) -> list[dict[str, Any]]:
    """Punto de acceso usado por AppController para poblar el Gestor de Recursos."""
    return AmbienceManager(paths).list_summaries()
=======
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import requests

from core.exceptions import IntegrityCheckError, ResourceDownloadError
from core.job_manager import CancellationToken
from core.paths import AppPaths
from models.model_manager import compute_sha256

logger = logging.getLogger("zentralaudio.environments.ambience_manager")


class ResourceStatus(str, Enum):
    AVAILABLE = "available"
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"

    @property
    def label(self) -> str:
        mapping = {
            ResourceStatus.AVAILABLE: "Disponible",
            ResourceStatus.NOT_DOWNLOADED: "No descargado",
            ResourceStatus.DOWNLOADING: "Descargando",
            ResourceStatus.ERROR: "Error",
            ResourceStatus.INCOMPATIBLE: "Incompatible",
        }
        return mapping[self]


@dataclass(frozen=True)
class ManagedResource:
    id: str
    kind: str  # "ambience" | "ir"
    environment_id: str
    name: str
    filename: str
    source: str
    author: str
    license: str
    source_url: str
    download_url: str | None
    sha256: str | None
    restrictions: str


class AmbienceManager:
    """Administra el catálogo de capas ambientales e IRs en ambience_data/."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._resources = self._load_manifests()

    def _load_manifests(self) -> dict[str, ManagedResource]:
        items: dict[str, ManagedResource] = {}

        amb_path = self.paths.resolve_resource("manifests", "ambience_manifest.json")
        if amb_path.is_file():
            raw = json.loads(amb_path.read_text(encoding="utf-8"))
            for r in raw.get("resources", []):
                res = ManagedResource(
                    id=str(r["id"]),
                    kind="ambience",
                    environment_id=str(r.get("environment_id", "general")),
                    name=str(r.get("name", r["id"])),
                    filename=str(r["filename"]),
                    source=str(r.get("source", "")),
                    author=str(r.get("author", "")),
                    license=str(r.get("license", "")),
                    source_url=str(r.get("source_url", "")),
                    download_url=r.get("download_url"),
                    sha256=str(r["sha256"]).lower() if r.get("sha256") else None,
                    restrictions=str(r.get("restrictions", "")),
                )
                items[res.id] = res

        ir_path = self.paths.resolve_resource("manifests", "ir_manifest.json")
        if ir_path.is_file():
            raw_ir = json.loads(ir_path.read_text(encoding="utf-8"))
            for r in raw_ir.get("impulse_responses", []):
                res = ManagedResource(
                    id=str(r["id"]),
                    kind="ir",
                    environment_id=str(r.get("environment_id", "general")),
                    name=str(r.get("name", r["id"])),
                    filename=str(r["filename"]),
                    source=str(r.get("source", "")),
                    author=str(r.get("author", "")),
                    license=str(r.get("license", "")),
                    source_url=str(r.get("source_url", "")),
                    download_url=r.get("download_url"),
                    sha256=str(r["sha256"]).lower() if r.get("sha256") else None,
                    restrictions=str(r.get("restrictions", "")),
                )
                items[res.id] = res

        return items

    def get_resource(self, resource_id: str) -> ManagedResource | None:
        return self._resources.get(resource_id)

    def resolve_local_path(self, resource: ManagedResource) -> Path:
        if resource.kind == "ir":
            return self.paths.ambience_dir / "ir" / resource.filename
        return self.paths.ambience_dir / resource.environment_id / resource.filename

    def get_status(self, resource_id: str) -> ResourceStatus:
        res = self._resources.get(resource_id)
        if res is None:
            return ResourceStatus.INCOMPATIBLE
        local = self.resolve_local_path(res)
        part = local.with_name(f"{local.name}.part")
        if part.is_file():
            return ResourceStatus.ERROR
        if local.is_file() and local.stat().st_size > 0:
            return ResourceStatus.AVAILABLE
        return ResourceStatus.NOT_DOWNLOADED

    def list_summaries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for res in self._resources.values():
            st = self.get_status(res.id)
            kind_lbl = f"{'IR' if res.kind == 'ir' else 'Ambiente'} ({res.environment_id})"
            out.append(
                {
                    "id": res.id,
                    "name": res.name,
                    "kind": res.kind,
                    "kind_label": kind_lbl,
                    "source_author": f"{res.source} — {res.author}",
                    "license": res.license,
                    "source_url": res.source_url,
                    "download_url": res.download_url,
                    "status_code": st.value,
                    "status_label": st.label,
                    "restrictions": res.restrictions,
                    "local_path": str(self.resolve_local_path(res)),
                }
            )
        return out

    def download_resource(
        self,
        resource_id: str,
        *,
        user_consented: bool,
        progress_cb: Callable[[float, str], None] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> Path:
        if not user_consented:
            raise ResourceDownloadError(
                "Descarga rechazada: se requiere consentimiento expreso del usuario."
            )
        res = self.get_resource(resource_id)
        if res is None or not res.download_url:
            raise ResourceDownloadError(
                f"El recurso '{resource_id}' no tiene URL directa configurada.",
                details=res.restrictions if res else None,
            )

        dest = self.resolve_local_path(res)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(f"{dest.name}.part")
        existing = part.stat().st_size if part.is_file() else 0
        headers = {"User-Agent": "ZenTralAudio/0.1"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            with requests.get(res.download_url, headers=headers, stream=True, timeout=(15, 60)) as resp:
                resp.raise_for_status()
                resuming = resp.status_code == 206 and existing > 0
                mode = "ab" if resuming else "wb"
                downloaded = existing if resuming else 0
                total_len = int(resp.headers.get("Content-Length") or 0)
                total = (existing + total_len) if resuming and total_len > 0 else total_len
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=128 * 1024):
                        if cancel_token:
                            cancel_token.raise_if_cancelled()
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            frac = (downloaded / float(total)) if total > 0 else 0.5
                            progress_cb(min(0.95, frac), f"Descargando {res.name}...")
        except requests.RequestException as exc:
            raise ResourceDownloadError(f"Error descargando '{res.name}'.", details=str(exc)) from exc

        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            raise ResourceDownloadError(f"El recurso '{res.name}' resultó vacío.")

        if res.sha256:
            actual = compute_sha256(part)
            if actual != res.sha256.lower():
                part.unlink(missing_ok=True)
                raise IntegrityCheckError(f"SHA-256 inválido para '{res.filename}'.")
        else:
            logger.info("Recurso '%s' guardado sin hash SHA-256 oficial en origen.", res.filename)

        part.replace(dest)
        if progress_cb:
            progress_cb(1.0, f"{res.name} listo")
        return dest

    def delete_resource(self, resource_id: str) -> None:
        res = self.get_resource(resource_id)
        if res is None:
            return
        local = self.resolve_local_path(res)
        part = local.with_name(f"{local.name}.part")
        local.unlink(missing_ok=True)
        part.unlink(missing_ok=True)


def list_all_managed_resources(paths: AppPaths) -> list[dict[str, Any]]:
    """Punto de acceso usado por AppController para poblar el Gestor de Recursos."""
    return AmbienceManager(paths).list_summaries()

>>>>>>> 2f6beded7ad89e5c8d2dfbb7461aba8cf002655a