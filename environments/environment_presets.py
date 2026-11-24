"""Construccion del estado inicial de interfaz para cualquiera de los 14 ambientes."""

<<<<<<< HEAD
from __future__ import annotations

from typing import Any

from core.paths import AppPaths
from environments.ambience_manager import AmbienceManager, ResourceStatus
from environments.environment_catalog import EnvironmentCatalog


def build_environment_ui_state(
    environment_id: str,
    paths: AppPaths,
) -> tuple[dict[str, float], list[dict[str, Any]], str | None]:
    """Devuelve (parámetros acústicos, capas con su estado real, nombre de IR disponible)."""
    catalog = EnvironmentCatalog(paths)
    amb_manager = AmbienceManager(paths)
    env_def = catalog.get(environment_id)
    if env_def is None:
        return {}, [], None

    layers_ui: list[dict[str, Any]] = []
    for layer in env_def.default_layers:
        res_id = layer.get("resource_id")
        file_path: str | None = None
        filename: str | None = None
        status_text = "Sin archivo asignado"
        license_info = "Asigna un archivo WAV/FLAC/MP3 local con 'Archivo...'."

        if res_id:
            res = amb_manager.get_resource(str(res_id))
            if res is not None:
                st = amb_manager.get_status(res.id)
                local = amb_manager.resolve_local_path(res)
                if st == ResourceStatus.AVAILABLE:
                    file_path = str(local)
                    filename = res.filename
                    status_text = "Disponible"
                else:
                    status_text = st.label
                license_info = f"{res.source} | {res.license} | {res.restrictions}"

        layers_ui.append(
            {
                "layer_id": str(layer["layer_id"]),
                "name": str(layer["name"]),
                "resource_id": res_id,
                "file_path": file_path,
                "filename": filename,
                "status": status_text,
                "license_info": license_info,
                "enabled": True,
                "volume": float(layer.get("volume", 0.35)),
                "pan": float(layer.get("pan", 0.0)),
                "solo": False,
            }
        )

    return dict(env_def.acoustics), layers_ui, None
=======
from __future__ import annotations

from typing import Any

from core.paths import AppPaths
from environments.ambience_manager import AmbienceManager, ResourceStatus
from environments.environment_catalog import EnvironmentCatalog


def build_environment_ui_state(
    environment_id: str,
    paths: AppPaths,
) -> tuple[dict[str, float], list[dict[str, Any]], str | None]:
    """Devuelve (parámetros acústicos, capas con su estado real, nombre de IR disponible)."""
    catalog = EnvironmentCatalog(paths)
    amb_manager = AmbienceManager(paths)
    env_def = catalog.get(environment_id)
    if env_def is None:
        return {}, [], None

    layers_ui: list[dict[str, Any]] = []
    for layer in env_def.default_layers:
        res_id = layer.get("resource_id")
        file_path: str | None = None
        filename: str | None = None
        status_text = "Sin archivo asignado"
        license_info = "Asigna un archivo WAV/FLAC/MP3 local con 'Archivo...'."

        if res_id:
            res = amb_manager.get_resource(str(res_id))
            if res is not None:
                st = amb_manager.get_status(res.id)
                local = amb_manager.resolve_local_path(res)
                if st == ResourceStatus.AVAILABLE:
                    file_path = str(local)
                    filename = res.filename
                    status_text = "Disponible"
                else:
                    status_text = st.label
                license_info = f"{res.source} | {res.license} | {res.restrictions}"

        layers_ui.append(
            {
                "layer_id": str(layer["layer_id"]),
                "name": str(layer["name"]),
                "resource_id": res_id,
                "file_path": file_path,
                "filename": filename,
                "status": status_text,
                "license_info": license_info,
                "enabled": True,
                "volume": float(layer.get("volume", 0.35)),
                "pan": float(layer.get("pan", 0.0)),
                "solo": False,
            }
        )

    return dict(env_def.acoustics), layers_ui, None

>>>>>>> 2f6beded7ad89e5c8d2dfbb7461aba8cf002655a