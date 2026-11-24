"""Catálogo verificado de los 14 ambientes acústicos y sus definiciones por defecto."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from core.paths import AppPaths

logger = logging.getLogger("zentralaudio.environments.catalog")


@dataclass(frozen=True)
class EnvironmentDefinition:
    environment_id: str
    name: str
    acoustics: dict[str, float]
    default_layers: list[dict[str, Any]]


class EnvironmentCatalog:
    """Lee presets base de los 14 ambientes desde presets/environments/builtin_environments.json."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._items = self._load()

    def _load(self) -> dict[str, EnvironmentDefinition]:
        candidates = (
            self.paths.presets_dir / "environments" / "builtin_environments.json",
            self.paths.bundle_root / "presets" / "environments" / "builtin_environments.json",
            Path(__file__).resolve().parent.parent / "presets" / "environments" / "builtin_environments.json",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            logger.warning(
                "No se encontró builtin_environments.json en ninguna ruta: %s",
                ", ".join(str(candidate) for candidate in candidates),
            )
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, EnvironmentDefinition] = {}
        for item in raw.get("environments", []):
            env = EnvironmentDefinition(
                environment_id=str(item["environment_id"]),
                name=str(item["name"]),
                acoustics={k: float(v) for k, v in (item.get("acoustics") or {}).items()},
                default_layers=[dict(l) for l in (item.get("default_layers") or [])],
            )
            out[env.environment_id] = env
        return out

    def get(self, environment_id: str) -> EnvironmentDefinition | None:
        return self._items.get(environment_id)

    def list_all(self) -> list[EnvironmentDefinition]:
        return list(self._items.values())
