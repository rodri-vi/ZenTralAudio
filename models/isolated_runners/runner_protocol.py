"""Protocolo JSON limpio para invocar modelos IA en entornos Python aislados."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IsolatedRunRequest:
    """Solicitud serializable enviada a un proceso de inferencia aislado."""

    model_id: str
    input_wav: str
    output_wav: str
    weights_dir: str
    device: str
    params: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
