"""Representación serializable, validable y determinista de la cadena de procesamiento."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.validation import validate_numeric_range


@dataclass
class PipelineConfig:
    """Especificación única compartida por PreviewEngine y ExportEngine."""

    model_id: str = "deepfilternet3"
    ai_enabled: bool = True
    preset_id: str = "studio_voice"
    intensity: int = 75
    advanced_params: dict[str, Any] = field(default_factory=dict)

    environment_enabled: bool = False
    environment_id: str = "estudio"
    ir_path: str | None = None
    acoustics: dict[str, float] = field(default_factory=dict)
    master_ambience_volume: float = 0.60
    ambience_layers: list[dict[str, Any]] = field(default_factory=list)

    true_peak_ceiling_db: float = -1.0
    loudness_match_preview: bool = True

    def validate(self) -> None:
        """Valida los límites reales de los parámetros antes de ejecutar audio."""
        self.intensity = int(
            validate_numeric_range(self.intensity, name="intensity", min_value=0, max_value=100)
        )
        self.master_ambience_volume = validate_numeric_range(
            self.master_ambience_volume,
            name="master_ambience_volume",
            min_value=0.0,
            max_value=1.5,
        )
        self.true_peak_ceiling_db = validate_numeric_range(
            self.true_peak_ceiling_db,
            name="true_peak_ceiling_db",
            min_value=-12.0,
            max_value=-0.1,
        )

        p = self.advanced_params
        if "eq_hp_freq" in p:
            validate_numeric_range(p["eq_hp_freq"], name="eq_hp_freq", min_value=20, max_value=1000)
        if "eq_lp_freq" in p:
            validate_numeric_range(p["eq_lp_freq"], name="eq_lp_freq", min_value=500, max_value=24000)
        if "comp_ratio" in p:
            validate_numeric_range(p["comp_ratio"], name="comp_ratio", min_value=1.0, max_value=20.0)
        if "stereo_width" in p:
            validate_numeric_range(p["stereo_width"], name="stereo_width", min_value=0.0, max_value=2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_controller_state(cls, state: dict[str, Any]) -> PipelineConfig:
        """Construye un PipelineConfig validado desde el diccionario del AppController."""
        env = state.get("environment") or {}
        safety = state.get("output_safety") or {}
        cfg = cls(
            model_id=str(state.get("model_id", "deepfilternet3")),
            ai_enabled=bool(state.get("ai_enabled", True)),
            preset_id=str(state.get("preset_id", "studio_voice")),
            intensity=int(state.get("intensity", 75)),
            advanced_params=dict(state.get("advanced_params") or {}),
            environment_enabled=bool(env.get("enabled", False)),
            environment_id=str(env.get("environment_id", "estudio")),
            ir_path=env.get("ir_path"),
            acoustics={k: float(v) for k, v in (env.get("acoustics") or {}).items()},
            master_ambience_volume=float(env.get("master_volume", 0.60)),
            ambience_layers=[dict(layer) for layer in (env.get("layers") or [])],
            true_peak_ceiling_db=float(safety.get("true_peak_ceiling_db", -1.0)),
            loudness_match_preview=bool(safety.get("loudness_match_preview", True)),
        )
        cfg.validate()
        return cfg

    def ai_stage_cache_key(self, source_hash: str) -> str:
        """Clave de caché exclusiva para la salida cruda del modelo IA."""
        model_specific = {
            k: v for k, v in self.advanced_params.items() if str(k).startswith("model_")
        }
        payload = {
            "source_hash": source_hash,
            "model_id": self.model_id,
            "ai_enabled": self.ai_enabled,
            "model_params": model_specific,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def voice_stage_cache_key(self, source_hash: str) -> str:
        """Clave del resultado vocal al 100 %, independiente de la intensidad Dry/Wet."""
        dsp_params = {
            key: value
            for key, value in self.advanced_params.items()
            if not str(key).startswith("model_") and key != "device"
        }
        payload = {
            "source_hash": source_hash,
            "ai_stage_key": self.ai_stage_cache_key(source_hash),
            "dsp_params": dsp_params,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def full_cache_key(self, source_hash: str) -> str:
        """Clave de caché completa de toda la cadena de procesamiento."""
        payload = {"source_hash": source_hash, "config": self.to_dict()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
