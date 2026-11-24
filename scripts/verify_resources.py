"""Verifica la validez estructural de todos los manifiestos, presets y recursos visuales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths  # noqa: E402
from models.model_manager import ModelManager  # noqa: E402
from presets.preset_manager import PresetManager  # noqa: E402


def main() -> int:
    paths = AppPaths.detect()

    required_files = [
        paths.resolve_resource("icons", "zentralaudio.svg"),
        paths.resolve_resource("styles", "zentral_dark.qss"),
        paths.resolve_resource("manifests", "models_manifest.json"),
        paths.resolve_resource("manifests", "ambience_manifest.json"),
        paths.resolve_resource("manifests", "ir_manifest.json"),
    ]
    for rf in required_files:
        if not rf.is_file():
            print(f"ERROR: Falta recurso obligatorio: {rf}", file=sys.stderr)
            return 1
        if rf.suffix.lower() == ".json":
            json.loads(rf.read_text(encoding="utf-8"))

    mm = ModelManager(paths)
    entries = mm.list_entries()
    if len(entries) != 6:
        print(f"ERROR: Se esperaban 6 modelos en manifiesto, encontrados {len(entries)}", file=sys.stderr)
        return 1

    pm = PresetManager(paths)
    if len(pm.list_all()) < 9:
        print("ERROR: Faltan presets integrados en presets/restoration/builtin_presets.json", file=sys.stderr)
        return 1

    print("Manifiestos, estilos, iconos y presets verificados correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
