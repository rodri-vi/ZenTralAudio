"""Herramienta CLI opcional para listar o descargar modelos IA con consentimiento explícito."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import AppPaths  # noqa: E402
from models.model_registry import ModelRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Gestor CLI de modelos IA de ZenTralAudio")
    parser.add_argument("--list", action="store_true", help="Muestra modelos, licencias y estado actual.")
    parser.add_argument("--model", type=str, help="Identificador del modelo a descargar.")
    parser.add_argument(
        "--consent",
        action="store_true",
        help="Confirma consentimiento explícito del usuario para descargar desde la fuente oficial.",
    )
    args = parser.parse_args()

    paths = AppPaths.detect()
    paths.ensure_writable_directories()
    registry = ModelRegistry(paths)

    if args.list or not args.model:
        for s in registry.list_ui_summaries():
            print(f"- {s['model_id']}: {s['display_name']} | Estado={s['status_label']} | Licencia={s['license_summary']}")
        return 0

    if not args.consent:
        print("Error: debes pasar --consent para autorizar expresamente la descarga de pesos.", file=sys.stderr)
        return 2

    res = registry.manager.download_model(
        args.model,
        user_consented=True,
        progress_cb=lambda p, msg: print(f"[{int(p * 100):3d}%] {msg}"),
    )
    print(f"Completado en: {res.target_dir}\nVerificación: {res.verification_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
