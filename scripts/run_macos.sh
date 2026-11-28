#!/bin/bash
# Inicia ZenTralAudio desde el código fuente con el entorno .venv del proyecto.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No existe el entorno de ZenTralAudio. Ejecuta primero:  $ROOT/scripts/setup_macos.sh" >&2
  exit 1
fi
if ! "$PY" -c "import PySide6, numpy, scipy, sounddevice, soundfile, torch" 2>/dev/null; then
  echo "Faltan dependencias en .venv. Ejecuta:  $ROOT/scripts/setup_macos.sh" >&2
  "$PY" -c "import PySide6, numpy, scipy, sounddevice, soundfile, torch" || true
  exit 1
fi
# Comparte modelos, ajustes y runtime IA con ZenTralAudio.app
# (~/Library/Application Support/ZenTralAudio).
export ZENTRALAUDIO_MACOS_USER_DIRS="${ZENTRALAUDIO_MACOS_USER_DIRS:-1}"
cd "$ROOT"
exec "$PY" "$ROOT/main.py" "$@"
