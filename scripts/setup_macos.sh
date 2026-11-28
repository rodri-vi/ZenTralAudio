#!/bin/bash
# Prepara ZenTralAudio en macOS (Apple Silicon o Intel) sin Homebrew ni sudo.
#
#   scripts/setup_macos.sh                     # entorno de la app (.venv)
#   scripts/setup_macos.sh --with-ai           # + runtime IA aislado (4 modelos extra)
#   scripts/setup_macos.sh --import-models DIR # copia pesos ya descargados (models_data)
#
# - Usa uv (instalado desde PyPI en el usuario si falta) para obtener Python 3.11
#   nativo de la arquitectura real del Mac.
# - Los datos (modelos, ajustes, runtime IA) van a ~/Library/Application Support/ZenTralAudio,
#   compartidos entre el modo desarrollo y la ZenTralAudio.app empaquetada.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ZENTRALAUDIO_DATA_DIR:-$HOME/Library/Application Support/ZenTralAudio}"
WITH_AI=0
IMPORT_FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ai) WITH_AI=1 ;;
    --import-models) IMPORT_FROM="${2:?Falta la carpeta models_data}"; shift ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "Opción desconocida: $1" >&2; exit 2 ;;
  esac
  shift
done

ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" && "$(sysctl -in sysctl.proc_translated)" == "1" ]]; then
  echo "⚠️  Esta terminal se ejecuta bajo Rosetta en un Mac Apple Silicon."
  echo "    Abre Terminal sin 'Abrir con Rosetta' para obtener un entorno arm64 nativo."
  exit 1
fi
echo "▶ Mac $(sw_vers -productVersion) · arquitectura $ARCH"

find_uv() {
  command -v uv 2>/dev/null && return 0
  for candidate in "$HOME/.local/bin/uv" "$HOME/Library/Python/"*/bin/uv /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [[ -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}
UV="$(find_uv || true)"
if [[ -z "$UV" ]]; then
  echo "▶ Instalando uv (gestor de Python) para el usuario actual..."
  /usr/bin/python3 -m pip install --user --quiet uv
  UV="$(find_uv)"
fi
echo "▶ uv: $UV"

# El intérprete se guarda en la carpeta de datos: el runtime IA no depende de
# cachés externas ni del entorno de desarrollo.
export UV_PYTHON_INSTALL_DIR="$DATA/runtimes/python"
"$UV" python install --no-bin 3.11

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "▶ Creando entorno de la aplicación en $ROOT/.venv"
  "$UV" venv --python 3.11 "$ROOT/.venv"
fi
echo "▶ Instalando dependencias de la aplicación"
"$UV" pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements-macos.txt"

if [[ $WITH_AI -eq 1 ]]; then
  RT="$DATA/runtimes/ai-python"
  if [[ ! -x "$RT/bin/python" ]]; then
    echo "▶ Creando runtime IA aislado en $RT"
    "$UV" venv --python 3.11 "$RT"
  fi
  "$UV" pip install --python "$RT/bin/python" -r "$ROOT/requirements-ai-runtime.txt"
  "$RT/bin/python" "$ROOT/models/isolated_runners/worker_entry.py" --check-runtime
fi

if [[ -n "$IMPORT_FROM" ]]; then
  mkdir -p "$DATA/models_data"
  for model in deepfilternet3 mossformer2_se_48k mossformergan_se_16k frcrn_se_16k voicefixer; do
    if [[ -d "$IMPORT_FROM/$model" && ! -e "$DATA/models_data/$model" ]]; then
      echo "▶ Importando $model (clon APFS, sin duplicar espacio)"
      cp -cR "$IMPORT_FROM/$model" "$DATA/models_data/$model" 2>/dev/null || cp -R "$IMPORT_FROM/$model" "$DATA/models_data/$model"
    fi
  done
fi

echo "▶ Comprobación de arranque"
ZENTRALAUDIO_MACOS_USER_DIRS=1 "$ROOT/.venv/bin/python" "$ROOT/main.py" --check
ZENTRALAUDIO_MACOS_USER_DIRS=1 "$ROOT/.venv/bin/python" "$ROOT/scripts/check_environment.py" | tail -n 12
echo "✅ Listo. Abre la app con:  $ROOT/ZenTralAudio.command  (o doble clic en Finder)"
