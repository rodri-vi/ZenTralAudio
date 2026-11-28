#!/bin/bash
# Doble clic en Finder para abrir ZenTralAudio desde el código fuente.
# (Para la aplicación empaquetada usa dist/ZenTralAudio.app o /Applications.)
exec "$(cd "$(dirname "$0")" && pwd)/scripts/run_macos.sh" "$@"
