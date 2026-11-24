# ZenTralAudio

**ZenTralAudio** es una aplicación profesional de escritorio para Windows orientada a la restauración, mejora y contextualización acústica de grabaciones de voz mediante inteligencia artificial y procesamiento digital de señales (DSP).

> Estado actual: **Fase 2 — Base del proyecto** (arquitectura núcleo, rutas portables, configuración, logging, validación y gestor de tareas asíncronas).

---

## Características previstas en la arquitectura

1. **Restauración IA y DSP**:
   - Adaptadores modulares para modelos de restauración vocal (`MossFormer2_SE_48K`, `MossFormerGAN_SE_16K`, `FRCRN_SE_16K`, `DeepFilterNet3`, `VoiceFixer`, `GTCRN`) según disponibilidad y verificación técnica de cada implementación.
   - Presets profesionales adaptativos y control real de intensidad (0–100 %).
   - Opciones avanzadas generadas dinámicamente según las capacidades reales del modelo y los módulos DSP activos.
2. **Ambientes Acústicos**:
   - Simulación mediante convolución de Respuestas Impulsionales (IR) y reverberación algorítmica paramétrica.
   - Mezclador multicapa de ambientes con gestión transparente de licencias y recursos locales/descargables bajo consentimiento.
3. **Previsualización y Exportación coherentes**:
   - Reproductor permanente inferior con comparación instantánea `ON/OFF` (Original/Procesado) y `A/B`.
   - Cadena de procesamiento compartida para asegurar que el audio exportado (`WAV`, `FLAC`, `MP3`) coincida con la previsualización.
4. **Distribución Portable para Windows**:
   - Empaquetado mediante PyInstaller (`--onedir`) sin instalador tradicional ni dependencias de rutas absolutas.

---

## Requisitos del entorno de desarrollo

- **Sistema operativo**: Windows 10 / 11 (64-bit).
- **Python**: **3.11.x** (obligatorio por compatibilidad de wheels de audio e inferencia).
- **FFmpeg** (opcional en esta fase, necesario para MP3/M4A/AAC): ejecutable `ffmpeg.exe` en `bin/` o en el `PATH` del sistema.

---

## Instalación en desarrollo

```bash
git clone https://github.com/TU_USUARIO/ZenTralAudio.git
cd ZenTralAudio

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
