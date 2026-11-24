# ZenTralAudio

**ZenTralAudio** es una aplicación profesional de escritorio para Windows (portable) diseñada para restaurar, mejorar y contextualizar grabaciones de voz mediante inteligencia artificial y procesamiento digital de señales (DSP).

---

## Características principales

1. **Pestaña 1 — RESTAURACIÓN IA**:
   - Carga por explorador o *drag & drop* (`WAV`, `FLAC`, `AIFF`, `OGG`, y `MP3`/`M4A`/`AAC` mediante `FFmpeg`).
   - Visualización de forma de onda interactiva (cursor y selección de fragmento) y espectrograma STFT real en dBFS.
   - Adaptadores modulares para modelos IA de restauración vocal con verificación explícita de estado, licencias y descargas bajo consentimiento.
   - 9 presets profesionales adaptativos (`Studio Voice`, `Podcast Pro`, `Noise Removal`, `Heavy Noise`, `Echo Removal`, `Old Recording`, `Telephone Voice`, `Natural Voice`, `Maximum Recovery`) + guardado de presets personalizados.
   - Control real de intensidad (0–100 %) mediante mezcla *dry/wet* alineada y parámetros nativos.
   - Panel **OPCIONES AVANZADAS** generado dinámicamente según las capacidades reales del modelo y los módulos DSP implementados (`Reducción de ruido`, `Ecualización`, `Dinámica`, `Tratamiento vocal`, `Tratamiento espacial`).

2. **Pestaña 2 — AMBIENTES ACÚSTICOS**:
   - 14 entornos configurados (`Bosque`, `Playa`, `Calle`, `WC o baño`, `Oficina`, `Restaurante`, `Habitación`, `Montaña`, `Estación`, `Cafetería`, `Iglesia`, `Pasillo`, `Exterior urbano`, `Estudio de grabación`).
   - Simulación acústica mediante convolución de Respuestas Impulsionales (IR) reales o motor algorítmico paramétrico explícito.
   - Mezclador multicapa con activación independiente, volumen, panorámica, `Solo`, carga de archivos locales y gestor transparente de licencias y recursos.

3. **Reproductor Permanente y Exportación Coherente**:
   - Transporte completo (`Play`, `Pause`, `Stop`, barra de tiempo, `Loop`, fragmento, medidor RMS/pico).
   - Comparación instantánea **PROCESADO ON / OFF** (Original vs Procesado) con *crossfade* suave y compensación opcional de sonoridad, además de comparación **A / B**.
   - **Paridad garantizada**: previsualización y exportación (`WAV`, `FLAC`, `MP3`) utilizan exactamente la misma instancia `ProcessingPipeline` y configuración serializable.

---

## Estado real y verificación de modelos IA

ZenTralAudio **no simula operaciones de IA** ni inventa pesos o parámetros:

| Modelo | Frecuencia | Licencia (Código / Pesos) | Estado y Notas |
| :--- | :--- | :--- | :--- |
| **DeepFilterNet3** | 48.000 Hz | MIT / Apache-2.0 | Fuente, ZIP fijado y SHA-256 oficial verificados. Requiere `deepfilternet==0.5.6` (`numpy<2.0`). |
| **MossFormer2_SE_48K** | 48.000 Hz | Apache-2.0 / Apache-2.0 | Checkpoint `last_best_checkpoint.pt` y SHA-256 oficial verificados. Segmentación solapada integrada; requiere `clearvoice` o entorno aislado `.venv`. |
| **MossFormerGAN_SE_16K** | 16.000 Hz | Apache-2.0 / Apache-2.0 | Fuente verificada (techo físico 8 kHz; sin SHA-256 oficial publicado en origen). Requiere `clearvoice` o `.venv` aislado. |
| **FRCRN_SE_16K** | 16.000 Hz | Apache-2.0 / Apache-2.0 | Fuente verificada (techo físico 8 kHz; sin SHA-256 oficial publicado en origen). Requiere `clearvoice` o `.venv` aislado. |
| **VoiceFixer** | 44.100 Hz | MIT / **CC-BY-4.0** (Zenodo 5600188) | Ambos checkpoints y SHA-256 oficiales verificados. Requiere atribución CC-BY-4.0; recomendado en `.venv` aislado. |
| **GTCRN** | 16.000 Hz | MIT | Marcado explícitamente como **Incompatible / Pendiente de verificación** hasta confirmar nombres de checkpoints y paquete Python oficial en `Xiaobin-Rong/gtcrn`. |

La aplicación arranca y funciona completamente desde el primer momento con la cadena DSP y simulación acústica incluso sin haber descargado modelos IA.

---

## Instalación en desarrollo (Windows / Python 3.11)

```bash
git clone https://github.com/TU_USUARIO/ZenTralAudio.git
cd ZenTralAudio

py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
