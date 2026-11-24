# Licencias y atribuciones — ZenTralAudio Offline AI

Esta edición incluye pesos de modelos para permitir inferencia local sin conexión. Las licencias
del código y de los pesos no siempre coinciden.

| Modelo | Código / runtime | Pesos | Observaciones |
|---|---|---|---|
| DeepFilterNet3 | MIT | MIT / Apache-2.0 según el repositorio | Requiere DeepFilterLib para Windows x64. |
| MossFormer2_SE_48K | Apache-2.0 (ClearerVoice) | Apache-2.0 | El checkpoint principal tiene SHA-256 oficial; el archivo auxiliar no publica hash oficial. |
| MossFormerGAN_SE_16K | Apache-2.0 (ClearerVoice) | Apache-2.0 | El origen no publica SHA-256; se incluye hash local en el inventario del paquete. |
| FRCRN_SE_16K | Apache-2.0 (ClearerVoice) | Apache-2.0 | El origen no publica SHA-256; se incluye hash local en el inventario del paquete. |
| VoiceFixer | MIT | CC-BY-4.0 | Atribución obligatoria a Haohe Liu et al. |

## Atribución de VoiceFixer

Incluye el modelo VoiceFixer de Haohe Liu et al. Sus pesos proceden de Zenodo record 5600188
y se distribuyen bajo Creative Commons Attribution 4.0 International (CC-BY-4.0).
Se conserva la atribución y la referencia al registro original:
https://zenodo.org/records/5600188

## Fuentes oficiales

- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
- ClearerVoice-Studio: https://github.com/modelscope/ClearerVoice-Studio
- MossFormer2_SE_48K: https://huggingface.co/alibabasglab/MossFormer2_SE_48K
- MossFormerGAN_SE_16K: https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K
- FRCRN_SE_16K: https://huggingface.co/alibabasglab/FRCRN_SE_16K
- VoiceFixer: https://github.com/haoheliu/voicefixer
- VoiceFixer weights: https://zenodo.org/records/5600188

## Modelo no incluido

GTCRN permanece deshabilitado porque el proyecto actual no dispone de una descarga automatizada
con nombres de checkpoints, pesos y runtime verificados. No se presenta como funcional.
