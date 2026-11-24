"""Pruebas unitarias del registro de modelos, detección de estado e integridad de descargas."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from core.exceptions import IntegrityCheckError, ResourceDownloadError
from core.paths import AppPaths
from models.base_adapter import ModelStatus
from models.model_manager import ModelManager, _safe_extract_zip, compute_sha256
from models.model_registry import ModelRegistry


@pytest.mark.unit
def test_registry_lists_six_models_and_honest_uninstalled_status(portable_paths: AppPaths) -> None:
    registry = ModelRegistry(portable_paths)
    summaries = registry.list_ui_summaries()
    ids = {item["model_id"] for item in summaries}
    assert ids == {
        "deepfilternet3",
        "mossformer2_se_48k",
        "mossformergan_se_16k",
        "frcrn_se_16k",
        "gtcrn",
    }

    # Con models_data vacío, ninguno puede afirmarse como disponible/instalado
    for item in summaries:
        assert item["status_code"] in (ModelStatus.NOT_INSTALLED.value, ModelStatus.INCOMPATIBLE.value)

    gtcrn = registry.get_adapter("gtcrn")
    assert gtcrn is not None
    assert gtcrn.get_status() == ModelStatus.INCOMPATIBLE


@pytest.mark.unit
def test_download_requires_explicit_user_consent(portable_paths: AppPaths) -> None:
    manager = ModelManager(portable_paths)
    with pytest.raises(ResourceDownloadError):
        manager.download_model("deepfilternet3", user_consented=False)


@pytest.mark.unit
def test_sha256_and_zipslip_protection(tmp_path: Path) -> None:
    sample = tmp_path / "weights.bin"
    sample.write_bytes(b"zentralaudio-test-weights")
    digest = compute_sha256(sample)
    assert len(digest) == 64

    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../outside.txt", "blocked")

    dest = tmp_path / "extract"
    dest.mkdir()
    with pytest.raises(IntegrityCheckError):
        _safe_extract_zip(bad_zip, dest)
