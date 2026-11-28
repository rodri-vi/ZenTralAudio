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
        "voicefixer",
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


@pytest.mark.unit
def test_deepfilternet_git_metadata_is_optional_when_git_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DeepFilterNet debe poder inicializarse en un portable sin git.exe."""
    import logging

    from models.deepfilternet_adapter import _make_git_root_optional

    def missing_git() -> str:
        raise FileNotFoundError(2, "The system cannot find the file specified", "git")

    safe_get_git_root = _make_git_root_optional(missing_git)
    with caplog.at_level(logging.INFO, logger="zentralaudio.models.deepfilternet3"):
        assert safe_get_git_root() is None

    assert "Git no está instalado" in caplog.text

    # Cuando Git existe y el proyecto está dentro de un repositorio, se conserva
    # el comportamiento original y se devuelve la raíz detectada.
    assert _make_git_root_optional(lambda: "C:/source/ZenTralAudio")() == "C:/source/ZenTralAudio"


@pytest.mark.unit
def test_deepfilternet_initialization_has_safe_stdout_and_stderr_without_console(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import sys

    from models.deepfilternet_adapter import _call_with_safe_stderr

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    def fake_library_initializer() -> str:
        assert sys.stdout is not None
        assert sys.stderr is not None
        sys.stdout.write("DeepFilterNet stdout initialized\\n")
        sys.stderr.write("DeepFilterNet stderr initialized\\n")
        return "ready"

    with caplog.at_level(logging.INFO, logger="zentralaudio.models.deepfilternet3"):
        assert _call_with_safe_stderr(fake_library_initializer) == "ready"

    assert sys.stdout is None
    assert sys.stderr is None
    assert "DeepFilterNet: DeepFilterNet stdout initialized" in caplog.text
    assert "DeepFilterNet: DeepFilterNet stderr initialized" in caplog.text
