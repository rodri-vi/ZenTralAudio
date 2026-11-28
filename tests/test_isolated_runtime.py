"""Runtime IA aislado: descubrimiento, entorno sin red y rutas del worker."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from models.isolated_runners import subprocess_runner
from models.isolated_runners.subprocess_runner import find_isolated_python, shared_runtime_python

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.unit
def test_shared_runtime_is_found_for_every_model(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    model_dir = data_root / "models_data" / "frcrn_se_16k"
    model_dir.mkdir(parents=True)
    assert find_isolated_python(model_dir) is None

    python = shared_runtime_python(data_root)
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    assert find_isolated_python(model_dir) == python

    # Un .venv propio del modelo tiene prioridad sobre el runtime compartido.
    own = model_dir / ".venv" / "bin" / "python"
    own.parent.mkdir(parents=True)
    own.write_text("#!/bin/sh\n")
    assert find_isolated_python(model_dir) == own


@pytest.mark.unit
def test_worker_environment_is_offline_and_contained(tmp_path: Path) -> None:
    weights = tmp_path / "data" / "models_data" / "voicefixer"
    weights.mkdir(parents=True)
    env = subprocess_runner._portable_worker_environment(weights)
    assert env["HOME"].startswith(str((tmp_path / "data").resolve()))
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


@pytest.mark.unit
def test_voicefixer_adapter_points_to_existing_worker_script() -> None:
    source = (ROOT / "models" / "isolated_runners" / "voicefixer_adapter.py").read_text(encoding="utf-8")
    assert '"isolated_runners" / "worker_entry.py"' not in source
    assert (ROOT / "models" / "isolated_runners" / "worker_entry.py").is_file()


@pytest.mark.unit
def test_worker_never_imports_voicefixer_before_placing_local_weights() -> None:
    """``import voicefixer`` descarga ~600 MB si los pesos no están ya en su caché."""
    path = ROOT / "models" / "isolated_runners" / "worker_entry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    check = ast.unparse(functions["_check_runtime"])
    assert "import voicefixer" not in check

    run = functions["_run_voicefixer"]
    statements = [ast.unparse(stmt) for stmt in run.body]
    import_index = next(i for i, text in enumerate(statements) if "from voicefixer import" in text)
    link_indexes = [i for i, text in enumerate(statements) if text.startswith("_link_or_copy(")]
    assert link_indexes and max(link_indexes) < import_index


@pytest.mark.unit
def test_clearvoice_receives_pcm16_input() -> None:
    source = (ROOT / "models" / "isolated_runners" / "worker_entry.py").read_text(encoding="utf-8")
    assert 'subtype="PCM_16"' in source
