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

    run = functions["_load_voicefixer"]
    statements = [ast.unparse(stmt) for stmt in run.body]
    import_index = next(i for i, text in enumerate(statements) if "from voicefixer import" in text)
    link_indexes = [i for i, text in enumerate(statements) if text.startswith("_link_or_copy(")]
    assert link_indexes and max(link_indexes) < import_index


@pytest.mark.unit
def test_clearvoice_receives_pcm16_input() -> None:
    source = (ROOT / "models" / "isolated_runners" / "worker_entry.py").read_text(encoding="utf-8")
    assert 'subtype="PCM_16"' in source


@pytest.mark.unit
@pytest.mark.parametrize("length", [500, 16000 * 3, 16000 * 7 + 123, 16000 * 31])
def test_chunked_processing_reconstructs_signal_exactly(length: int) -> None:
    import numpy as np

    from models.chunking import process_in_chunks

    x = np.random.default_rng(length).standard_normal(length).astype(np.float32)
    calls: list[int] = []

    def identity(segment):
        calls.append(segment.shape[0])
        assert segment.shape[0] <= 16000 * 3
        return segment * 2.0

    out = process_in_chunks(x, 16000, chunk_seconds=3.0, overlap_seconds=0.5, infer=identity)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, x * 2.0, atol=1e-5)
    if length > 16000 * 3:
        assert len(calls) > 1 and min(calls) == 16000 * 3


@pytest.mark.unit
def test_persistent_worker_serves_multiple_requests_and_restarts(tmp_path: Path) -> None:
    """Un worker falso con el mismo protocolo: varias solicitudes, error y reinicio."""
    import sys

    import numpy as np

    from engine.audio_types import AudioBuffer
    from models.isolated_runners import subprocess_runner as runner

    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, os, sys\n"
        "import soundfile as sf\n"
        "P='@@ZENTRAL '\n"
        "print('ruido informativo'); print(P + json.dumps({'event': 'ready', 'pid': os.getpid()}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    if req.get('cmd') == 'exit': break\n"
        "    if req['params'].get('fail'):\n"
        "        print(P + json.dumps({'request_id': req['request_id'], 'ok': False, 'error': 'boom'}), flush=True); continue\n"
        "    d, sr = sf.read(req['input_wav'], dtype='float32')\n"
        "    sf.write(req['output_wav'], d * 0.5, sr, subtype='FLOAT')\n"
        "    print(P + json.dumps({'request_id': req['request_id'], 'ok': True, 'pid': os.getpid()}), flush=True)\n",
        encoding="utf-8",
    )
    weights = tmp_path / "data" / "models_data" / "m"
    weights.mkdir(parents=True)
    buf = AudioBuffer(samples=np.ones((1, 1600), np.float32), sample_rate=16000)
    try:
        for _ in range(3):
            out = runner.run_in_isolated_subprocess(
                Path(sys.executable), script, buf, "m", weights, "cpu", {}
            )
            np.testing.assert_allclose(out.samples, 0.5)
        workers = list(runner._WORKERS.values())
        assert len(workers) == 1 and workers[0].alive
        pid = workers[0]._proc.pid

        with pytest.raises(Exception, match="Fallo"):
            runner.run_in_isolated_subprocess(Path(sys.executable), script, buf, "m", weights, "cpu", {"fail": 1})
        assert workers[0].alive and workers[0]._proc.pid == pid  # un error no mata el worker

        workers[0]._kill()
        out = runner.run_in_isolated_subprocess(Path(sys.executable), script, buf, "m", weights, "cpu", {})
        np.testing.assert_allclose(out.samples, 0.5)
        assert workers[0]._proc.pid != pid  # se reinicia solo
    finally:
        runner.shutdown_isolated_workers()


@pytest.mark.unit
def test_old_worker_without_serve_mode_falls_back_to_one_shot(tmp_path: Path) -> None:
    import sys

    import numpy as np

    from engine.audio_types import AudioBuffer
    from models.isolated_runners import subprocess_runner as runner

    script = tmp_path / "old_worker.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "import soundfile as sf\n"
        "req = json.loads(Path(sys.argv[1]).read_text())\n"
        "d, sr = sf.read(req['input_wav'], dtype='float32')\n"
        "sf.write(req['output_wav'], d * 0.25, sr, subtype='FLOAT')\n",
        encoding="utf-8",
    )
    weights = tmp_path / "data" / "models_data" / "m"
    weights.mkdir(parents=True)
    buf = AudioBuffer(samples=np.ones((1, 800), np.float32), sample_rate=16000)
    try:
        out = runner.run_in_isolated_subprocess(Path(sys.executable), script, buf, "m", weights, "cpu", {})
        np.testing.assert_allclose(out.samples, 0.25)
    finally:
        runner.shutdown_isolated_workers()
        runner._SERVER_DISABLED.clear()
