"""Pruebas unitarias de concurrencia, cancelación y exclusión de recursos en JobManager."""

from __future__ import annotations

import time

import pytest

from core.exceptions import JobCancelledError
from core.job_manager import JobContext, JobManager


@pytest.mark.unit
def test_job_manager_supersedes_obsolete_group_jobs() -> None:
    jm = JobManager(max_workers=1)
    try:
        def slow_job(ctx: JobContext) -> str:
            for _ in range(20):
                ctx.check_cancelled()
                time.sleep(0.01)
            return "done_1"

        h1 = jm.submit("job1", slow_job, group_key="preview")
        h2 = jm.submit("job2", lambda ctx: "done_2", group_key="preview")

        assert h2.wait(timeout=5.0) == "done_2"
        with pytest.raises(JobCancelledError):
            h1.wait(timeout=5.0)
    finally:
        jm.shutdown()


@pytest.mark.unit
def test_replacing_pending_group_job_does_not_deadlock_and_cancels_old_job() -> None:
    """Cancelar un Future en cola no debe bloquear el lock interno del gestor."""
    import threading

    jm = JobManager(max_workers=1)
    release_blocker = threading.Event()
    blocker_started = threading.Event()
    cancelled_callbacks: list[str] = []

    def blocker(ctx: JobContext) -> str:
        blocker_started.set()
        if not release_blocker.wait(timeout=5.0):
            raise TimeoutError("El trabajo de bloqueo no recibió la señal de salida.")
        return "released"

    try:
        blocking = jm.submit("blocker", blocker)
        assert blocker_started.wait(timeout=2.0)

        old = jm.submit(
            "old_preview",
            lambda ctx: "stale",
            group_key="preview",
            on_cancelled=lambda: cancelled_callbacks.append("old"),
        )
        replacement = jm.submit(
            "new_preview",
            lambda ctx: "fresh",
            group_key="preview",
        )

        assert old.token.is_cancelled
        assert jm.cancel_group("preview") is True
        release_blocker.set()

        assert blocking.wait(timeout=3.0) == "released"
        # La orden de cancelación también alcanza la sustitución que seguía en cola.
        with pytest.raises(JobCancelledError):
            old.wait(timeout=3.0)
        with pytest.raises(JobCancelledError):
            replacement.wait(timeout=3.0)
        assert cancelled_callbacks == ["old"]
    finally:
        release_blocker.set()
        jm.shutdown()


@pytest.mark.unit
def test_job_manager_resource_lock_prevents_overlap() -> None:
    jm = JobManager(max_workers=2)
    active = 0
    max_concurrent = 0

    def exclusive_job(ctx: JobContext) -> int:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        time.sleep(0.05)
        active -= 1
        return 1

    try:
        h1 = jm.submit("r1", exclusive_job, resource_key="gpu_model")
        h2 = jm.submit("r2", exclusive_job, resource_key="gpu_model")
        assert h1.wait(timeout=5.0) + h2.wait(timeout=5.0) == 2
        assert max_concurrent == 1
    finally:
        jm.shutdown()
