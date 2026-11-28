"""Gestor de tareas asíncronas con cancelación cooperativa, reemplazo de obsoletos y exclusión."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from core.exceptions import JobCancelledError

logger = logging.getLogger("zentralaudio.jobs")

T = TypeVar("T")


class JobStatus(str, Enum):
    """Estados del ciclo de vida de una tarea en segundo plano."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CancellationToken:
    """Token seguro entre hilos para cancelar trabajos de larga duración."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelledError("Operación cancelada por el usuario o reemplazada por una nueva.")


@dataclass
class JobContext:
    """Contexto inyectado en cada función de trabajo ejecutada por JobManager."""

    job_id: str
    token: CancellationToken
    _progress_callback: Callable[[float, str], None] | None = None

    def check_cancelled(self) -> None:
        self.token.raise_if_cancelled()

    def report_progress(self, fraction: float, message: str = "") -> None:
        self.check_cancelled()
        clamped = max(0.0, min(1.0, float(fraction)))
        if self._progress_callback is not None:
            self._progress_callback(clamped, message)


@dataclass
class JobHandle(Generic[T]):
    """Referencia observable a una tarea enviada al gestor."""

    job_id: str
    name: str
    token: CancellationToken
    group_key: str | None = None
    resource_key: str | None = None
    status: JobStatus = JobStatus.PENDING
    error: BaseException | None = None
    result: T | None = None
    _future: Future[T] | None = field(default=None, repr=False)
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        """Solicita la cancelación cooperativa de esta tarea."""
        self.token.cancel()
        if self._future is not None and self._future.cancel():
            self.status = JobStatus.CANCELLED
            self._done_event.set()

    def wait(self, timeout: float | None = None) -> T:
        """Espera de forma bloqueante (útil en tests o cierre) y devuelve el resultado."""
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(f"La tarea '{self.name}' ({self.job_id}) superó el tiempo de espera.")
        if self.status == JobStatus.CANCELLED:
            raise JobCancelledError(f"La tarea '{self.name}' fue cancelada.")
        if self.error is not None:
            raise self.error
        return self.result  # type: ignore[return-value]


class JobManager:
    """Ejecutor concurrente con cancelación de trabajos obsoletos y bloqueo por recurso.

    Características clave:
    - `group_key`: si se envía un trabajo nuevo con el mismo `group_key` (por ejemplo,
      'preview_render'), los trabajos anteriores pendientes o activos de ese grupo se
      marcan como cancelados automáticamente.
    - `resource_key`: garantiza que dos trabajos incompatibles no usen simultáneamente
      el mismo recurso exclusivo (por ejemplo, un modelo IA o dispositivo CUDA).
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="ZenTralWorker",
        )
        self._lock = threading.Lock()
        self._active_handles: dict[str, JobHandle[Any]] = {}
        self._group_latest: dict[str, JobHandle[Any]] = {}
        self._resource_locks: dict[str, threading.Lock] = {}
        self._shutdown = False

    def submit(
        self,
        name: str,
        fn: Callable[[JobContext], T],
        *,
        group_key: str | None = None,
        resource_key: str | None = None,
        replace_group: bool = True,
        on_progress: Callable[[float, str], None] | None = None,
        on_completed: Callable[[T], None] | None = None,
        on_failed: Callable[[BaseException], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
    ) -> JobHandle[T]:
        """Envía una tarea al pool en segundo plano sin bloquear el hilo principal."""
        previous_to_cancel: JobHandle[Any] | None = None
        with self._lock:
            if self._shutdown:
                raise RuntimeError("No se pueden enviar tareas: JobManager está cerrado.")

            if group_key and replace_group:
                previous = self._group_latest.get(group_key)
                if previous is not None and previous.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    logger.debug("Cancelando tarea obsoleta '%s' (%s)", previous.name, previous.job_id)
                    # Future.cancel() puede ejecutar sus callbacks inmediatamente.
                    # Nunca cancelar dentro de _lock: el callback necesita adquirir
                    # este mismo lock para retirar la tarea terminada.
                    previous_to_cancel = previous

            job_id = uuid.uuid4().hex[:12]
            token = CancellationToken()
            handle: JobHandle[T] = JobHandle(
                job_id=job_id,
                name=name,
                token=token,
                group_key=group_key,
                resource_key=resource_key,
            )
            self._active_handles[job_id] = handle
            if group_key:
                self._group_latest[group_key] = handle

            resource_lock = None
            if resource_key:
                resource_lock = self._resource_locks.setdefault(resource_key, threading.Lock())

        # Cancelar fuera del lock evita un deadlock cuando Future.cancel()
        # dispara _done_callback de forma síncrona para tareas que siguen en cola.
        if previous_to_cancel is not None:
            previous_to_cancel.cancel()

        context = JobContext(job_id=job_id, token=token, _progress_callback=on_progress)

        def _runner() -> T:
            if token.is_cancelled:
                raise JobCancelledError(f"Tarea '{name}' cancelada antes de iniciar.")

            if resource_lock is not None:
                # Esperar de forma cancelable a que quede libre el recurso exclusivo
                while not resource_lock.acquire(timeout=0.05):
                    token.raise_if_cancelled()

            try:
                token.raise_if_cancelled()
                with self._lock:
                    handle.status = JobStatus.RUNNING
                logger.debug("Iniciando tarea '%s' (%s)", name, job_id)
                result = fn(context)
                token.raise_if_cancelled()
                return result
            finally:
                if resource_lock is not None:
                    resource_lock.release()

        def _done_callback(future: Future[T]) -> None:
            try:
                if future.cancelled():
                    with self._lock:
                        handle.status = JobStatus.CANCELLED
                    if on_cancelled:
                        on_cancelled()
                    return

                exc = future.exception()
                if exc is None and handle.token.is_cancelled:
                    # Cubre la carrera en la que el token se cancela justo después
                    # del return del worker y antes de ejecutar este callback.
                    with self._lock:
                        handle.status = JobStatus.CANCELLED
                    logger.debug("Tarea cancelada '%s' (%s)", name, job_id)
                    if on_cancelled:
                        on_cancelled()
                    return

                if exc is not None:
                    if isinstance(exc, JobCancelledError):
                        with self._lock:
                            handle.status = JobStatus.CANCELLED
                        logger.debug("Tarea cancelada '%s' (%s)", name, job_id)
                        if on_cancelled:
                            on_cancelled()
                    else:
                        with self._lock:
                            handle.status = JobStatus.FAILED
                            handle.error = exc
                        logger.exception("Error en tarea '%s' (%s)", name, job_id, exc_info=exc)
                        if on_failed:
                            on_failed(exc)
                    return

                res = future.result()
                with self._lock:
                    handle.status = JobStatus.COMPLETED
                    handle.result = res
                logger.debug("Tarea completada '%s' (%s)", name, job_id)
                if on_completed:
                    on_completed(res)
            finally:
                with self._lock:
                    self._active_handles.pop(job_id, None)
                handle._done_event.set()

        future = self._executor.submit(_runner)
        handle._future = future
        future.add_done_callback(_done_callback)
        return handle

    def cancel_group(self, group_key: str) -> bool:
        """Cancela la última tarea activa de un grupo y señala si había algo que cancelar."""
        with self._lock:
            handle = self._group_latest.get(group_key)
            active = handle is not None and handle.status in (JobStatus.PENDING, JobStatus.RUNNING)
        if not active or handle is None:
            return False
        # Cancelar fuera del lock: Future.cancel() puede ejecutar callbacks síncronos.
        handle.cancel()
        return True

    def cancel_all(self) -> None:
        """Solicita cancelación de todas las tareas activas o en cola."""
        with self._lock:
            handles = list(self._active_handles.values())
        for handle in handles:
            handle.cancel()

    def active_count(self) -> int:
        with self._lock:
            return len(self._active_handles)

    def shutdown(self, *, wait: bool = True) -> None:
        """Cancela trabajos pendientes y libera el pool de hilos de forma segura."""
        with self._lock:
            self._shutdown = True
        self.cancel_all()
        self._executor.shutdown(wait=wait, cancel_futures=True)
