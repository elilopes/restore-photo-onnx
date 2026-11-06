"""Progresso em memória por job (consultado pelo worker Node via GET /progress/{id})."""
from __future__ import annotations

import threading
import time


class ProgressTracker:
    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._data: dict[str, tuple[float, str, float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def update(self, job_id: str, progress: float, step: str) -> None:
        with self._lock:
            self._data[job_id] = (min(max(progress, 0.0), 1.0), step, time.monotonic())
            self._purge_locked()

    def get(self, job_id: str) -> tuple[float, str] | None:
        with self._lock:
            entry = self._data.get(job_id)
            return (entry[0], entry[1]) if entry else None

    def finish(self, job_id: str) -> None:
        with self._lock:
            self._data.pop(job_id, None)

    def _purge_locked(self) -> None:
        limit = time.monotonic() - self._ttl
        for key in [k for k, v in self._data.items() if v[2] < limit]:
            del self._data[key]
