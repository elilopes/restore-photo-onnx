"""Rastreia o progresso do download dos modelos ONNX em background."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelDownloadState:
    name: str
    status: Literal["pending", "downloading", "done", "skipped", "error"] = "pending"
    size_total: int = 0
    size_done: int = 0
    error: str | None = None

    @property
    def percent(self) -> int:
        if self.size_total <= 0:
            return 0
        return min(100, round(self.size_done * 100 / self.size_total))


class DownloadTracker:
    """Thread-safe: lido pelo endpoint GET /models/download/status enquanto o thread baixa."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._models: list[ModelDownloadState] = []
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Escrita (thread de download)
    # ------------------------------------------------------------------

    def start(self, names: list[str]) -> None:
        with self._lock:
            self._running = True
            self._error = None
            self._models = [ModelDownloadState(n) for n in names]

    def update(self, name: str, *, status: str, size_total: int = 0, size_done: int = 0, error: str | None = None) -> None:
        with self._lock:
            for m in self._models:
                if m.name == name:
                    m.status = status  # type: ignore[assignment]
                    if size_total:
                        m.size_total = size_total
                    if size_done:
                        m.size_done = size_done
                    if error is not None:
                        m.error = error
                    break

    def finish(self, global_error: str | None = None) -> None:
        with self._lock:
            self._running = False
            self._error = global_error

    # ------------------------------------------------------------------
    # Leitura (endpoint HTTP)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "models": [
                    {
                        "name": m.name,
                        "status": m.status,
                        "size_total": m.size_total,
                        "size_done": m.size_done,
                        "percent": m.percent,
                        "error": m.error,
                    }
                    for m in self._models
                ],
            }

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running
