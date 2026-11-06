"""Sessão onnxruntime com carregamento preguiçoso e escolha de provedores (CUDA/CPU)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort


def select_providers(preferred: Sequence[str]) -> list[str]:
    """Filtra a lista preferida pelos provedores realmente disponíveis (sempre termina em CPU)."""
    available = set(ort.get_available_providers())
    chosen = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple  # dims int ou str/None quando dinâmicos
    channels: int | None
    height: int | None  # None => dinâmico
    width: int | None


def _static(dim) -> int | None:
    return dim if isinstance(dim, int) and dim > 0 else None


class OnnxModel:
    def __init__(self, path: Path, providers: Sequence[str], intra_threads: int = 0) -> None:
        self.path = path
        self._providers = list(providers)
        self._intra_threads = intra_threads
        self._session: ort.InferenceSession | None = None
        self._lock = threading.Lock()

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    if self._intra_threads > 0:
                        opts.intra_op_num_threads = self._intra_threads
                    self._session = ort.InferenceSession(
                        str(self.path), sess_options=opts, providers=self._providers
                    )
        return self._session

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def providers_in_use(self) -> list[str]:
        return self.session.get_providers()

    def inputs(self) -> list[InputSpec]:
        specs = []
        for node in self.session.get_inputs():
            shape = tuple(node.shape)
            nchw = len(shape) == 4
            specs.append(InputSpec(
                name=node.name,
                shape=shape,
                channels=_static(shape[1]) if nchw else None,
                height=_static(shape[2]) if nchw else None,
                width=_static(shape[3]) if nchw else None,
            ))
        return specs

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        # InferenceSession.run é thread-safe.
        return self.session.run(None, feeds)
