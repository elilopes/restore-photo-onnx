"""Adaptador SwinIR para super-resolução.

Interface idêntica ao OnnxUpscaler (Real-ESRGAN) para funcionar como fallback.
SwinIR espera entrada [1, C, H, W] float32 normalizado [0,1] e retorna o mesmo
formato com H*scale × W*scale. Suporta inferência em tiles para imagens grandes.
"""
from __future__ import annotations

import logging

import numpy as np

from .session import OnnxModel
from .superres import OnnxUpscaler

log = logging.getLogger("worker.swinir")


class SwinIRUpscaler(OnnxUpscaler):
    """SwinIR: mesma interface que OnnxUpscaler, herda tiling e blending.

    A única diferença é que alguns checkpoints SwinIR expõem a entrada como
    'input' (em vez de 'x') e esperam padding para múltiplos de ``window_size``
    (tipicamente 8). Esta subclasse ajusta o padding se necessário.
    """

    def __init__(self, model: OnnxModel, scale: int, tile: int = 256,
                 overlap: int = 32, window_size: int = 8) -> None:
        super().__init__(model, scale, tile, overlap)
        self._window = window_size
        # Descobrir nome do input
        self._input_name = model.session.get_inputs()[0].name

    def _pad_to_window(self, patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Adiciona padding para que H e W sejam múltiplos de window_size."""
        _, _, h, w = patch.shape
        ph = (self._window - h % self._window) % self._window
        pw = (self._window - w % self._window) % self._window
        if ph == 0 and pw == 0:
            return patch, (0, 0)
        padded = np.pad(patch, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
        return padded, (ph, pw)

    def _run_patch(self, patch_nchw: np.ndarray) -> np.ndarray:
        """Executa inferência em um patch, com padding e recorte automático."""
        padded, (ph, pw) = self._pad_to_window(patch_nchw)
        out = self.model.session.run(None, {self._input_name: padded})[0]
        # Remover padding escalado
        s = self._scale
        if ph > 0 or pw > 0:
            out_h = out.shape[2] - ph * s
            out_w = out.shape[3] - pw * s
            out = out[:, :, :out_h, :out_w]
        return out
