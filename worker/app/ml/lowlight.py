"""Adaptador SCI (Self-Calibrated Illumination) para fotos escuras.

SCI é um modelo leve de realce de iluminação que processa imagens escuras como
pré-processamento antes das demais etapas. Opera no espaço RGB normalizado [0,1].

Entrada: [1, 3, H, W] float32 — sem restrição de tamanho, mas processamos em tiles.
Saída: [1, 3, H, W] float32 no mesmo espaço.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ..classic.safe_math import to_float, to_uint8
from .session import OnnxModel

log = logging.getLogger("worker.lowlight")

# Threshold de brilho médio para aplicar o modelo (0-255)
DARK_THRESHOLD = 100


def is_dark(bgr: np.ndarray) -> bool:
    """Retorna True se a imagem for considerada escura."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < DARK_THRESHOLD


class OnnxLowlight:
    """Realce de iluminação com SCI via ONNX."""

    def __init__(self, model: OnnxModel, max_tile: int = 512) -> None:
        self.model = model
        self.max_tile = max_tile
        inputs = model.session.get_inputs()
        self._input_name = inputs[0].name
        # Verificar se a entrada é dinâmica ou fixa
        shape = inputs[0].shape  # [1, 3, H, W] ou [1, 3, None, None]
        self._fixed_h: int | None = None
        self._fixed_w: int | None = None
        if isinstance(shape[2], int) and shape[2] > 0:
            self._fixed_h = shape[2]
        if isinstance(shape[3], int) and shape[3] > 0:
            self._fixed_w = shape[3]

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        """Realça a iluminação de uma imagem escura."""
        rgb = cv2.cvtColor(to_float(bgr), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if self._fixed_h and self._fixed_w:
            # Modelo com tamanho fixo — redimensionar e restaurar
            orig_size = (w, h)
            rgb_resized = cv2.resize(rgb, (self._fixed_w, self._fixed_h),
                                     interpolation=cv2.INTER_LANCZOS4)
            out_rgb = self._run_patch(rgb_resized)
            out_rgb = cv2.resize(out_rgb, orig_size, interpolation=cv2.INTER_LANCZOS4)
        elif max(h, w) <= self.max_tile:
            # Imagem pequena — inferência direta
            out_rgb = self._run_patch(rgb)
        else:
            # Imagem grande — dividir em tiles com overlap
            out_rgb = self._run_tiled(rgb)

        return to_uint8(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    def _run_patch(self, rgb: np.ndarray) -> np.ndarray:
        """Inferência em um patch RGB float32 [0,1]."""
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        out = self.model.session.run(None, {self._input_name: tensor})[0]
        out = np.asarray(out, dtype=np.float32)[0].transpose(1, 2, 0)
        return np.clip(out, 0, 1)

    def _run_tiled(self, rgb: np.ndarray) -> np.ndarray:
        """Processa a imagem em tiles com overlap para evitar bordas visíveis."""
        h, w = rgb.shape[:2]
        tile = self.max_tile
        overlap = tile // 8
        step = tile - overlap

        result = np.zeros_like(rgb)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        # Ramp de fusão
        ramp = np.ones(tile, dtype=np.float32)
        ramp[:overlap] = np.linspace(0, 1, overlap)
        ramp[-overlap:] = np.linspace(1, 0, overlap)
        ramp_2d = np.outer(ramp, ramp)[:, :, None]

        for yi in range(0, h, step):
            for xi in range(0, w, step):
                y0, x0 = yi, xi
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                patch = rgb[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]

                if ph < 16 or pw < 16:
                    continue

                # Pad para tile × tile se necessário
                if ph < tile or pw < tile:
                    padded = np.pad(patch, ((0, tile - ph), (0, tile - pw), (0, 0)), "reflect")
                    out_padded = self._run_patch(padded)
                    out_patch = out_padded[:ph, :pw]
                    w_patch = ramp_2d[:ph, :pw]
                else:
                    out_patch = self._run_patch(patch)
                    w_patch = ramp_2d

                result[y0:y1, x0:x1] += out_patch * w_patch
                weight[y0:y1, x0:x1] += w_patch

        weight = np.maximum(weight, 1e-6)
        return np.clip(result / weight, 0, 1)
