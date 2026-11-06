"""Adaptador de inpainting (LaMa em ONNX) para remover rasgos/riscos.

Modelo de referência: Carve/LaMa-ONNX `lama_fp32.onnx` (entrada fixa 512x512).
O adaptador descobre as entradas pelo número de canais (3 = imagem, 1 = máscara)
em vez de depender dos nomes, e detecta se a saída vem em 0..1 ou 0..255.
Confira com GET /models no primeiro uso do seu arquivo.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..classic.safe_math import to_float, to_uint8
from .session import OnnxModel
from .tiled import tiled_apply


class OnnxInpainter:
    def __init__(self, model: OnnxModel, default_tile: int = 512, overlap: int = 64) -> None:
        specs = model.inputs()
        image_spec = next((s for s in specs if s.channels == 3), None)
        mask_spec = next((s for s in specs if s.channels == 1), None)
        if image_spec is None or mask_spec is None:
            raise RuntimeError(f"modelo de inpainting sem entradas imagem(3ch)/máscara(1ch): {specs}")
        self.model = model
        self.image_name, self.mask_name = image_spec.name, mask_spec.name
        self.tile = image_spec.height or default_tile
        if image_spec.height and image_spec.width and image_spec.height != image_spec.width:
            raise RuntimeError("apenas entradas quadradas são suportadas")
        self.overlap = min(overlap, self.tile // 4)
        self._out_is_255: bool | None = None  # decidido no primeiro bloco

    def _infer(self, rgb_tile: np.ndarray, mask_tile: np.ndarray | None) -> np.ndarray:
        assert mask_tile is not None
        image = np.ascontiguousarray(rgb_tile.transpose(2, 0, 1)[None])
        mask = np.ascontiguousarray((mask_tile > 0.5).astype(np.float32)[None, None])
        out = self.model.run({self.image_name: image, self.mask_name: mask})[0]
        out = np.asarray(out, dtype=np.float32)[0].transpose(1, 2, 0)
        if self._out_is_255 is None:
            self._out_is_255 = float(np.nanmax(out)) > 2.0
        return out / 255.0 if self._out_is_255 else out

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """`mask` uint8 (0/255, 255 = preencher). Só os blocos que tocam a máscara são processados."""
        rgb = cv2.cvtColor(to_float(bgr), cv2.COLOR_BGR2RGB)
        mask_f = (mask > 0).astype(np.float32)
        result = tiled_apply(rgb, self.tile, self.overlap, 1, self._infer, aux=mask_f, skip_empty_aux=True)
        if result is None:
            return bgr.copy()
        filled, covered = result
        merged = np.where(covered[..., None], filled, rgb)
        return cv2.cvtColor(to_uint8(merged), cv2.COLOR_RGB2BGR)
