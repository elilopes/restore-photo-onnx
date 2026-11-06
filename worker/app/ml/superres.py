"""Adaptador de super-resolução (Real-ESRGAN x4 em ONNX).

Entrada NCHW RGB em [0,1]; saída NCHW RGB em [0,1] com escala `native_scale`
(4 para o x4plus). Se o modelo tiver entrada fixa, o tamanho do bloco vem dele;
senão usa `default_tile`. Escalas pedidas diferentes da nativa (ex.: 2x com
modelo 4x) são obtidas redimensionando a saída.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..classic.safe_math import to_float, to_uint8
from .session import OnnxModel
from .tiled import tiled_apply


class OnnxUpscaler:
    def __init__(self, model: OnnxModel, native_scale: int = 4, default_tile: int = 256, overlap: int = 16) -> None:
        specs = model.inputs()
        if len(specs) != 1 or specs[0].channels != 3:
            raise RuntimeError(f"upscaler espera 1 entrada [1,3,H,W]; encontrado: {specs}")
        self.model = model
        self.input_name = specs[0].name
        self.native_scale = native_scale
        self.tile = specs[0].height or default_tile
        if specs[0].height and specs[0].width and specs[0].height != specs[0].width:
            raise RuntimeError("apenas entradas quadradas são suportadas")
        self.overlap = min(overlap, self.tile // 4)

    def _infer(self, rgb_tile: np.ndarray, _aux: np.ndarray | None) -> np.ndarray:
        tensor = np.ascontiguousarray(rgb_tile.transpose(2, 0, 1)[None])
        out = np.asarray(self.model.run({self.input_name: tensor})[0], dtype=np.float32)[0]
        return out.transpose(1, 2, 0)

    def upscale(self, bgr: np.ndarray, target_scale: int) -> np.ndarray:
        rgb = cv2.cvtColor(to_float(bgr), cv2.COLOR_BGR2RGB)
        result = tiled_apply(rgb, self.tile, self.overlap, self.native_scale, self._infer)
        assert result is not None
        up, _ = result
        out = cv2.cvtColor(to_uint8(up), cv2.COLOR_RGB2BGR)
        if target_scale != self.native_scale:
            h, w = bgr.shape[:2]
            size = (w * target_scale, h * target_scale)
            interp = cv2.INTER_AREA if target_scale < self.native_scale else cv2.INTER_LANCZOS4
            out = cv2.resize(out, size, interpolation=interp)
        return out
