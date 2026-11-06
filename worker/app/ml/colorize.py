"""Adaptador de colorização (DDColor em ONNX).

Modelo de referência: edgetools/ddcolor `ddcolor-tiny-fp16.onnx`
(entrada float32 [1,3,512,512], saída float32 [1,2,512,512] = canais a/b do Lab).

Fluxo (o mesmo do DDColor original):
  1. L (Lab, resolução original) -> redimensiona para SxS;
  2. monta um RGB "cinza" a partir de L com a=b=0 (Lab -> RGB), valores em [0,1];
  3. o modelo prevê só a/b; redimensiona a/b para o tamanho original;
  4. junta com o L ORIGINAL, preservando todo o detalhe de luminância.

Outros colorizadores (ex.: DeOldify) usam outro pré/pós-processamento e
precisam de um adaptador próprio.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..classic.safe_math import to_float, to_uint8
from .session import OnnxModel


class OnnxColorizer:
    def __init__(self, model: OnnxModel, default_size: int = 512) -> None:
        specs = model.inputs()
        if len(specs) != 1 or specs[0].channels != 3:
            raise RuntimeError(f"colorizador espera 1 entrada [1,3,S,S]; encontrado: {specs}")
        self.model = model
        self.input_name = specs[0].name
        self.size = specs[0].height or default_size

    def colorize(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        lab = cv2.cvtColor(to_float(bgr), cv2.COLOR_BGR2Lab)
        lum = lab[..., :1]

        lum_small = cv2.resize(lum, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if lum_small.ndim == 2:
            lum_small = lum_small[..., None]
        gray_lab = np.concatenate([lum_small, np.zeros_like(lum_small), np.zeros_like(lum_small)], axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_Lab2RGB)
        tensor = np.ascontiguousarray(gray_rgb.transpose(2, 0, 1)[None].astype(np.float32))

        ab = np.asarray(self.model.run({self.input_name: tensor})[0], dtype=np.float32)[0].transpose(1, 2, 0)
        if ab.shape[2] != 2:
            raise RuntimeError(f"saída do colorizador com {ab.shape[2]} canais; esperado 2 (a/b)")
        ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

        out_lab = np.concatenate([lum, ab], axis=-1).astype(np.float32)
        out_lab[..., 1:] = np.clip(out_lab[..., 1:], -127.0, 127.0)
        return to_uint8(cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR))
