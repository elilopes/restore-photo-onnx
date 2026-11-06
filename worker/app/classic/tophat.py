"""Top-Hat Transform para destacar defeitos finos (riscos, fissuras, poeira).

- White Top-Hat  = imagem - abertura(imagem)   -> estruturas CLARAS menores que o kernel
- Black Top-Hat  = fechamento(imagem) - imagem -> estruturas ESCURAS menores que o kernel

Como `abertura <= imagem <= fechamento`, as duas diferenças nunca ficam
negativas, então não há underflow em uint8. Mesmo assim, qualquer outra
subtração entre imagens deve usar `safe_math.saturating_sub`.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def top_hat_response(gray: np.ndarray, kernel_sizes: Sequence[int]) -> np.ndarray:
    """Resposta uint8 (0..255) = máximo entre Top-Hat branco e preto, em todas as escalas.

    Valores altos indicam estruturas finas e contrastantes (candidatas a defeito).
    A resposta é mantida em escala ABSOLUTA (sem normalizar) para que uma foto
    limpa não gere falsos positivos.
    """
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("top_hat_response espera uma imagem uint8 em tons de cinza")
    response = np.zeros_like(gray)
    for size in kernel_sizes:
        kernel = _ellipse(int(size))
        white = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        response = cv2.max(response, cv2.max(white, black))
    return response
