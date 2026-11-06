"""Máscara híbrida de defeitos (rasgos, riscos, fissuras).

"Híbrida" porque combina fontes diferentes:

1. Resposta do Top-Hat (estrutura fina e contrastante, em escala absoluta);
2. Adaptive Threshold sobre essa resposta (anomalia em relação à vizinhança,
   o que absorve manchas e iluminação irregular);
3. Histerese: componentes "fracos" só sobrevivem se contiverem um núcleo "forte";
4. Filtro de área (tira poeira minúscula e regiões grandes demais para serem riscos);
5. Máscara do usuário (opcional), unida à detectada.

Saídas:
- `binary`: 0/255, usada pelo inpainting;
- `soft`: float32 [0,1] com borda suave, usada para misturar o resultado com o
  original (pixels fora da máscara nunca são alterados).

Limitação conhecida: o Top-Hat só pega defeitos mais estreitos que o kernel.
Rasgos largos (perda de papel) precisam de máscara desenhada pelo usuário.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .safe_math import clip01
from .tophat import top_hat_response


@dataclass(frozen=True)
class MaskParams:
    kernel_sizes: tuple[int, ...]
    min_response: int      # piso absoluto da resposta do Top-Hat (0..255)
    strong_response: int   # resposta que qualifica um "núcleo forte"
    block_size: int        # janela do Adaptive Threshold (ímpar)
    adaptive_c: float      # C negativo => exige ficar C acima da média local
    min_area: int
    max_area_ratio: float
    dilate_px: int
    feather_sigma: float


@dataclass(frozen=True)
class HybridMask:
    binary: np.ndarray    # uint8 0/255
    soft: np.ndarray      # float32 [0,1]
    response: np.ndarray  # uint8, resposta do Top-Hat (debug)
    coverage: float       # fração da imagem coberta pela máscara
    components: int       # defeitos detectados (antes da dilatação)


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 else n + 1


def params_from_sensitivity(sensitivity: float, shape: tuple[int, ...]) -> MaskParams:
    """Converte sensibilidade [0,1] (maior = detecta mais) em parâmetros pelo tamanho da imagem."""
    s = float(np.clip(sensitivity, 0.0, 1.0))
    short = min(shape[0], shape[1])
    base = max(5.0, short * 0.004)
    min_response = int(round(40 - 28 * s))  # 40 -> 12
    return MaskParams(
        kernel_sizes=(_odd(base * 1.5), _odd(base * 3)),
        min_response=min_response,
        strong_response=min(255, int(round(min_response * 2.5))),
        block_size=_odd(max(15.0, short * 0.05)),
        adaptive_c=-(2.0 + 6.0 * (1.0 - s)),  # -8 -> -2
        min_area=max(10, int(short * short * 1e-5)),
        max_area_ratio=0.05,
        dilate_px=max(1, int(round(short * 0.0012))),
        feather_sigma=max(0.8, short * 0.0008),
    )


def _hysteresis(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    """Mantém só os componentes de `weak` que contêm algum pixel de `strong`."""
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return np.zeros_like(weak)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[strong > 0])] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def _filter_area(mask: np.ndarray, min_area: int, max_area: int) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), 0
    areas = stats[:, cv2.CC_STAT_AREA]
    ok = (areas >= min_area) & (areas <= max_area)
    ok[0] = False
    return ok[labels].astype(np.uint8), int(ok.sum())


def detect_defects(bgr: np.ndarray, params: MaskParams) -> tuple[np.ndarray, np.ndarray, int]:
    """Detecção clássica. Retorna (máscara 0/1, resposta do Top-Hat, nº de defeitos)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    response = top_hat_response(gray, params.kernel_sizes)

    local = cv2.adaptiveThreshold(
        response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        params.block_size, params.adaptive_c,
    )
    weak = ((response >= params.min_response) & (local > 0)).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # fecha furos de 1 px
    strong = ((response >= params.strong_response) & (weak > 0)).astype(np.uint8)

    mask = _hysteresis(weak, strong)
    max_area = int(params.max_area_ratio * mask.shape[0] * mask.shape[1])
    mask, components = _filter_area(mask, params.min_area, max_area)
    return mask, response, components


def build_hybrid_mask(
    bgr: np.ndarray,
    sensitivity: float = 0.5,
    user_mask: np.ndarray | None = None,
) -> HybridMask:
    params = params_from_sensitivity(sensitivity, bgr.shape)
    mask, response, components = detect_defects(bgr, params)

    if params.dilate_px > 0:
        size = 2 * params.dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)

    binary = mask * 255
    if user_mask is not None:
        um = user_mask if user_mask.ndim == 2 else cv2.cvtColor(user_mask, cv2.COLOR_BGR2GRAY)
        if um.shape != binary.shape:
            um = cv2.resize(um, (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = cv2.bitwise_or(binary, ((um > 127) * 255).astype(np.uint8))

    hard = binary.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(hard, (0, 0), params.feather_sigma)
    soft = clip01(np.maximum(hard, blurred))  # núcleo 100% substituído, borda com queda suave

    coverage = float(np.count_nonzero(binary)) / binary.size
    return HybridMask(binary=binary, soft=soft, response=response,
                      coverage=coverage, components=components)
