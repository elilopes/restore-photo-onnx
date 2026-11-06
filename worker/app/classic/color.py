"""Análise de tom e realce de cores desbotadas.

Tudo é feito em float32 (ver safe_math). Lab em float32 no OpenCV usa
L em [0, 100] e a/b em ~[-127, 127] (sem o offset de 128 do uint8).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .safe_math import blend, clip01, safe_divide, stretch, to_float, to_uint8


@dataclass(frozen=True)
class ToneAnalysis:
    mean_chroma: float
    hue_spread: float    # ~0 => todos os pixels têm a mesma matiz (cinza, sépia)
    is_monochrome: bool


def analyze_tone(bgr: np.ndarray) -> ToneAnalysis:
    """Heurística: a foto é P&B (inclui sépia/viragem) ou colorida?

    - croma médio muito baixo => P&B neutro;
    - croma moderado, mas com os pontos (a, b) alinhados numa única direção
      (razão entre desvios padrão dos eixos menor/maior da nuvem) => P&B virado.
    """
    h, w = bgr.shape[:2]
    scale = 256.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    lab = cv2.cvtColor(to_float(small), cv2.COLOR_BGR2Lab)
    a, b = lab[..., 1].ravel(), lab[..., 2].ravel()
    mean_chroma = float(np.hypot(a, b).mean())

    if mean_chroma < 3.0:
        return ToneAnalysis(mean_chroma, 0.0, True)

    eig = np.linalg.eigvalsh(np.cov(np.stack([a, b])))  # crescente
    spread = float(np.sqrt(max(eig[0], 0.0)) / (np.sqrt(max(eig[1], 0.0)) + 1e-6))
    return ToneAnalysis(mean_chroma, spread, bool(mean_chroma < 25.0 and spread < 0.15))


def _percentile_bounds(channel: np.ndarray, lo_p: float = 0.5, hi_p: float = 99.5) -> tuple[float, float]:
    lo, hi = np.percentile(channel[::4, ::4], (lo_p, hi_p))
    return float(lo), float(hi)


def enhance_colors(bgr: np.ndarray, strength: float = 0.5, monochrome: bool = False) -> np.ndarray:
    """Realça contraste e cor de fotos desbotadas. `strength` em [0, 1].

    1. Auto-níveis por canal (recupera contraste e reduz dominante amarelada);
    2. Balanço de branco "gray world" suave, com ganho limitado;
    3. CLAHE no canal L (contraste local, sem estourar);
    4. Vibrance: aumenta mais a croma dos pixels pouco saturados.

    Em fotos monocromáticas, só o passo 3 é aplicado (a viragem é preservada).
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return bgr.copy()

    x = to_float(bgr)

    if not monochrome:
        stretched = np.empty_like(x)
        for c in range(3):
            lo, hi = _percentile_bounds(x[..., c])
            stretched[..., c] = stretch(x[..., c], lo, hi)
        x = blend(x, stretched, 0.8 * s)

        means = x.reshape(-1, 3).mean(axis=0)  # B, G, R
        gains = np.clip(safe_divide(float(means.mean()), means), 0.85, 1.18)
        x = clip01(x * (1.0 + (gains - 1.0) * s))

    lab = cv2.cvtColor(x, cv2.COLOR_BGR2Lab)
    lightness = lab[..., 0]
    l16 = np.clip(lightness * (65535.0 / 100.0), 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.0 + 2.0 * s, tileGridSize=(8, 8))
    equalized = clahe.apply(l16).astype(np.float32) * (100.0 / 65535.0)
    lab[..., 0] = lightness + (equalized - lightness) * s

    if not monochrome:
        a, b = lab[..., 1], lab[..., 2]
        chroma = np.hypot(a, b)
        gain = 1.0 + 0.9 * s * (1.0 - np.clip(chroma / 60.0, 0.0, 1.0))
        lab[..., 1] = np.clip(a * gain, -127.0, 127.0)
        lab[..., 2] = np.clip(b * gain, -127.0, 127.0)

    return to_uint8(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR))
