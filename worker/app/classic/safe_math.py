"""Operações numéricas protegidas contra overflow/underflow.

Regra do projeto: toda aritmética sobre pixels acontece em float32 no intervalo
[0, 1]. A volta para uint8 passa SEMPRE por `to_uint8`, que sanitiza NaN/Inf,
faz clip e arredonda. Nunca faça `a + b` ou `a - b` direto em arrays uint8: o
numpy "dá a volta" (250 + 10 == 4) e isso vira artefato visível na foto.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def sanitize(x: np.ndarray) -> np.ndarray:
    """Troca NaN por 0 e ±Inf por 1/0 (devolve uma cópia)."""
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8/uint16/float -> float32 em [0, 1] (apenas para imagens)."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) * np.float32(1.0 / 255.0)
    if img.dtype == np.uint16:
        return img.astype(np.float32) * np.float32(1.0 / 65535.0)
    if np.issubdtype(img.dtype, np.floating):
        return clip01(sanitize(img.astype(np.float32, copy=False)))
    raise TypeError(f"dtype não suportado: {img.dtype}")


def to_uint8(x: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8, com sanitização, clip e arredondamento."""
    x = sanitize(np.asarray(x, dtype=np.float32))
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def saturating_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Soma saturada em uint8 (250 + 10 == 255).

    Feito em numpy (e não com cv2.add) porque o OpenCV trata arrays com até 4
    elementos como escalares, o que dá resultado errado em formatos pequenos.
    """
    return np.minimum(a.astype(np.uint16) + b, 255).astype(np.uint8)


def saturating_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Subtração saturada em uint8 (5 - 10 == 0)."""
    return np.clip(a.astype(np.int16) - b, 0, 255).astype(np.uint8)


def safe_divide(a, b, eps: float = _EPS):
    """a / b sem divisão por zero (|b| < eps vira eps)."""
    b = np.asarray(b, dtype=np.float32)
    return np.asarray(a, dtype=np.float32) / np.where(np.abs(b) < eps, eps, b)


def blend(base: np.ndarray, other: np.ndarray, alpha) -> np.ndarray:
    """Mistura `base` e `other` em float32: base + (other - base) * alpha.

    `alpha` pode ser escalar, HxW (expandido para HxWx3) ou do mesmo formato da
    imagem. O resultado é sempre clipado em [0, 1].
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if 0 < a.ndim == base.ndim - 1:
        a = a[..., None]
    return clip01(base + (other - base) * a)


def stretch(x: np.ndarray, lo: float, hi: float, min_span: float = 0.25) -> np.ndarray:
    """Estica [lo, hi] para [0, 1]. `min_span` limita o ganho máximo (1/min_span)."""
    span = max(hi - lo, min_span)
    return clip01((x - lo) / span)
