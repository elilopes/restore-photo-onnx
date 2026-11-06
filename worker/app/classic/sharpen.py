"""Nitidez por unsharp mask aplicada só na luminância (sem franjas coloridas)."""
from __future__ import annotations

import cv2
import numpy as np

from .safe_math import clip01, to_float, to_uint8


def unsharp_luma(bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Unsharp mask no canal L, com duas proteções:

    - limiar suave (soft-threshold) no detalhe, para não amplificar grão/ruído;
    - limite de overshoot pelo mín/máx local, para evitar halos claros/escuros.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return bgr.copy()

    lab = cv2.cvtColor(to_float(bgr), cv2.COLOR_BGR2Lab)
    lum = lab[..., 0] / 100.0  # [0, 1]

    sigma = max(0.8, min(bgr.shape[:2]) / 1800.0)
    detail = lum - cv2.GaussianBlur(lum, (0, 0), sigma)

    threshold = 0.004 * (1.0 - 0.5 * s)
    detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)

    sharpened = lum + (0.4 + 1.6 * s) * detail

    kernel = np.ones((3, 3), np.uint8)
    margin = 0.03
    sharpened = np.clip(sharpened, cv2.erode(lum, kernel) - margin, cv2.dilate(lum, kernel) + margin)

    lab[..., 0] = clip01(sharpened) * 100.0
    return to_uint8(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR))
