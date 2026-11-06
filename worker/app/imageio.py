"""Leitura e gravação de imagens (com limites de tamanho e escrita atômica)."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Arquivo ausente de formato de imagem válido."""


def load_image(path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Lê como BGR uint8 e reduz se o maior lado passar de `max_side`.

    Retorna (imagem, (largura_original, altura_original)). `imdecode` respeita a
    orientação EXIF e converte 16 bits/alfa/cinza para BGR de 8 bits.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise InvalidImageError(f"não foi possível decodificar a imagem: {path.name}")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return img, (w, h)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def save_image(path: Path, img: np.ndarray, fmt: str, jpeg_quality: int = 95) -> None:
    if fmt == "jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"falha ao codificar {fmt}")
    _atomic_write(path, buf.tobytes())


def save_mask(path: Path, mask: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("falha ao codificar a máscara")
    _atomic_write(path, buf.tobytes())
