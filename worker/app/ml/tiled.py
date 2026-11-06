"""Inferência em blocos (tiles) com mistura suave nas sobreposições.

Serve para modelos de entrada fixa (ex.: LaMa 512x512) e para evitar estourar a
memória em imagens grandes. Os blocos se sobrepõem e são combinados com uma
janela em rampa, cuja soma nas sobreposições é constante, então não aparecem
emendas.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # último bloco encosta na borda
    return starts


def ramp_window(tile: int, overlap: int, scale: int) -> np.ndarray:
    n, o = tile * scale, overlap * scale
    ramp = np.ones(n, dtype=np.float32)
    if o > 0:
        r = (np.arange(o, dtype=np.float32) + 1.0) / (o + 1.0)
        ramp[:o] = r
        ramp[-o:] = np.minimum(ramp[-o:], r[::-1])
    return np.outer(ramp, ramp)


def tiled_apply(
    image: np.ndarray,
    tile: int,
    overlap: int,
    scale: int,
    fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    aux: np.ndarray | None = None,
    skip_empty_aux: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Aplica `fn` em blocos de `image` (HxWxC float32) e recompõe o resultado.

    - `fn(bloco, aux_bloco)` recebe SEMPRE blocos `tile x tile` (com padding
      simétrico nas bordas) e devolve `(tile*scale) x (tile*scale) x C'`;
    - `aux` (ex.: máscara HxW) é recortada junto com a imagem;
    - `skip_empty_aux=True` pula blocos cuja `aux` é toda zero (inpainting só
      onde há máscara).

    Retorna (resultado, cobertura_bool) ou None se nenhum bloco foi processado.
    """
    h, w = image.shape[:2]
    overlap = max(0, min(overlap, tile // 2))
    window = ramp_window(tile, overlap, scale)
    acc: np.ndarray | None = None
    weights = np.zeros((h * scale, w * scale), dtype=np.float32)

    for y0 in tile_starts(h, tile, overlap):
        for x0 in tile_starts(w, tile, overlap):
            th, tw = min(tile, h - y0), min(tile, w - x0)
            aux_t = aux[y0:y0 + th, x0:x0 + tw] if aux is not None else None
            if skip_empty_aux and aux_t is not None and not aux_t.any():
                continue
            img_t = image[y0:y0 + th, x0:x0 + tw]
            if th < tile or tw < tile:
                pad = ((0, tile - th), (0, tile - tw))
                img_t = np.pad(img_t, pad + ((0, 0),), mode="symmetric")
                if aux_t is not None:
                    aux_t = np.pad(aux_t, pad + ((0, 0),) * (aux_t.ndim - 2), mode="symmetric")

            out = fn(img_t, aux_t)
            expected = (tile * scale, tile * scale)
            if out.shape[:2] != expected:
                raise RuntimeError(f"saída do modelo {out.shape[:2]} != esperado {expected}")
            out = out[: th * scale, : tw * scale]
            win = window[: th * scale, : tw * scale]

            if acc is None:
                acc = np.zeros((h * scale, w * scale, out.shape[2]), dtype=np.float32)
            ys = slice(y0 * scale, (y0 + th) * scale)
            xs = slice(x0 * scale, (x0 + tw) * scale)
            acc[ys, xs] += out * win[..., None]
            weights[ys, xs] += win

    if acc is None:
        return None
    covered = weights > 0
    np.divide(acc, weights[..., None], out=acc, where=covered[..., None])
    return acc, covered
