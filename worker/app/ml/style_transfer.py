"""Adaptador de Style Transfer artístico (Fast Neural Style Transfer em ONNX).

Modelos disponíveis (cada um é um arquivo .onnx separado):
  candy.onnx, mosaic.onnx, rain_princess.onnx, udnie.onnx

Entrada: [1, 3, H, W] float32 em [0, 255] (sem normalização — esse é o padrão
dos modelos PyTorch/ONNX de fast neural style do repositório oficial da PyTorch).
Saída: mesma forma.

A intensidade do estilo é controlada por interpolação com a imagem original.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..classic.safe_math import to_float, to_uint8
from .session import OnnxModel, select_providers

log = logging.getLogger("worker.style_transfer")

KNOWN_STYLES = ("candy", "mosaic", "rain_princess", "udnie")

# Dimensão máxima de entrada para style transfer (redimensiona se necessário)
MAX_SIDE = 720


class OnnxStyleTransfer:
    """Aplica estilo artístico via Fast Neural Style Transfer."""

    def __init__(self, model: OnnxModel, strength: float = 0.7) -> None:
        self.model = model
        self.strength = float(np.clip(strength, 0.0, 1.0))
        inputs = model.session.get_inputs()
        self._input_name = inputs[0].name
        # Verificar se a entrada tem tamanho fixo
        shape = inputs[0].shape
        self._fixed_h: int | None = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else None
        self._fixed_w: int | None = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else None

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        """Aplica o estilo artístico à imagem."""
        orig_bgr = bgr.copy()
        h, w = bgr.shape[:2]

        # Redimensionar se necessário
        scale = 1.0
        if self._fixed_h and self._fixed_w:
            target_h, target_w = self._fixed_h, self._fixed_w
        elif max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            target_h = int(h * scale)
            target_w = int(w * scale)
        else:
            target_h, target_w = h, w

        resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Tensor [1, 3, H, W] em [0, 255]
        tensor = rgb.transpose(2, 0, 1)[None]
        out = self.model.session.run(None, {self._input_name: tensor})[0]
        out_rgb = np.asarray(out, dtype=np.float32)[0].transpose(1, 2, 0)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
        styled_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # Restaurar resolução original
        if (target_h, target_w) != (h, w):
            styled_bgr = cv2.resize(styled_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Blending com original para controlar intensidade
        if self.strength < 1.0:
            orig_f = to_float(orig_bgr)
            styled_f = to_float(styled_bgr)
            blended = styled_f * self.strength + orig_f * (1.0 - self.strength)
            return to_uint8(blended)
        return styled_bgr


class StyleTransferRegistry:
    """Gerencia múltiplos modelos de estilo — um arquivo ONNX por estilo."""

    def __init__(self, models_dir: Path, model_prefix: str, providers: tuple[str, ...],
                 ort_threads: int = 0) -> None:
        self._dir = models_dir
        self._prefix = model_prefix
        self._providers = providers
        self._threads = ort_threads
        self._cache: dict[str, OnnxStyleTransfer | None] = {}

    def get(self, style_name: str, strength: float = 0.7) -> OnnxStyleTransfer | None:
        """Retorna o adaptador para o estilo, ou None se o arquivo não existir."""
        key = f"{style_name}:{strength:.2f}"
        if key in self._cache:
            return self._cache[key]

        path = self._dir / f"{self._prefix}{style_name}.onnx"
        if not path.is_file():
            log.warning("modelo de estilo não encontrado: %s", path)
            self._cache[key] = None
            return None

        try:
            model = OnnxModel(path, self._providers, self._threads)
            adapter = OnnxStyleTransfer(model, strength)
            self._cache[key] = adapter
            log.info("estilo '%s' carregado: %s", style_name, path.name)
            return adapter
        except Exception as exc:  # noqa: BLE001
            log.warning("falha ao carregar estilo '%s': %s", style_name, exc)
            self._cache[key] = None
            return None

    def available_styles(self) -> list[str]:
        """Lista os estilos disponíveis (arquivos presentes na pasta)."""
        return [
            path.stem[len(self._prefix):]
            for path in self._dir.glob(f"{self._prefix}*.onnx")
        ]
