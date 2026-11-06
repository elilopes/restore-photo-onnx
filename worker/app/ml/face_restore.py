"""Adaptador de restauração facial — GFPGAN / CodeFormer (ONNX).

Fluxo:
1. Detectar rostos com Haar cascade
2. Para cada rosto: recortar, alinhar, normalizar, inferir, desnormalizar
3. Mesclar o resultado na imagem original com blending suave na borda

A entrada esperada pelos modelos GFPGAN/CodeFormer é [1, 3, 512, 512] float32
no espaço RGB normalizado [-1, 1]. A saída é o mesmo shape no mesmo espaço.

CodeFormer tem um input extra "w" (qualidade 0..1); se o modelo tiver 2 inputs,
o segundo recebe o valor de quality configurado.
"""
from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from ..classic.face_detect import FaceAnalysis, analyze_faces
from ..classic.safe_math import to_float, to_uint8
from .session import OnnxModel

log = logging.getLogger("worker.face_restore")

# Tamanho de entrada dos modelos de restauração facial
FACE_SIZE = 512


def _crop_face(img: np.ndarray, x: int, y: int, w: int, h: int,
               margin: float = 0.4) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recorta um rosto com margem e redimensiona para FACE_SIZE×FACE_SIZE."""
    ih, iw = img.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(iw, x + w + mx)
    y2 = min(ih, y + h + my)
    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return resized, (x1, y1, x2, y2)


def _to_tensor(bgr_face: np.ndarray) -> np.ndarray:
    """BGR uint8 → [1, 3, H, W] float32 em [-1, 1]."""
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5          # [-1, 1]
    return rgb.transpose(2, 0, 1)[None]  # NCHW


def _from_tensor(tensor: np.ndarray) -> np.ndarray:
    """[1, 3, H, W] float32 [-1, 1] → BGR uint8."""
    rgb = tensor[0].transpose(1, 2, 0)  # HWC
    rgb = np.clip(rgb * 0.5 + 0.5, 0, 1)
    return to_uint8(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _blend_face(img: np.ndarray, restored_face: np.ndarray,
                bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Mescla o rosto restaurado na imagem com borda suave."""
    x1, y1, x2, y2 = bbox
    # Redimensionar para o tamanho original do recorte
    rw, rh = x2 - x1, y2 - y1
    face_resized = cv2.resize(restored_face, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
    # Máscara de borda suave (gaussian blur de máscara sólida)
    mask = np.ones((rh, rw), dtype=np.float32)
    border = max(4, min(rw, rh) // 10)
    mask = cv2.GaussianBlur(mask, (border * 2 + 1, border * 2 + 1), border * 0.5)
    mask = mask[:, :, None]  # HWC para broadcast
    result = img.copy().astype(np.float32)
    region = result[y1:y2, x1:x2]
    result[y1:y2, x1:x2] = (
        to_float(face_resized) * mask + to_float(region) * (1 - mask)
    )
    return to_uint8(result)


class OnnxFaceRestorer:
    """Restauração facial com GFPGAN ou CodeFormer via ONNX."""

    def __init__(self, model: OnnxModel, model_type: Literal["gfpgan", "codeformer"] = "gfpgan",
                 quality: float = 0.7, min_face_size: int = 60) -> None:
        self.model = model
        self.model_type = model_type
        self.quality = float(np.clip(quality, 0.0, 1.0))
        self.min_face_size = min_face_size
        inputs = model.session.get_inputs()
        self._input_name = inputs[0].name
        self._has_quality = len(inputs) >= 2  # CodeFormer tem input 'w'
        self._quality_name = inputs[1].name if self._has_quality else None

    def restore(self, bgr: np.ndarray) -> tuple[np.ndarray, FaceAnalysis]:
        """Restaura rostos na imagem. Retorna (imagem restaurada, análise de rostos)."""
        analysis = analyze_faces(bgr, min_size=self.min_face_size)
        if analysis.face_count == 0:
            log.info("nenhum rosto detectado; pulando restauração facial")
            return bgr, analysis

        # Obter bounding boxes dos rostos
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        import cv2 as _cv2
        cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = _cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return bgr, analysis

        result = bgr
        for x, y, w, h in faces:
            try:
                face_crop, bbox = _crop_face(result, x, y, w, h)
                tensor = _to_tensor(face_crop)
                feed: dict[str, np.ndarray] = {self._input_name: tensor}
                if self._has_quality and self._quality_name:
                    feed[self._quality_name] = np.array([[self.quality]], dtype=np.float32)
                out = self.model.session.run(None, feed)[0]
                restored = _from_tensor(out)
                result = _blend_face(result, restored, bbox)
                log.debug("rosto (%d,%d,%d,%d) restaurado com %s", x, y, w, h, self.model_type)
            except Exception as exc:  # noqa: BLE001
                log.warning("falha ao restaurar rosto (%d,%d,%d,%d): %s", x, y, w, h, exc)

        return result, analysis
