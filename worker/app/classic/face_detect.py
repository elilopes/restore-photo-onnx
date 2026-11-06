"""Detecção de rostos com Haar cascade do OpenCV (sem modelo extra).

Calcula:
- número de rostos detectados
- proporção da área de rostos em relação à imagem (coverage 0..1)
- sugestão de restauração facial (cobertura > 30% ou mais de 2 rostos)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("worker.face_detect")

# Caminho padrão do cascade que vem com o OpenCV
_CASCADE_PATHS = [
    "haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml",
]


def _load_cascade() -> cv2.CascadeClassifier | None:
    """Carrega o Haar cascade. Retorna None se não encontrar."""
    cv_data = cv2.data.haarcascades  # type: ignore[attr-defined]
    for rel in _CASCADE_PATHS:
        path = Path(cv_data) / rel if cv_data else Path(rel)
        if path.is_file():
            cc = cv2.CascadeClassifier(str(path))
            if not cc.empty():
                return cc
    # Fallback: procurar em localizações conhecidas do sistema
    for candidate in [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]:
        if Path(candidate).is_file():
            cc = cv2.CascadeClassifier(candidate)
            if not cc.empty():
                return cc
    return None


_cascade: cv2.CascadeClassifier | None | bool = False   # False = não tentou ainda


def _get_cascade() -> cv2.CascadeClassifier | None:
    global _cascade
    if _cascade is False:
        _cascade = _load_cascade()
        if _cascade is None:
            log.warning("Haar cascade não encontrado; detecção de rostos desabilitada")
    return _cascade  # type: ignore[return-value]


@dataclass
class FaceAnalysis:
    face_count: int
    face_coverage: float          # 0..1
    suggest_face_restore: bool    # True se vale oferecer restauração facial


def analyze_faces(bgr: np.ndarray, min_size: int = 60) -> FaceAnalysis:
    """Detecta rostos na imagem BGR e retorna a análise.

    Args:
        bgr: imagem em formato BGR (uint8)
        min_size: tamanho mínimo de rosto em pixels (lado do quadrado)

    Returns:
        FaceAnalysis com contagem, cobertura e sugestão
    """
    cascade = _get_cascade()
    if cascade is None:
        return FaceAnalysis(face_count=0, face_coverage=0.0, suggest_face_restore=False)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    h, w = bgr.shape[:2]
    img_area = h * w

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return FaceAnalysis(face_count=0, face_coverage=0.0, suggest_face_restore=False)

    face_area = int(sum(fw * fh for _, _, fw, fh in faces))
    coverage = min(1.0, face_area / max(img_area, 1))
    face_count = len(faces)

    # Sugerir restauração facial se:
    # - cobertura > 30% (rosto(s) dominam a foto), ou
    # - mais de 2 rostos (foto de família/grupo)
    suggest = coverage > 0.30 or face_count > 2

    log.debug("rostos=%d cobertura=%.2f%% sugerir=%s", face_count, coverage * 100, suggest)
    return FaceAnalysis(
        face_count=face_count,
        face_coverage=round(coverage, 4),
        suggest_face_restore=suggest,
    )
