from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import get_settings


def make_photo(h: int = 300, w: int = 400, seed: int = 0) -> np.ndarray:
    """Foto sintética suave (gradientes + formas + grão leve), BGR uint8."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([
        110 + 60 * np.sin(xx / 70.0),
        120 + 50 * np.cos(yy / 60.0),
        130 + 40 * np.sin((xx + yy) / 90.0),
    ], axis=-1)
    cv2.circle(img, (w // 3, h // 2), h // 5, (60, 90, 170), -1)
    cv2.rectangle(img, (w // 2, h // 4), (w - 40, h - 60), (150, 170, 90), -1)
    img += rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_scratch(img: np.ndarray, p1, p2, thickness: int = 3, value: int = 245) -> np.ndarray:
    out = img.copy()
    cv2.line(out, p1, p2, (value, value, value), thickness)
    return out


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """STORAGE_DIR/MODELS_DIR temporários e Settings recriadas."""
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ORT_PROVIDERS", "CPUExecutionProvider")
    (tmp_path / "models").mkdir()
    get_settings.cache_clear()
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True)
    settings.results_dir.mkdir(parents=True)
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def put_upload(storage):
    def _put(img: np.ndarray, ext: str = "png") -> tuple[str, str]:
        job_id = str(uuid.uuid4())
        name = f"{job_id}.{ext}"
        ok, buf = cv2.imencode(f".{ext}", img)
        assert ok
        Path(storage.uploads_dir / name).write_bytes(buf.tobytes())
        return job_id, name
    return _put
