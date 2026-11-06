import cv2
import numpy as np

from app.classic.color import analyze_tone, enhance_colors
from app.classic.safe_math import to_float
from app.classic.sharpen import unsharp_luma

from .conftest import make_photo


def _chroma(bgr):
    lab = cv2.cvtColor(to_float(bgr), cv2.COLOR_BGR2Lab)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def _fade(bgr, k=0.45):
    """Simula foto desbotada: baixa contraste e saturação, dominante amarelada."""
    x = bgr.astype(np.float32)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * 0.35
    x = 140 + (x - 128) * k
    x[..., 0] -= 12  # menos azul => amarelado
    return np.clip(x, 0, 255).astype(np.uint8)


def test_tone_analysis_gray_sepia_color():
    photo = make_photo()
    gray = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert analyze_tone(gray).is_monochrome

    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sepia = np.stack([g * 0.55, g * 0.80, g * 1.00], axis=-1)  # BGR com viragem
    sepia = (np.clip(sepia, 0, 1) * 255).astype(np.uint8)
    assert analyze_tone(sepia).is_monochrome

    assert not analyze_tone(photo).is_monochrome


def test_enhance_restores_contrast_and_chroma_of_faded_photo():
    faded = _fade(make_photo())
    out = enhance_colors(faded, strength=0.7)
    assert out.dtype == np.uint8 and out.shape == faded.shape
    assert out.std() > faded.std() * 1.3
    assert _chroma(out) > _chroma(faded) * 1.3


def test_enhance_strength_zero_is_identity_and_monochrome_keeps_tint():
    photo = make_photo()
    assert np.array_equal(enhance_colors(photo, 0.0), photo)

    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sepia = (np.clip(np.stack([g * 0.55, g * 0.80, g], axis=-1), 0, 1) * 255).astype(np.uint8)
    out = enhance_colors(sepia, 0.8, monochrome=True)
    assert analyze_tone(out).is_monochrome  # continua monocromática


def test_enhance_extremes_never_overflow():
    for fill in (0, 255):
        flat = np.full((64, 64, 3), fill, np.uint8)
        out = enhance_colors(flat, 1.0)
        assert out.dtype == np.uint8 and np.isfinite(out).all()


def test_unsharp_increases_edge_energy_without_overflow_on_extremes():
    photo = cv2.GaussianBlur(make_photo(), (0, 0), 1.6)
    out = unsharp_luma(photo, 0.8)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    assert lap(out) > lap(photo) * 1.2

    # imagem com bordas pretas/brancas: sem wrap-around (pixel preto virando branco)
    hard = np.zeros((80, 80, 3), np.uint8)
    hard[:, 40:] = 255
    res = unsharp_luma(hard, 1.0)
    assert res[:, :30].max() <= 5 and res[:, 50:].min() >= 250
    assert np.array_equal(unsharp_luma(photo, 0.0), photo)
