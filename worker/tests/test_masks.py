import cv2
import numpy as np

from app.classic.masks import build_hybrid_mask
from app.classic.tophat import top_hat_response

from .conftest import add_scratch, make_photo


def _line_mask(shape, p1, p2, thickness):
    m = np.zeros(shape[:2], np.uint8)
    cv2.line(m, p1, p2, 255, thickness)
    return m > 0


def test_tophat_flags_thin_bright_and_dark_lines_only():
    gray = np.full((120, 120), 100, np.uint8)
    cv2.line(gray, (10, 30), (110, 30), 230, 2)  # risco claro
    cv2.line(gray, (10, 90), (110, 90), 20, 2)   # vinco escuro
    resp = top_hat_response(gray, (9,))
    assert resp[30, 60] > 80 and resp[90, 60] > 60
    assert resp[60, 60] == 0  # área plana: resposta nula


def test_hybrid_mask_covers_scratch_and_leaves_clean_areas_alone():
    clean = make_photo()
    scratched = add_scratch(clean, (60, 20), (330, 270), thickness=3)
    hm = build_hybrid_mask(scratched, sensitivity=0.5)

    line = _line_mask(scratched.shape, (60, 20), (330, 270), 3)
    recall = (hm.binary[line] > 0).mean()
    assert recall > 0.9, f"riscado coberto apenas {recall:.0%}"
    assert hm.coverage < 0.08          # não pode engolir a foto
    assert hm.components >= 1


def test_clean_photo_produces_almost_empty_mask():
    hm = build_hybrid_mask(make_photo(), sensitivity=0.5)
    assert hm.coverage < 0.005


def test_sensitivity_is_monotonic():
    scratched = add_scratch(make_photo(), (60, 20), (330, 270), thickness=2, value=170)
    low = build_hybrid_mask(scratched, 0.1).coverage
    high = build_hybrid_mask(scratched, 0.95).coverage
    assert high >= low


def test_user_mask_is_united_and_soft_mask_is_bounded():
    img = make_photo()
    user = np.zeros(img.shape[:2], np.uint8)
    user[100:120, 100:140] = 255
    hm = build_hybrid_mask(img, 0.5, user_mask=user)
    assert (hm.binary[100:120, 100:140] == 255).all()
    assert hm.soft.dtype == np.float32 and hm.soft.min() >= 0 and hm.soft.max() <= 1
    assert (hm.soft[hm.binary > 0] == 1.0).all()  # núcleo totalmente substituído
