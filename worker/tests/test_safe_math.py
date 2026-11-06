import numpy as np

from app.classic import safe_math as sm


def test_numpy_uint8_wraps_but_saturating_ops_do_not():
    a = np.array([250, 5], dtype=np.uint8)
    b = np.array([10, 10], dtype=np.uint8)
    assert list(a + b) == [4, 15]  # o problema que a proteção evita
    assert sm.saturating_add(a, b).ravel().tolist() == [255, 15]
    assert sm.saturating_sub(a, b).ravel().tolist() == [240, 0]


def test_to_uint8_clips_and_sanitizes():
    x = np.array([-0.5, 0.0, 0.5, 1.0, 1.7, np.nan, np.inf, -np.inf], dtype=np.float32)
    assert sm.to_uint8(x).tolist() == [0, 0, 128, 255, 255, 0, 255, 0]


def test_to_float_ranges():
    assert sm.to_float(np.array([0, 255], np.uint8)).tolist() == [0.0, 1.0]
    assert sm.to_float(np.array([0, 65535], np.uint16)).tolist() == [0.0, 1.0]
    out = sm.to_float(np.array([-1.0, 2.0, np.nan], np.float64))
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_to_float_does_not_mutate_input():
    x = np.array([np.nan, 0.5], dtype=np.float32)
    sm.to_float(x)
    assert np.isnan(x[0])


def test_blend_is_bounded_and_broadcasts_hw_alpha():
    base = np.zeros((4, 4, 3), np.float32)
    other = np.ones((4, 4, 3), np.float32)
    alpha = np.full((4, 4), 0.25, np.float32)
    out = sm.blend(base, other, alpha)
    assert np.allclose(out, 0.25)
    assert sm.blend(base, other * 5, 2.0).max() == 1.0  # nunca passa de 1


def test_safe_divide_and_stretch_have_no_inf():
    assert np.isfinite(sm.safe_divide(np.ones(3), np.zeros(3))).all()
    x = np.full((3, 3), 0.5, np.float32)
    out = sm.stretch(x, 0.5, 0.5)  # lo == hi: span mínimo evita divisão por zero
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
