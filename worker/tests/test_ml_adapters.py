import cv2
import numpy as np

from app.classic.safe_math import to_float
from app.ml.colorize import OnnxColorizer
from app.ml.inpaint import OnnxInpainter
from app.ml.registry import ModelRegistry
from app.ml.session import OnnxModel, select_providers
from app.ml.superres import OnnxUpscaler

from .conftest import make_photo
from .onnx_models import make_colorize_model, make_inpaint_model, make_upscale_model

CPU = ["CPUExecutionProvider"]


def test_select_providers_always_ends_with_cpu():
    assert select_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])[-1] == "CPUExecutionProvider"
    assert select_providers(["ProvedorInexistente"]) == ["CPUExecutionProvider"]


def test_inpainter_discovers_inputs_tile_and_output_range(tmp_path):
    model = OnnxModel(make_inpaint_model(tmp_path / "lama.onnx", size=64), CPU)
    inpainter = OnnxInpainter(model)
    assert inpainter.tile == 64  # vem da forma estática do modelo
    assert {inpainter.image_name, inpainter.mask_name} == {"image", "mask"}

    img = make_photo(200, 150)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[90:110, 60:100] = 255
    out = inpainter.inpaint(img, mask)

    assert out.shape == img.shape and out.dtype == np.uint8
    # a saída do modelo falso vem em 0..255 e é detectada; área mascarada vira cinza médio
    assert np.abs(out[95:105, 70:90].astype(int) - 128).max() <= 1
    # fora de blocos mascarados, nada muda; dentro, só difere por arredondamento
    outside = np.ones(mask.shape, bool)
    outside[mask > 0] = False
    assert np.abs(out.astype(int) - img.astype(int))[outside].max() <= 1


def test_inpainter_without_mask_returns_input_untouched(tmp_path):
    inpainter = OnnxInpainter(OnnxModel(make_inpaint_model(tmp_path / "l.onnx"), CPU))
    img = make_photo(100, 100)
    assert np.array_equal(inpainter.inpaint(img, np.zeros(img.shape[:2], np.uint8)), img)


def test_upscaler_native_and_resized_scales(tmp_path):
    model = OnnxModel(make_upscale_model(tmp_path / "up.onnx", scale=2), CPU)
    up = OnnxUpscaler(model, native_scale=2, default_tile=64, overlap=8)
    img = make_photo(90, 110)

    out2 = up.upscale(img, 2)
    assert out2.shape == (180, 220, 3)
    expected = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    assert np.abs(out2.astype(int) - expected.astype(int)).max() <= 1

    out4 = up.upscale(img, 4)  # modelo 2x pedindo 4x: redimensiona a saída
    assert out4.shape == (360, 440, 3)


def test_colorizer_adds_chroma_and_keeps_original_luminance(tmp_path):
    colorizer = OnnxColorizer(OnnxModel(make_colorize_model(tmp_path / "c.onnx", size=128), CPU))
    assert colorizer.size == 128
    gray = cv2.cvtColor(cv2.cvtColor(make_photo(180, 240), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = colorizer.colorize(gray)

    assert out.shape == gray.shape
    lab_in = cv2.cvtColor(to_float(gray), cv2.COLOR_BGR2Lab)
    lab_out = cv2.cvtColor(to_float(out), cv2.COLOR_BGR2Lab)
    assert np.hypot(lab_out[..., 1], lab_out[..., 2]).mean() > 5     # ganhou cor
    assert np.abs(lab_out[..., 0] - lab_in[..., 0]).mean() < 4       # L original preservado


def test_registry_missing_corrupt_and_valid(storage):
    reg = ModelRegistry(storage)
    assert reg.get("inpaint") is None and reg.status()["models"]["inpaint"]["present"] is False

    (storage.models_dir / storage.inpaint_model).write_bytes(b"isto nao e um modelo onnx")
    assert reg.get("inpaint") is None
    st = reg.status()["models"]["inpaint"]
    assert st["present"] and not st["loaded"] and st["error"]

    make_upscale_model(storage.models_dir / storage.superres_model, scale=4)
    assert reg.get("superres") is not None
    st = reg.status()["models"]["superres"]
    assert st["loaded"] and st["providers"] == CPU
