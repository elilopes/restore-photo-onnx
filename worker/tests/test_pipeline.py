import dataclasses

import cv2
import numpy as np
import pytest

from app.imageio import InvalidImageError
from app.ml.registry import ModelRegistry
from app.pipeline import Pipeline
from app.schemas import ProcessRequest, RestoreOptions

from .conftest import add_scratch, make_photo
from .onnx_models import make_colorize_model, make_inpaint_model, make_upscale_model


def _run(storage, put_upload, img, settings=None, **opts):
    job_id, name = put_upload(img)
    pipe = Pipeline(settings or storage, ModelRegistry(settings or storage))
    req = ProcessRequest(job_id=job_id, input_name=name, options=RestoreOptions(**opts))
    resp = pipe.run(req)
    out = cv2.imread(str(storage.results_dir / job_id / resp.result_name))
    return resp, out, job_id


def _step(resp, name):
    return next(s for s in resp.steps if s.name == name)


def test_classic_pipeline_repairs_scratch_without_models(storage, put_upload):
    clean = make_photo()
    scratched = add_scratch(clean, (60, 20), (330, 270), thickness=3)
    resp, out, job_id = _run(storage, put_upload, scratched, enhance_colors=False, sharpen=False)

    assert (resp.width, resp.height) == (400, 300) and out.shape == clean.shape
    assert _step(resp, "repair").status == "done" and _step(resp, "repair").engine == "opencv-telea"
    assert _step(resp, "colorize").status == "skipped"  # foto colorida, modo auto
    assert resp.mask_name == "mask.png" and (storage.results_dir / job_id / "mask.png").is_file()

    line = np.zeros(clean.shape[:2], np.uint8)
    cv2.line(line, (60, 20), (330, 270), 255, 3)
    err_before = np.abs(scratched.astype(int) - clean.astype(int))[line > 0].mean()
    err_after = np.abs(out.astype(int) - clean.astype(int))[line > 0].mean()
    assert err_after < err_before * 0.35, (err_before, err_after)

    # fora da máscara, a imagem não é tocada
    mask = cv2.imread(str(storage.results_dir / job_id / "mask.png"), cv2.IMREAD_GRAYSCALE)
    far = cv2.erode((mask == 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(out.astype(int) - scratched.astype(int))[far].max() == 0


def test_clean_photo_reports_no_defects_and_all_options_off_is_lossless(storage, put_upload):
    img = make_photo()
    resp, out, _ = _run(storage, put_upload, img, enhance_colors=False, sharpen=False)
    assert _step(resp, "repair").status == "skipped" and resp.mask_name is None
    assert np.array_equal(out, img)


def test_defaults_run_every_classic_step_and_jpg_output(storage, put_upload):
    resp, out, job_id = _run(storage, put_upload, make_photo(), output_format="jpg")
    assert resp.result_name == "result.jpg" and out is not None
    assert [s.name for s in resp.steps] == ["load", "repair", "colorize", "enhance_colors", "upscale", "sharpen", "save"]
    assert _step(resp, "enhance_colors").status == "done" and _step(resp, "sharpen").status == "done"


def test_upscale_fallback_lanczos_and_output_limit(storage, put_upload):
    resp, out, _ = _run(storage, put_upload, make_photo(), upscale=2, repair=False, enhance_colors=False, sharpen=False)
    assert out.shape[:2] == (600, 800) and _step(resp, "upscale").engine == "opencv-lanczos"

    limited = dataclasses.replace(storage, max_output_side=500)
    resp, out, _ = _run(storage, put_upload, make_photo(), settings=limited, upscale=2,
                        repair=False, enhance_colors=False, sharpen=False)
    assert out.shape[:2] == (300, 400) and _step(resp, "upscale").status == "skipped"


def test_monochrome_photo_without_model_skips_colorize_but_still_enhances(storage, put_upload):
    gray = cv2.cvtColor(cv2.cvtColor(make_photo(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    resp, _, _ = _run(storage, put_upload, gray, repair=False, sharpen=False)
    assert resp.was_monochrome
    col = _step(resp, "colorize")
    assert col.status == "skipped" and "indisponível" in col.detail
    assert _step(resp, "enhance_colors").detail == "modo monocromático"


def test_with_onnx_models_all_three_are_used(storage, put_upload, monkeypatch):
    monkeypatch.setenv("SUPERRES_SCALE", "2")
    from app.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    make_inpaint_model(settings.models_dir / settings.inpaint_model, 64)
    make_colorize_model(settings.models_dir / settings.colorize_model, 128)
    make_upscale_model(settings.models_dir / settings.superres_model, 2)

    gray = cv2.cvtColor(cv2.cvtColor(make_photo(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    gray = add_scratch(gray, (60, 20), (330, 270), 3)
    resp, out, _ = _run(storage, put_upload, gray, settings=settings, upscale=2)

    assert out.shape[:2] == (600, 800)
    for step in ("repair", "colorize", "upscale"):
        s = _step(resp, step)
        assert s.status == "done" and s.engine.startswith("onnx:"), (step, s)
    assert _step(resp, "enhance_colors").detail is None  # já colorida: não é mais monocromática
    chroma = np.hypot(*(cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[..., 1:].astype(float) - 128).transpose(2, 0, 1)).mean()
    assert chroma > 3


def test_broken_model_falls_back_instead_of_failing_the_job(storage, put_upload):
    (storage.models_dir / storage.inpaint_model).write_bytes(b"lixo")
    scratched = add_scratch(make_photo(), (60, 20), (330, 270), 3)
    resp, _, _ = _run(storage, put_upload, scratched)
    assert _step(resp, "repair").status == "done" and _step(resp, "repair").engine == "opencv-telea"


def test_step_failure_is_reported_not_raised(storage, put_upload, monkeypatch):
    import app.pipeline as pl
    monkeypatch.setattr(pl, "unsharp_luma", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    resp, out, _ = _run(storage, put_upload, make_photo(), repair=False)
    assert _step(resp, "sharpen").status == "failed" and "boom" in _step(resp, "sharpen").detail
    assert out is not None


def test_errors_missing_upload_and_invalid_image(storage):
    pipe = Pipeline(storage, ModelRegistry(storage))
    jid = "11111111-1111-4111-8111-111111111111"
    with pytest.raises(FileNotFoundError):
        pipe.run(ProcessRequest(job_id=jid, input_name="nao-existe.png"))
    (storage.uploads_dir / "lixo.png").write_bytes(b"nao sou uma imagem")
    with pytest.raises(InvalidImageError):
        pipe.run(ProcessRequest(job_id=jid, input_name="lixo.png"))


def test_large_input_is_downscaled_to_max_side(storage, put_upload):
    small_limit = dataclasses.replace(storage, max_side=200)
    resp, out, _ = _run(storage, put_upload, make_photo(), settings=small_limit,
                        repair=False, enhance_colors=False, sharpen=False)
    assert (resp.input_width, resp.input_height) == (400, 300)
    assert out.shape[:2] == (150, 200)
