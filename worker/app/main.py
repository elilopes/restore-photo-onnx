"""FastAPI do worker de inferência.

Contrato (consumido pelo worker BullMQ em api/src/worker.ts):
  GET  /health                   liveness
  GET  /models                   modelos ONNX presentes/carregados e provedores
  GET  /progress/{job_id}        progresso do job em andamento (404 se não há)
  POST /process                  executa o pipeline (síncrono) e devolve o relatório
  POST /analyze                  analisa rostos na imagem (sem processar)
  POST /models/download          inicia download dos modelos ONNX em background
  GET  /models/download/status   estado atual do download
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path as FilePath

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .download_tracker import DownloadTracker
from .imageio import InvalidImageError
from .ml.registry import ModelRegistry
from .pipeline import Pipeline
from .progress import ProgressTracker
from pydantic import BaseModel
from .classic.face_detect import analyze_faces
from .imageio import load_image
from .schemas import (
    UUID_PATTERN, AnalyzeFaceResponse, FILENAME_PATTERN,
    ProcessRequest, ProcessResponse, ProgressResponse,
)

# Modelos disponíveis para download — mesma lista de scripts/download_models.py
_MODELS: dict[str, tuple[str, str]] = {
    "lama_fp32.onnx": (
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "inpaint",
    ),
    "ddcolor-tiny-fp16.onnx": (
        "https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx",
        "colorize",
    ),
    "realesrgan_x4plus.onnx": (
        "https://huggingface.co/Qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx",
        "superres",
    ),
    "swinir_x4.onnx": (
        "https://huggingface.co/eugenesiow/SwinIR/resolve/main/swinir_real_sr_x4.onnx",
        "superres_fb",
    ),
    "gfpgan_v1.4.onnx": (
        "https://huggingface.co/facefusion/models-3.1.0/resolve/main/gfpgan_1.4.onnx",
        "face",
    ),
    "codeformer.onnx": (
        "https://huggingface.co/facefusion/models-3.1.0/resolve/main/codeformer.onnx",
        "face",
    ),
    "sci_medium.onnx": (
        "https://huggingface.co/eugenesiow/SCI/resolve/main/sci_medium.onnx",
        "lowlight",
    ),
    "style_candy.onnx": (
        "https://huggingface.co/onnx/style-transfer/resolve/main/candy.onnx",
        "style",
    ),
    "style_mosaic.onnx": (
        "https://huggingface.co/onnx/style-transfer/resolve/main/mosaic.onnx",
        "style",
    ),
}
_MIN_VALID_BYTES = 1_000_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("worker")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(settings)
    tracker = ProgressTracker()
    dl_tracker = DownloadTracker()
    app.state.registry = registry
    app.state.tracker = tracker
    app.state.dl_tracker = dl_tracker
    app.state.pipeline = Pipeline(settings, registry, tracker)
    app.state.gate = threading.BoundedSemaphore(settings.max_concurrency)
    log.info("storage=%s models=%s providers=%s", settings.storage_dir, settings.models_dir, registry.status()["providers"])
    yield


app = FastAPI(title="restore-photo worker", version="0.1.0", lifespan=lifespan)


@app.exception_handler(InvalidImageError)
async def _invalid_image(_: Request, exc: InvalidImageError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def _not_found(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("erro inesperado")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models(request: Request) -> dict:
    return request.app.state.registry.status()


@app.get("/progress/{job_id}", response_model=ProgressResponse)
def progress(request: Request, job_id: str = Path(pattern=UUID_PATTERN)) -> ProgressResponse:
    entry = request.app.state.tracker.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="sem progresso para este job")
    return ProgressResponse(progress=entry[0], step=entry[1])


class _AnalyzeRequest(BaseModel):
    input_name: str


@app.post("/analyze", response_model=AnalyzeFaceResponse)
def analyze(req: _AnalyzeRequest, request: Request) -> AnalyzeFaceResponse:
    """Detecta rostos na imagem sem processá-la. Usado pelo frontend para sugerir restauração facial."""
    settings = get_settings()
    base = settings.uploads_dir.resolve()
    path = (base / req.input_name).resolve()
    if path.parent != base or not path.is_file():
        raise HTTPException(status_code=404, detail="upload não encontrado")
    bgr, _ = load_image(path, settings.max_side)
    result = analyze_faces(bgr, min_size=settings.face_detect_min_size)
    return AnalyzeFaceResponse(
        face_count=result.face_count,
        face_coverage=result.face_coverage,
        suggest_face_restore=result.suggest_face_restore,
    )


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest, request: Request) -> ProcessResponse:
    # `def` (não async): o FastAPI roda em thread pool, então o CPU pesado não bloqueia o loop.
    with request.app.state.gate:
        return request.app.state.pipeline.run(req)


# ---------------------------------------------------------------------------
# Download de modelos ONNX
# ---------------------------------------------------------------------------

def _download_worker(models_dir: FilePath, only: str | None, tracker: DownloadTracker) -> None:
    """Executa em background thread; atualiza tracker conforme avança."""
    to_download = {
        name: (url, role)
        for name, (url, role) in _MODELS.items()
        if only is None or role == only
    }
    tracker.start(list(to_download.keys()))

    for filename, (url, _role) in to_download.items():
        dest = models_dir / filename
        if dest.is_file() and dest.stat().st_size >= _MIN_VALID_BYTES:
            tracker.update(filename, status="skipped", size_done=dest.stat().st_size, size_total=dest.stat().st_size)
            continue

        tracker.update(filename, status="downloading")
        tmp = dest.with_name(dest.name + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "restore-photo-onnx/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                tracker.update(filename, status="downloading", size_total=total)
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    tracker.update(filename, status="downloading", size_done=done)
            if tmp.stat().st_size < _MIN_VALID_BYTES:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("arquivo muito pequeno; URL pode ter mudado")
            os.replace(tmp, dest)
            tracker.update(filename, status="done", size_done=dest.stat().st_size)
            log.info("modelo baixado: %s", dest)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            log.error("falha ao baixar %s: %s", filename, exc)
            tracker.update(filename, status="error", error=str(exc))

    tracker.finish()
    log.info("download de modelos concluído")


@app.post("/models/download", status_code=202)
async def start_download(request: Request, only: str | None = None) -> dict:
    """Inicia o download dos modelos em background. Idempotente se já estiver rodando."""
    tracker: DownloadTracker = request.app.state.dl_tracker
    if tracker.is_running:
        return {"status": "already_running"}

    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    t = threading.Thread(
        target=_download_worker,
        args=(settings.models_dir, only, tracker),
        daemon=True,
        name="model-download",
    )
    t.start()
    return {"status": "started"}


@app.get("/models/download/status")
async def download_status(request: Request) -> dict:
    return request.app.state.dl_tracker.snapshot()
