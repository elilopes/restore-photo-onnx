"""Configuração do worker (variáveis de ambiente)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    storage_dir: Path
    models_dir: Path
    max_side: int            # lado máximo da entrada (protege a memória)
    max_output_side: int     # lado máximo da saída após upscale
    jpeg_quality: int
    max_concurrency: int     # jobs simultâneos dentro deste processo
    ort_providers: tuple[str, ...]
    ort_threads: int         # 0 = padrão do onnxruntime
    inpaint_model: str
    colorize_model: str
    superres_model: str
    superres_scale: int      # escala nativa do modelo de super-resolução
    inpaint_tile: int        # usado só se o modelo tiver entrada dinâmica
    superres_tile: int
    colorize_size: int
    tile_overlap: int
    # Novos modelos
    swinir_model: str            # fallback de super-resolução (SwinIR)
    face_restore_gfpgan: str     # restauração facial GFPGAN
    face_restore_codeformer: str # restauração facial CodeFormer
    lowlight_model: str          # pré-processamento low-light (SCI)
    style_model_prefix: str      # prefixo para modelos de style transfer
    face_detect_min_size: int    # tamanho mínimo de rosto para detecção (px)

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / "results"


@lru_cache
def get_settings() -> Settings:
    providers = _str("ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
    return Settings(
        storage_dir=Path(_str("STORAGE_DIR", "../storage")).resolve(),
        models_dir=Path(_str("MODELS_DIR", "./models")).resolve(),
        max_side=_int("MAX_SIDE", 4096),
        max_output_side=_int("MAX_OUTPUT_SIDE", 8192),
        jpeg_quality=_int("JPEG_QUALITY", 95),
        max_concurrency=max(1, _int("MAX_CONCURRENCY", 1)),
        ort_providers=tuple(p.strip() for p in providers.split(",") if p.strip()),
        ort_threads=_int("ORT_INTRA_THREADS", 0),
        inpaint_model=_str("INPAINT_MODEL", "lama_fp32.onnx"),
        colorize_model=_str("COLORIZE_MODEL", "ddcolor-tiny-fp16.onnx"),
        superres_model=_str("SUPERRES_MODEL", "realesrgan_x4plus.onnx"),
        superres_scale=_int("SUPERRES_SCALE", 4),
        inpaint_tile=_int("INPAINT_TILE", 512),
        superres_tile=_int("SUPERRES_TILE", 256),
        colorize_size=_int("COLORIZE_SIZE", 512),
        tile_overlap=_int("TILE_OVERLAP", 32),
        swinir_model=_str("SWINIR_MODEL", "swinir_x4.onnx"),
        face_restore_gfpgan=_str("FACE_RESTORE_GFPGAN", "gfpgan_v1.4.onnx"),
        face_restore_codeformer=_str("FACE_RESTORE_CODEFORMER", "codeformer.onnx"),
        lowlight_model=_str("LOWLIGHT_MODEL", "sci_medium.onnx"),
        style_model_prefix=_str("STYLE_MODEL_PREFIX", "style_"),
        face_detect_min_size=_int("FACE_DETECT_MIN_SIZE", 60),
    )
