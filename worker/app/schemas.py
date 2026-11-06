"""Contrato HTTP do worker. O espelho em TypeScript fica em api/src/schemas.ts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
FILENAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class RestoreOptions(BaseModel):
    repair: bool = True                         # remover rasgos/riscos
    repair_sensitivity: float = Field(0.5, ge=0, le=1)
    colorize: Literal["auto", "on", "off"] = "auto"
    enhance_colors: bool = True                 # realçar cores fracas/desbotadas
    enhance_strength: float = Field(0.5, ge=0, le=1)
    sharpen: bool = True
    sharpen_strength: float = Field(0.5, ge=0, le=1)
    upscale: Literal[1, 2, 4] = 1
    output_format: Literal["png", "jpg"] = "png"
    use_models: bool = True                     # False = só pipeline clássico
    # Novos modelos
    low_light: bool = False                     # pré-processamento para fotos escuras (SCI)
    face_restore: bool = False                  # restauração facial (GFPGAN ou CodeFormer)
    face_restore_model: Literal["gfpgan", "codeformer"] = "gfpgan"
    style_transfer: bool = False                # transferência de estilo artístico
    style_name: str = "candy"                   # nome do estilo (candy, mosaic, rain_princess, udnie)
    style_strength: float = Field(0.7, ge=0, le=1)


class AnalyzeFaceResponse(BaseModel):
    """Resultado da análise de rostos na imagem."""
    face_count: int
    face_coverage: float          # proporção da área de rostos em relação à imagem (0..1)
    suggest_face_restore: bool    # True se deve sugerir restauração facial ao usuário


class ProcessRequest(BaseModel):
    job_id: str = Field(pattern=UUID_PATTERN)
    input_name: str = Field(pattern=FILENAME_PATTERN)  # arquivo em STORAGE_DIR/uploads
    options: RestoreOptions = Field(default_factory=RestoreOptions)


class StepReport(BaseModel):
    name: str
    status: Literal["done", "skipped", "failed"]
    engine: str | None = None
    detail: str | None = None
    ms: int = 0


class ProcessResponse(BaseModel):
    job_id: str
    result_name: str
    mask_name: str | None = None
    input_width: int
    input_height: int
    width: int
    height: int
    was_monochrome: bool
    steps: list[StepReport]
    total_ms: int


class ProgressResponse(BaseModel):
    progress: float   # 0..1
    step: str
