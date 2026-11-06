"""Registro dos modelos ONNX: carrega sob demanda e degrada com elegância.

Se o arquivo do modelo não existe, ou falha ao carregar (arquivo corrompido,
operador não suportado pelo provedor), `get()` devolve None e o pipeline usa o
fallback clássico. Nenhum job falha só porque falta um modelo.

Chains de fallback:
  superres: Real-ESRGAN → SwinIR → Lanczos (OpenCV)
  face:     GFPGAN → CodeFormer → (sem restauração)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

from ..config import Settings
from .colorize import OnnxColorizer
from .face_restore import OnnxFaceRestorer
from .inpaint import OnnxInpainter
from .lowlight import OnnxLowlight
from .session import OnnxModel, select_providers
from .style_transfer import StyleTransferRegistry
from .superres import OnnxUpscaler
from .swinir import SwinIRUpscaler

Kind = Literal["inpaint", "colorize", "superres", "superres_fb", "face_gfpgan", "face_codeformer", "lowlight"]
KINDS: tuple[Kind, ...] = ("inpaint", "colorize", "superres", "superres_fb", "face_gfpgan", "face_codeformer", "lowlight")

log = logging.getLogger("worker.models")


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._providers = select_providers(settings.ort_providers)
        self._adapters: dict[str, object] = {}
        self._broken: dict[str, str] = {}
        self._lock = threading.Lock()
        # Style Transfer registry (múltiplos modelos)
        self.styles = StyleTransferRegistry(
            settings.models_dir, settings.style_model_prefix,
            self._providers, settings.ort_threads,
        )

    def _model_file(self, kind: Kind) -> str:
        return {
            "inpaint": self._s.inpaint_model,
            "colorize": self._s.colorize_model,
            "superres": self._s.superres_model,
            "superres_fb": self._s.swinir_model,
            "face_gfpgan": self._s.face_restore_gfpgan,
            "face_codeformer": self._s.face_restore_codeformer,
            "lowlight": self._s.lowlight_model,
        }[kind]

    def filename(self, kind: Kind) -> str:
        return self._model_file(kind)

    def path(self, kind: Kind) -> Path:
        return self._s.models_dir / self._model_file(kind)

    def _build(self, kind: Kind, model: OnnxModel) -> object:
        s = self._s
        if kind == "inpaint":
            return OnnxInpainter(model, s.inpaint_tile, overlap=64)
        if kind == "colorize":
            return OnnxColorizer(model, s.colorize_size)
        if kind == "superres":
            return OnnxUpscaler(model, s.superres_scale, s.superres_tile, s.tile_overlap)
        if kind == "superres_fb":
            return SwinIRUpscaler(model, s.superres_scale, s.superres_tile, s.tile_overlap)
        if kind == "face_gfpgan":
            return OnnxFaceRestorer(model, model_type="gfpgan",
                                    min_face_size=s.face_detect_min_size)
        if kind == "face_codeformer":
            return OnnxFaceRestorer(model, model_type="codeformer",
                                    min_face_size=s.face_detect_min_size)
        if kind == "lowlight":
            return OnnxLowlight(model)
        raise ValueError(f"kind desconhecido: {kind}")

    def get(self, kind: Kind) -> object | None:
        """Adaptador pronto para uso, ou None se o modelo está ausente/inutilizável."""
        with self._lock:
            if kind in self._adapters:
                return self._adapters[kind]
            if kind in self._broken or not self.path(kind).is_file():
                return None
            try:
                model = OnnxModel(self.path(kind), self._providers, self._s.ort_threads)
                adapter = self._build(kind, model)
                self._adapters[kind] = adapter
                log.info("modelo %s carregado: %s (%s)", kind, self.path(kind).name, model.providers_in_use)
                return adapter
            except Exception as exc:  # noqa: BLE001
                self._broken[kind] = f"{type(exc).__name__}: {exc}"
                log.warning("modelo %s inutilizável, usando fallback: %s", kind, self._broken[kind])
                return None

    def get_upscaler(self) -> OnnxUpscaler | SwinIRUpscaler | None:
        """Retorna Real-ESRGAN ou, se ausente, SwinIR (fallback)."""
        primary = self.get("superres")
        if primary is not None:
            return primary  # type: ignore[return-value]
        fallback = self.get("superres_fb")
        if fallback is not None:
            log.info("usando SwinIR como fallback de super-resolução")
        return fallback  # type: ignore[return-value]

    def get_face_restorer(self, prefer: str = "gfpgan") -> OnnxFaceRestorer | None:
        """Retorna o restaurador facial preferido ou o alternativo."""
        primary_kind: Kind = "face_gfpgan" if prefer == "gfpgan" else "face_codeformer"
        fallback_kind: Kind = "face_codeformer" if prefer == "gfpgan" else "face_gfpgan"
        result = self.get(primary_kind)
        if result is not None:
            return result  # type: ignore[return-value]
        return self.get(fallback_kind)  # type: ignore[return-value]

    def status(self) -> dict:
        """Retorna status de todos os modelos para o endpoint /models."""
        all_kinds: list[Kind] = ["inpaint", "colorize", "superres", "superres_fb",
                                  "face_gfpgan", "face_codeformer", "lowlight"]
        models_status = []
        for kind in all_kinds:
            adapter = self._adapters.get(kind)
            models_status.append({
                "name": self.filename(kind),
                "role": kind,
                "present": self.path(kind).is_file(),
                "loaded": adapter is not None,
                "error": self._broken.get(kind),
                "providers": adapter.model.providers_in_use if adapter is not None and hasattr(adapter, "model") else None,
            })
        # Estilos
        for style in ("candy", "mosaic", "rain_princess", "udnie"):
            path = self._s.models_dir / f"{self._s.style_model_prefix}{style}.onnx"
            models_status.append({
                "name": f"{self._s.style_model_prefix}{style}.onnx",
                "role": f"style_{style}",
                "present": path.is_file(),
                "loaded": False,
                "error": None,
                "providers": None,
            })
        return {"providers": self._providers, "models": models_status}
