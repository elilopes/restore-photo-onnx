"""Orquestra a restauração: reparo -> colorização -> cor -> upscale -> nitidez.

Cada etapa devolve (imagem, status, engine, detalhe). Falha de etapa NÃO derruba o
job: o erro vai para o relatório de etapas e a imagem segue como estava. Modelos
ONNX ausentes ou quebrados caem no fallback clássico (ver ml/registry.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .classic.color import ToneAnalysis, analyze_tone, enhance_colors
from .classic.face_detect import analyze_faces
from .classic.masks import HybridMask, build_hybrid_mask
from .classic.safe_math import blend, to_float, to_uint8
from .classic.sharpen import unsharp_luma
from .config import Settings
from .imageio import load_image, save_image, save_mask
from .ml.lowlight import is_dark
from .ml.registry import ModelRegistry
from .progress import ProgressTracker
from .schemas import ProcessRequest, ProcessResponse, RestoreOptions, StepReport

log = logging.getLogger("worker.pipeline")

# (imagem, status, engine, detalhe)
StepResult = tuple[np.ndarray, str, str | None, str | None]


@dataclass
class _State:
    bgr: np.ndarray
    monochrome: bool
    mask: np.ndarray | None = None
    steps: list[StepReport] = field(default_factory=list)


class Pipeline:
    def __init__(self, settings: Settings, registry: ModelRegistry, progress: ProgressTracker | None = None) -> None:
        self.s = settings
        self.models = registry
        self.progress = progress or ProgressTracker()

    # ------------------------------------------------------------------ API

    def run(self, req: ProcessRequest) -> ProcessResponse:
        jid, opts = req.job_id, req.options
        t_total = perf_counter()
        try:
            self.progress.update(jid, 0.02, "load")
            t0 = perf_counter()
            path = self._input_path(req.input_name)
            bgr, (w0, h0) = load_image(path, self.s.max_side)
            h, w = bgr.shape[:2]
            tone = analyze_tone(bgr)
            state = _State(bgr=bgr, monochrome=tone.is_monochrome)
            state.steps.append(StepReport(
                name="load", status="done", ms=_ms(t0),
                detail=f"{w0}x{h0}" + (f" (reduzida para {w}x{h})" if (w, h) != (w0, h0) else "")
                       + f"; croma={tone.mean_chroma:.1f}, monocromática={tone.is_monochrome}",
            ))

            # Pré-processamento: realce de iluminação (SCI) — antes de tudo
            self._step(state, jid, 0.07, "low_light", lambda: self._low_light(state, opts),
                       enabled=opts.low_light)
            self._step(state, jid, 0.10, "repair", lambda: self._repair(state, opts), enabled=opts.repair)
            self._step(state, jid, 0.40, "colorize", lambda: self._colorize(state, opts, tone),
                       enabled=opts.colorize != "off")
            self._step(state, jid, 0.55, "enhance_colors", lambda: self._enhance(state, opts),
                       enabled=opts.enhance_colors)
            # Restauração facial (GFPGAN / CodeFormer) — após colorização
            self._step(state, jid, 0.65, "face_restore", lambda: self._face_restore(state, opts),
                       enabled=opts.face_restore)
            self._step(state, jid, 0.72, "upscale", lambda: self._upscale(state, opts), enabled=opts.upscale > 1)
            self._step(state, jid, 0.85, "sharpen", lambda: self._sharpen(state, opts), enabled=opts.sharpen)
            # Style transfer — última etapa (criativa, opcional)
            self._step(state, jid, 0.92, "style", lambda: self._style(state, opts),
                       enabled=opts.style_transfer)

            self.progress.update(jid, 0.96, "save")
            t0 = perf_counter()
            out_dir = self.s.results_dir / jid
            result_name = f"result.{opts.output_format}"
            save_image(out_dir / result_name, state.bgr, opts.output_format, self.s.jpeg_quality)
            (out_dir / ("result.png" if opts.output_format == "jpg" else "result.jpg")).unlink(missing_ok=True)

            mask_name = None
            if state.mask is not None:
                save_mask(out_dir / "mask.png", state.mask)
                mask_name = "mask.png"
            else:
                (out_dir / "mask.png").unlink(missing_ok=True)
            state.steps.append(StepReport(name="save", status="done", ms=_ms(t0), detail=result_name))

            oh, ow = state.bgr.shape[:2]
            return ProcessResponse(
                job_id=jid, result_name=result_name, mask_name=mask_name,
                input_width=w0, input_height=h0, width=ow, height=oh,
                was_monochrome=tone.is_monochrome, steps=state.steps, total_ms=_ms(t_total),
            )
        finally:
            self.progress.finish(jid)

    # ------------------------------------------------------------ internals

    def _input_path(self, name: str) -> Path:
        base = self.s.uploads_dir.resolve()
        path = (base / name).resolve()
        if path.parent != base:  # defesa extra além do regex do schema
            raise FileNotFoundError(name)
        if not path.is_file():
            raise FileNotFoundError(f"upload não encontrado: {name}")
        return path

    def _step(self, state: _State, jid: str, progress: float, name: str,
              fn: Callable[[], StepResult], enabled: bool) -> None:
        self.progress.update(jid, progress, name)
        t0 = perf_counter()
        if not enabled:
            state.steps.append(StepReport(name=name, status="skipped", detail="desativado", ms=0))
            return
        try:
            image, status, engine, detail = fn()
            state.bgr = image
        except Exception as exc:  # noqa: BLE001 - uma etapa ruim não derruba o job
            log.exception("etapa %s falhou", name)
            status, engine, detail = "failed", None, f"{type(exc).__name__}: {exc}"
        state.steps.append(StepReport(name=name, status=status, engine=engine, detail=detail, ms=_ms(t0)))

    # ---- etapas

    def _repair(self, state: _State, o: RestoreOptions) -> StepResult:
        bgr = state.bgr
        hm: HybridMask = build_hybrid_mask(bgr, o.repair_sensitivity)
        if hm.components == 0:
            return bgr, "skipped", None, "nenhum defeito detectado"
        state.mask = hm.binary

        inpainter = self.models.get("inpaint") if o.use_models else None
        engine, note = "opencv-telea", ""
        filled = None
        if inpainter is not None:
            try:
                filled = inpainter.inpaint(bgr, hm.binary)
                engine = f"onnx:{self.models.filename('inpaint')}"
            except Exception as exc:  # noqa: BLE001
                log.warning("inpainting ONNX falhou, usando Telea: %s", exc)
                note = f"; fallback após erro do modelo ({type(exc).__name__})"
        if filled is None:
            radius = max(3, round(min(bgr.shape[:2]) * 0.002))
            filled = cv2.inpaint(bgr, hm.binary, radius, cv2.INPAINT_TELEA)

        # Só a área da máscara (com borda suave) é substituída; o resto fica intacto.
        merged = to_uint8(blend(to_float(bgr), to_float(filled), hm.soft))
        return merged, "done", engine, f"{hm.components} defeitos, {hm.coverage * 100:.2f}% da imagem{note}"

    def _colorize(self, state: _State, o: RestoreOptions, tone: ToneAnalysis) -> StepResult:
        bgr = state.bgr
        if o.colorize == "auto" and not tone.is_monochrome:
            return bgr, "skipped", None, "a foto já é colorida"
        colorizer = self.models.get("colorize") if o.use_models else None
        if colorizer is None:
            return bgr, "skipped", None, (
                "modelo de colorização indisponível "
                f"(coloque {self.models.filename('colorize')} em MODELS_DIR)")
        out = colorizer.colorize(bgr)
        state.monochrome = False
        return out, "done", f"onnx:{self.models.filename('colorize')}", None

    def _enhance(self, state: _State, o: RestoreOptions) -> StepResult:
        out = enhance_colors(state.bgr, o.enhance_strength, monochrome=state.monochrome)
        return out, "done", "opencv", ("modo monocromático" if state.monochrome else None)

    def _upscale(self, state: _State, o: RestoreOptions) -> StepResult:
        bgr = state.bgr
        h, w = bgr.shape[:2]
        scale = o.upscale
        while scale > 1 and max(h, w) * scale > self.s.max_output_side:
            scale //= 2
        if scale == 1:
            return bgr, "skipped", None, f"saída passaria de {self.s.max_output_side}px"
        limited = f"; limitado a {scale}x" if scale != o.upscale else ""

        upscaler = self.models.get("superres") if o.use_models else None
        if upscaler is not None:
            try:
                out = upscaler.upscale(bgr, scale)
                return out, "done", f"onnx:{self.models.filename('superres')}", f"{scale}x{limited}"
            except Exception as exc:  # noqa: BLE001
                log.warning("super-resolução ONNX falhou, usando Lanczos: %s", exc)
                limited += f"; fallback após erro do modelo ({type(exc).__name__})"
        out = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        return out, "done", "opencv-lanczos", f"{scale}x{limited}"

    def _sharpen(self, state: _State, o: RestoreOptions) -> StepResult:
        return unsharp_luma(state.bgr, o.sharpen_strength), "done", "opencv-unsharp", None

    def _low_light(self, state: _State, o: RestoreOptions) -> StepResult:
        """Pré-processamento de iluminação com SCI para fotos escuras."""
        bgr = state.bgr
        if not is_dark(bgr):
            return bgr, "skipped", None, "foto não é escura (brilho médio acima do limiar)"
        enhancer = self.models.get("lowlight") if o.use_models else None
        if enhancer is None:
            return bgr, "skipped", None, (
                f"modelo SCI indisponível (coloque {self.models.filename('lowlight')} em MODELS_DIR)")
        from .ml.lowlight import OnnxLowlight
        assert isinstance(enhancer, OnnxLowlight)
        out = enhancer.enhance(bgr)
        return out, "done", f"onnx:{self.models.filename('lowlight')}", "realce de iluminação aplicado"

    def _face_restore(self, state: _State, o: RestoreOptions) -> StepResult:
        """Restauração facial com GFPGAN ou CodeFormer."""
        if not o.use_models:
            return state.bgr, "skipped", None, "modelos desabilitados"
        restorer = self.models.get_face_restorer(prefer=o.face_restore_model)
        if restorer is None:
            return state.bgr, "skipped", None, (
                "nenhum modelo de restauração facial disponível "
                f"(coloque {self.models.filename('face_gfpgan')} ou "
                f"{self.models.filename('face_codeformer')} em MODELS_DIR)")
        from .ml.face_restore import OnnxFaceRestorer
        assert isinstance(restorer, OnnxFaceRestorer)
        out, analysis = restorer.restore(state.bgr)
        if analysis.face_count == 0:
            return state.bgr, "skipped", None, "nenhum rosto detectado na imagem"
        engine = f"onnx:{self.models.filename('face_gfpgan' if restorer.model_type == 'gfpgan' else 'face_codeformer')}"
        detail = (f"{analysis.face_count} rosto(s), cobertura {analysis.face_coverage * 100:.1f}%")
        return out, "done", engine, detail

    def _style(self, state: _State, o: RestoreOptions) -> StepResult:
        """Style Transfer artístico (opcional)."""
        if not o.use_models:
            return state.bgr, "skipped", None, "modelos desabilitados"
        adapter = self.models.styles.get(o.style_name, strength=o.style_strength)
        if adapter is None:
            return state.bgr, "skipped", None, (
                f"modelo de estilo '{o.style_name}' indisponível "
                f"(coloque {self.s.style_model_prefix}{o.style_name}.onnx em MODELS_DIR)")
        out = adapter.apply(state.bgr)
        return out, "done", f"onnx:style_{o.style_name}", f"intensidade {o.style_strength:.0%}"


def _ms(t0: float) -> int:
    return int((perf_counter() - t0) * 1000)
