#!/usr/bin/env python3
"""Baixa os modelos ONNX UMA vez para a pasta de modelos.

Roda só quando você pede (não no startup do sistema). Arquivos já existentes
são pulados; use --force para baixar de novo.

Uso (Windows, PowerShell, na raiz do projeto):
    python scripts\\download_models.py
    python scripts\\download_models.py --dir models --only colorize
Com Docker (sem precisar de Python instalado):
    docker compose --profile setup run --rm models
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# (URL de origem, papel no sistema). Os nomes locais batem com os padrões de worker/app/config.py.
# As URLs seguem o padrão `resolve/main` do Hugging Face. Se alguma mudar (404), ajuste aqui.
MODELS: dict[str, tuple[str, str]] = {
    # ── Modelos principais ──────────────────────────────────────────────────
    "lama_fp32.onnx": (
        "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        "inpaint",  # remove rasgos e riscos
    ),
    "ddcolor-tiny-fp16.onnx": (
        "https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx",
        "colorize",  # colorização de P&B
    ),
    "realesrgan_x4plus.onnx": (
        "https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/blob/01179a4da7bf5ac91faca650e6afbf282ac93933/Real-ESRGAN-x4plus.onnx",
        "superres",  # aumento de resolução (principal)
    ),
    # ── Fallback / modelos adicionais ───────────────────────────────────────
    "swinir_x4.onnx": (
        "https://huggingface.co/Tomoali/SwinIR-M-x4-realSR-onnx/blob/main/swinir_nchw_256x256_fp32.onnx",
        "superres_fb",  # fallback do Real-ESRGAN
    ),
    # ── Restauração facial ──────────────────────────────────────────────────
    "gfpgan_v1.4.onnx": (
        "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/blob/main/GFPGANv1.4.onnx",
        "face_gfpgan",  # restaura rostos (rápido)
    ),
    "codeformer.onnx": (
        "https://huggingface.co/Oil3/faster_codeformer_onnx/blob/main/Codeformer.onnx",
        "face_codeformer",  # restaura rostos (mais detalhado)
    ),
    # ── Pré-processamento ───────────────────────────────────────────────────
    "sci_medium.onnx": (
        "https://github.com/mdturp/low-light-image-enhancement/blob/main/model/model.onnx",
        "lowlight",  # realce de iluminação em fotos escuras sci_medium.onnx
    ),
    # ── Estilo artístico ────────────────────────────────────────────────────
    "style_candy.onnx": (
        "https://huggingface.co/xjyplayer/candy_style/blob/main/candy_style_v1.tgz",
        "style",  # estilo Candy
    ),
    "style_mosaic.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        "style",  # estilo Mosaico
    ),
    "style_rain_princess.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
        "style",  # estilo Rain Princess
    ),
    "style_udnie.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
        "style",  # estilo Udnie
    ),
}

MIN_VALID_BYTES = 1_000_000  # abaixo disso é página de erro, não modelo


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "restore-photo-onnx/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size < MIN_VALID_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("arquivo baixado é muito pequeno; a URL pode ter mudado")
    os.replace(tmp, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=os.environ.get("MODELS_DIR", "models"))
    parser.add_argument("--only", choices=["inpaint", "colorize", "superres"], help="baixa só um papel")
    parser.add_argument("--force", action="store_true", help="baixa mesmo se já existir")
    args = parser.parse_args()

    target = Path(args.dir)
    failures = 0
    for filename, (url, role) in MODELS.items():
        if args.only and role != args.only:
            continue
        dest = target / filename
        if dest.is_file() and dest.stat().st_size >= MIN_VALID_BYTES and not args.force:
            print(f"[ok]    {filename} já existe ({dest.stat().st_size / 1e6:.0f} MB), pulando")
            continue
        print(f"[baixa] {filename} ({role})")
        try:
            download(url, dest)
            print(f"[pronto] {dest}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[erro]  {filename}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{failures} modelo(s) falharam. O sistema continua funcionando com o fallback clássico.")
        return 1
    print("\nModelos prontos. Não é preciso baixar de novo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
