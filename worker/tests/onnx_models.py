"""Modelos ONNX minúsculos, gerados em tempo de teste, que imitam o CONTRATO de E/S dos
modelos reais (LaMa, DDColor, Real-ESRGAN). Validam o encanamento dos adaptadores
(descoberta de entradas, faixa de saída, tiles, fallback), não a qualidade visual.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = [helper.make_opsetid("", 17)]


def _save(graph, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=OPSET, ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _const(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def make_inpaint_model(path: Path, size: int = 64) -> Path:
    """Estilo LaMa: image[1,3,S,S] (0..1) + mask[1,1,S,S] -> [1,3,S,S] em 0..255.
    Pixels mascarados viram cinza médio (0.5); os demais passam inalterados."""
    nodes = [
        helper.make_node("Sub", ["one", "mask"], ["keep"]),
        helper.make_node("Mul", ["image", "keep"], ["kept"]),
        helper.make_node("Mul", ["mask", "half"], ["filled"]),
        helper.make_node("Add", ["kept", "filled"], ["mix"]),
        helper.make_node("Mul", ["mix", "k255"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_lama",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size]),
         helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, size, size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, size, size])],
        initializer=[_const("one", 1.0), _const("half", 0.5), _const("k255", 255.0)],
    )
    return _save(graph, path)


def make_upscale_model(path: Path, scale: int = 2) -> Path:
    """Estilo Real-ESRGAN com entrada dinâmica: [1,3,h,w] -> [1,3,h*s,w*s] (vizinho mais próximo)."""
    node = helper.make_node("Resize", ["input", "roi", "scales"], ["output"], mode="nearest")
    graph = helper.make_graph(
        [node], "fake_esrgan",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        initializer=[_const("roi", []), _const("scales", [1, 1, scale, scale])],
    )
    return _save(graph, path)


def make_colorize_model(path: Path, size: int = 128) -> Path:
    """Estilo DDColor: [1,3,S,S] RGB cinza -> [1,2,S,S] (a/b). ab = (canais 0..1 - 0.5) * 80."""
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["two"]),
        helper.make_node("Sub", ["two", "half"], ["centered"]),
        helper.make_node("Mul", ["centered", "k80"], ["ab"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_ddcolor",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("ab", TensorProto.FLOAT, [1, 2, size, size])],
        initializer=[
            numpy_helper.from_array(np.array([0], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "ends"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            _const("half", 0.5), _const("k80", 80.0),
        ],
    )
    return _save(graph, path)
