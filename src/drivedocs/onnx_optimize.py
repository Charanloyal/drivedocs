"""Build-time ONNX graph optimization: int8 embedding tables.

The largest tensors in the text models are token-embedding lookup tables (CLIP text: 49,408×512 fp32 ≈ 97 MB;
MiniLM: 30,522×384 ≈ 45 MB). A lookup table can be stored as int8 with one fp32 scale per row and de-quantized
after the lookup, which cuts its memory ~4× while changing each looked-up vector by < 0.4% (max abs error
scale/2 per element). All matrix multiplications are left untouched.

    Gather(W_fp32, ids)  →  Mul(Cast(Gather(W_int8, ids), float), Gather(row_scale, ids))

The optimized graph is written next to the downloaded model (``*.emb8.onnx``) by scripts/download_models.py at
build time, so the server never holds the original and the optimized model in memory at the same time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("drivedocs.onnx_optimize")

SUFFIX = ".emb8.onnx"
MIN_ELEMENTS = 1_000_000


def optimized_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_name(path.name[: -len(".onnx")] + SUFFIX)


def quantize_embedding_tables(src: str | Path, dst: Optional[str | Path] = None) -> tuple[Path, int]:
    """Write a copy of `src` whose large Gather tables are int8; returns (path, number of tables quantized)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    src = Path(src)
    dst = Path(dst) if dst else optimized_path(src)
    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    graph_inputs = {i.name for i in graph.input}

    new_nodes = []
    quantized = 0
    replaced_tables = set()
    for node in graph.node:
        table = inits.get(node.input[0]) if node.op_type == "Gather" else None
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        if (
            table is None
            or axis != 0
            or table.data_type != TensorProto.FLOAT
            or len(table.dims) != 2
            or int(np.prod(table.dims)) < MIN_ELEMENTS
            or node.input[1] not in graph_inputs
        ):
            new_nodes.append(node)
            continue
        weights = numpy_helper.to_array(table).astype(np.float32)
        scale = np.abs(weights).max(axis=1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        base = node.input[0]
        q_name, s_name = f"{base}__int8", f"{base}__row_scale"
        graph.initializer.append(numpy_helper.from_array(q, q_name))
        graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), s_name))
        out = node.output[0]
        new_nodes += [
            helper.make_node("Gather", [q_name, node.input[1]], [f"{out}__q"], name=f"{node.name}__q", axis=0),
            helper.make_node("Cast", [f"{out}__q"], [f"{out}__f"], name=f"{node.name}__cast", to=TensorProto.FLOAT),
            helper.make_node("Gather", [s_name, node.input[1]], [f"{out}__s"], name=f"{node.name}__scale", axis=0),
            helper.make_node("Mul", [f"{out}__f", f"{out}__s"], [out], name=f"{node.name}__dequant"),
        ]
        replaced_tables.add(base)
        quantized += 1

    if not quantized:
        return src, 0
    still_used = {i for n in new_nodes for i in n.input}
    kept = [i for i in graph.initializer if i.name not in replaced_tables or i.name in still_used]
    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    tmp = dst.with_name(dst.name + ".tmp")
    onnx.save(model, str(tmp))
    tmp.replace(dst)
    logger.info("Quantized %d embedding table(s): %s -> %s", quantized, src.name, dst.name)
    return dst, quantized
