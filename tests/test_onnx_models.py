"""ONNX backend: preprocessing and encoder behaviour (real ONNX models, skipped when not cached)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from drivedocs.onnx_models import CLIP_MEAN, CLIP_STD, clip_preprocess


def test_clip_preprocess_shape_and_normalization():
    arr = clip_preprocess(Image.new("RGB", (640, 320), (255, 0, 0)))
    assert arr.shape == (3, 224, 224) and arr.dtype == np.float32
    expected_red = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0]
    assert np.allclose(arr[0], expected_red, atol=1e-4)
    assert np.allclose(arr[1], (0.0 - CLIP_MEAN[1]) / CLIP_STD[1], atol=1e-4)


def test_clip_preprocess_center_crops_shortest_edge():
    img = Image.new("RGB", (300, 100), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 50, 100))  # white strip at the far left is cropped away
    arr = clip_preprocess(img)
    assert np.allclose(arr[0], (0.0 - CLIP_MEAN[0]) / CLIP_STD[0], atol=1e-3)


@pytest.mark.real_models
def test_onnx_text_encoder_semantics():
    if os.environ.get("DRIVEDOCS_TEST_REAL_BACKEND") != "onnx":
        pytest.skip("ONNX backend not selected for real-model tests")
    from drivedocs.text_index import TextEmbeddingModel

    model = TextEmbeddingModel(backend="onnx")
    vecs = model.encode(["The Raft leader sends heartbeats every 150 ms.",
                         "Leader heartbeat interval in Raft consensus",
                         "Chocolate chip cookie recipe"])
    assert vecs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2] + 0.3


@pytest.mark.real_models
def test_onnx_clip_text_image_alignment():
    if os.environ.get("DRIVEDOCS_TEST_REAL_BACKEND") != "onnx":
        pytest.skip("ONNX backend not selected for real-model tests")
    from drivedocs.image_index import CLIPEmbeddingModel

    model = CLIPEmbeddingModel(backend="onnx")
    imgs = model.encode_images([Image.new("RGB", (100, 100), (0, 0, 255)), Image.new("RGB", (100, 100), (255, 0, 0))])
    texts = model.encode_text(["a solid blue square", "a solid red square"])
    assert imgs.shape == (2, 512) and texts.shape == (2, 512)
    sims = texts @ imgs.T
    assert sims[0, 0] > sims[0, 1] and sims[1, 1] > sims[1, 0]


def test_int8_embedding_quantization_preserves_lookups(tmp_path):
    onnx = pytest.importorskip("onnx")
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from drivedocs.onnx_optimize import optimized_path, quantize_embedding_tables

    rng = np.random.default_rng(0)
    table = rng.normal(size=(2500, 512)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0),
         helper.make_node("ReduceMean", ["emb"], ["out"], axes=[1], keepdims=0)],
        "g",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "t"])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["b", 512])],
        [numpy_helper.from_array(table, "table")],
    )
    src = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(src))
    dst, n = quantize_embedding_tables(src)
    assert n == 1 and dst == optimized_path(src) and dst.name == "model.emb8.onnx"
    assert dst.stat().st_size < src.stat().st_size / 3
    ids = np.array([[1, 7, 2499, 42]], dtype=np.int64)
    a = ort.InferenceSession(str(src)).run(None, {"input_ids": ids})[0]
    b = ort.InferenceSession(str(dst)).run(None, {"input_ids": ids})[0]
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999


@pytest.mark.real_models
def test_clip_vision_tower_is_lazy_and_unloads():
    if os.environ.get("DRIVEDOCS_TEST_REAL_BACKEND") != "onnx":
        pytest.skip("ONNX backend not selected for real-model tests")
    from drivedocs.onnx_models import OnnxClipEncoder
    from drivedocs.config import settings

    enc = OnnxClipEncoder(settings.image_onnx_repo, settings.image_onnx_text_file, settings.image_onnx_vision_file,
                          idle_unload_seconds=0.2)
    assert not enc.vision_loaded
    enc.encode_text(["fuse box"])
    assert not enc.vision_loaded
    enc.encode_images([Image.new("RGB", (64, 64), (0, 0, 0))])
    assert enc.vision_loaded
    import time
    time.sleep(0.6)
    assert not enc.vision_loaded
