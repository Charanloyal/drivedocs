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


class _FakeSession:
    def __init__(self, path):
        self.path = path

    def get_inputs(self):
        return []

    def run(self, names, feeds):
        return [np.ones((len(feeds["pixel_values"]), 512), dtype=np.float32)]


@pytest.fixture
def fake_clip(monkeypatch):
    import drivedocs.onnx_models as om

    created = []

    def session(path):
        created.append(path)
        return _FakeSession(path)

    monkeypatch.setattr(om, "_session", session)
    monkeypatch.setattr(om, "_resolve", lambda repo, name: f"{repo}/{name}")
    monkeypatch.setattr(om, "_tokenizer", lambda *a, **k: None)
    return om, created


def test_clip_vision_tower_is_lazy_and_unloads_when_idle(fake_clip):
    import time

    om, created = fake_clip
    enc = om.OnnxClipEncoder("repo", "text.onnx", "vision.onnx", idle_unload_seconds=0.2)
    assert created == ["repo/text.onnx"] and not enc.vision_loaded  # only the text tower at start-up
    out = enc.encode_images([Image.new("RGB", (64, 64))])
    assert out.shape == (1, 512) and enc.vision_loaded and enc.vision_loads == 1
    time.sleep(0.5)
    assert not enc.vision_loaded  # released after the idle period
    enc.encode_images([Image.new("RGB", (64, 64))])
    assert enc.vision_loaded and enc.vision_loads == 2  # transparently reloaded
    enc.unload_vision()
    assert not enc.vision_loaded


def test_clip_vision_tower_loads_once_under_concurrency_and_can_stay_loaded(fake_clip):
    import threading
    import time

    om, created = fake_clip
    enc = om.OnnxClipEncoder("repo", "text.onnx", "vision.onnx", idle_unload_seconds=0)
    threads = [threading.Thread(target=enc.encode_images, args=([Image.new("RGB", (32, 32))],)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert enc.vision_loads == 1 and created.count("repo/vision.onnx") == 1
    time.sleep(0.2)
    assert enc.vision_loaded  # idle_unload_seconds=0 keeps it loaded
