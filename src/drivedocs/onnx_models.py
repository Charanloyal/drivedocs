"""ONNX Runtime embedding encoders: the default, low-memory backend (no PyTorch needed).

* Text:  all-MiniLM-L6-v2 exported to ONNX (int8-quantized by default): tokens → mean pooling → L2 normalization,
  the same computation as Sentence-Transformers.
* Images/CLIP: the CLIP ViT-B/32 text and vision towers exported to ONNX (int8-quantized by default), with the
  standard CLIP preprocessing (resize shortest edge to 224 with bicubic resampling, center-crop 224×224, scale to
  [0, 1], normalize with the CLIP mean/std) implemented with Pillow + NumPy.

Model files come from the Hugging Face Hub (pre-exported by the Transformers.js project) and are cached in HF_HOME.
Quantization changes vectors slightly; see docs/RAG_DESIGN.md for the measured effect on retrieval quality.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


def _resolve(repo_or_dir: str, filename: str) -> str:
    local = Path(repo_or_dir)
    if local.is_dir():
        return str(local / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_or_dir, filename)


def _model_path(repo_or_dir: str, filename: str) -> str:
    """Path of the ONNX graph to load: the int8-embedding variant (see onnx_optimize) when enabled.

    The variant is created at build time by scripts/download_models.py. If it is missing it is created on first
    load (needs the `onnx` package and transiently ~3x the model size in RAM), else the original graph is used.
    """
    original = _resolve(repo_or_dir, filename)
    if os.getenv("DRIVEDOCS_ONNX_INT8_EMBEDDINGS", "true").strip().lower() in {"0", "false", "no", "off"}:
        return original
    from drivedocs.onnx_optimize import optimized_path, quantize_embedding_tables

    candidate = optimized_path(original)
    if candidate.exists():
        return str(candidate)
    try:
        path, _ = quantize_embedding_tables(original, candidate)
        return str(path)
    except Exception as exc:  # pragma: no cover - onnx missing or read-only cache
        import logging

        logging.getLogger("drivedocs.onnx").warning("Embedding quantization skipped for %s: %s", filename, exc)
        return original


def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    threads = int(os.getenv("DRIVEDOCS_ONNX_THREADS", "0") or 0) or min(4, os.cpu_count() or 1)
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    # Several sessions share the process; spinning worker threads would contend with each other (measured 4-10x
    # slower) and burn CPU on small shared instances.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    # The memory-pattern optimisation keeps per-shape arena buffers alive; disabling it keeps RSS flat.
    opts.enable_mem_pattern = False
    opts.enable_cpu_mem_arena = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def _tokenizer(path: str, max_length: int, pad_id: int, pad_token: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=pad_id, pad_token=pad_token)
    return tok


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype(np.float32)


class OnnxTextEncoder:
    """Sentence-Transformers-compatible encoder for BERT-style models with mean pooling."""

    def __init__(self, repo: str, model_file: str, max_length: int = 256) -> None:
        self.session = _session(_model_path(repo, model_file))
        tokenizer_json = json.loads(Path(_resolve(repo, "tokenizer.json")).read_text(encoding="utf-8"))
        pad = (tokenizer_json.get("padding") or {})
        self.tokenizer = _tokenizer(
            _resolve(repo, "tokenizer.json"), max_length, int(pad.get("pad_id", 0)), pad.get("pad_token", "[PAD]")
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: Sequence[str], batch_size: int = 32, **_: object) -> np.ndarray:
        out: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(list(texts[start : start + batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = self.session.run(None, feeds)[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(_l2_normalize(pooled))
        return np.vstack(out)


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def clip_preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """CLIP image preprocessing → float32 array (3, size, size)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.transpose(2, 0, 1)


class OnnxClipEncoder:
    """CLIP text and image towers (projected, 512-d) run with ONNX Runtime.

    The text tower is loaded eagerly (every chat searches diagrams by text). The vision tower is only needed to embed
    new images (uploads, image-to-image queries; the demo corpus index is built at build time), so it is loaded on
    first use and released after `idle_unload_seconds` without use (0 = keep loaded).
    """

    def __init__(self, repo: str, text_file: str, vision_file: str, idle_unload_seconds: Optional[float] = None) -> None:
        self.repo = repo
        self.vision_file = vision_file
        self.text_session = _session(_model_path(repo, text_file))
        # Resolve now so a missing file fails at start-up, not on the first upload.
        self.vision_path = _model_path(repo, vision_file)
        self._vision = None
        self._vision_lock = threading.Lock()
        self._unload_timer: Optional[threading.Timer] = None
        if idle_unload_seconds is None:
            idle_unload_seconds = float(os.getenv("DRIVEDOCS_VISION_IDLE_UNLOAD_SECONDS", "300") or 0)
        self.idle_unload_seconds = idle_unload_seconds
        # CLIP pads with <|endoftext|>; the model pools at the first end-of-text token, so padding is harmless.
        self.tokenizer = _tokenizer(_resolve(repo, "tokenizer.json"), 77, 49407, "<|endoftext|>")
        self.text_inputs = {i.name for i in self.text_session.get_inputs()}

    @property
    def vision_loaded(self) -> bool:
        return self._vision is not None

    def _vision_session(self):
        with self._vision_lock:
            if self._vision is None:
                self._vision = _session(self.vision_path)
            if self.idle_unload_seconds > 0:
                if self._unload_timer is not None:
                    self._unload_timer.cancel()
                self._unload_timer = threading.Timer(self.idle_unload_seconds, self.unload_vision)
                self._unload_timer.daemon = True
                self._unload_timer.start()
            return self._vision

    def unload_vision(self) -> None:
        with self._vision_lock:
            self._vision = None
            self._unload_timer = None

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(list(texts))
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
        if "attention_mask" in self.text_inputs:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return _l2_normalize(self.text_session.run(["text_embeds"], feeds)[0])

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixels = np.stack([clip_preprocess(img) for img in images]).astype(np.float32)
        session = self._vision_session()
        return _l2_normalize(session.run(["image_embeds"], {"pixel_values": pixels})[0])
