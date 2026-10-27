"""Shared, lazily-loaded embedding models with bounded inference concurrency.

* Each (model name, device) pair is loaded at most once per process and reused by every index/request.
* Loading is guarded by a lock so concurrent first requests do not load the same model twice.
* Load time and load errors are recorded for the readiness endpoint.
* A bounded semaphore limits concurrent inference calls (memory/CPU protection on small instances).
* A deterministic "mock" backend (hashing embeddings) exists for tests and must not be used in production.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from drivedocs.config import settings

logger = logging.getLogger("drivedocs.models")


class ModelLoadError(RuntimeError):
    """Raised when an embedding model cannot be initialized."""


@dataclass
class ModelState:
    key: str
    loaded: bool = False
    loading: bool = False
    load_seconds: Optional[float] = None
    error: Optional[str] = None
    objects: Tuple[Any, ...] = field(default_factory=tuple)


class ModelRegistry:
    """Process-wide cache of loaded models."""

    def __init__(self, max_concurrent_inference: int = 2) -> None:
        self._lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}
        self._states: Dict[str, ModelState] = {}
        self._inference = threading.BoundedSemaphore(max(1, max_concurrent_inference))
        self._torch_configured = False

    def _configure_torch(self) -> None:
        if self._torch_configured:
            return
        self._torch_configured = True
        if settings.torch_threads:
            try:
                import torch

                torch.set_num_threads(settings.torch_threads)
            except Exception:  # pragma: no cover - torch missing or already initialized
                pass

    def get(self, key: str, loader: Callable[[], Tuple[Any, ...]]) -> Tuple[Any, ...]:
        state = self._states.get(key)
        if state is not None and state.loaded:
            return state.objects
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
            state = self._states.setdefault(key, ModelState(key=key))
        with key_lock:
            if state.loaded:
                return state.objects
            state.loading = True
            self._configure_torch()
            started = time.perf_counter()
            try:
                objects = loader()
            except Exception as exc:
                state.error = f"{type(exc).__name__}: {exc}"[:300]
                state.loading = False
                logger.error("Failed to load model %s: %s", key, state.error)
                raise ModelLoadError(f"Model '{key.split('|')[0]}' could not be loaded.") from exc
            state.objects = objects
            state.loaded = True
            state.loading = False
            state.error = None
            state.load_seconds = round(time.perf_counter() - started, 3)
            logger.info("Loaded model %s in %.2fs", key, state.load_seconds)
            return objects

    def state(self, key: str) -> Optional[ModelState]:
        return self._states.get(key)

    def is_loaded(self, key: str) -> bool:
        state = self._states.get(key)
        return bool(state and state.loaded)

    @contextmanager
    def inference_slot(self) -> Iterator[None]:
        with self._inference:
            yield


registry = ModelRegistry(settings.max_concurrent_inference)


# ---------------------------------------------------------------------------
# Mock (hashing) embeddings: deterministic and lexical, so tests are meaningful without model downloads.
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _hash_index(token: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def mock_text_embedding(texts: List[str], dim: int) -> np.ndarray:
    from drivedocs.citations import STOPWORDS

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
        for tok in tokens:
            for feature in (tok, tok[:5]):
                idx, sign = _hash_index(feature, dim)
                out[row, idx] += sign
        norm = np.linalg.norm(out[row])
        if norm > 0:
            out[row] /= norm
        else:
            idx, _ = _hash_index("<empty>", dim)
            out[row, idx] = 1.0
    return out


def mock_image_embedding(images: List[Any], dim: int) -> np.ndarray:
    """Color-histogram features projected to `dim` with a fixed random matrix (visually similar → close)."""
    rng = np.random.RandomState(1234)
    projection = rng.randn(64, dim).astype(np.float32)
    out = np.zeros((len(images), dim), dtype=np.float32)
    for row, img in enumerate(images):
        small = np.asarray(img.convert("RGB").resize((32, 32)), dtype=np.int64) // 64
        bins = small[..., 0] * 16 + small[..., 1] * 4 + small[..., 2]
        hist = np.bincount(bins.ravel(), minlength=64).astype(np.float32)
        hist /= hist.sum() or 1.0
        vec = hist @ projection
        out[row] = vec / (np.linalg.norm(vec) or 1.0)
    return out
