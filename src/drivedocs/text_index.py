"""Text embedding (Sentence Transformers) and FAISS text index."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from drivedocs.config import settings
from drivedocs.models import mock_text_embedding, registry
from drivedocs.schemas import TextChunk, TextSearchResult
from drivedocs.storage import atomic_write_json

logger = logging.getLogger("drivedocs.text_index")


def write_faiss_atomic(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        faiss.write_index(index, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class NativeSentenceEncoder:
    """Runs a Sentence-Transformers model with plain `transformers` (Transformer → Pooling → optional Normalize).

    Importing `sentence_transformers` also imports scikit-learn and SciPy (~180 MB of extra RSS, measured), which
    DriveDocs does not need. This encoder reproduces the same computation from the model's own configuration files.
    It refuses models with other module types (e.g. Dense layers); those use the full library instead.
    """

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        modules = _read_model_json(model_name, "modules.json")
        if not modules:
            raise UnsupportedNativeModel("modules.json not found")
        kinds = [m.get("type", "").rsplit(".", 1)[-1] for m in modules]
        if kinds[:2] != ["Transformer", "Pooling"] or any(k not in ("Transformer", "Pooling", "Normalize") for k in kinds):
            raise UnsupportedNativeModel(f"unsupported module chain {kinds}")
        pooling_path = next(m.get("path", "") for m in modules if m.get("type", "").endswith("Pooling"))
        pooling = _read_model_json(model_name, f"{pooling_path}/config.json") or {}
        mode_keys = {"mean": "pooling_mode_mean_tokens", "cls": "pooling_mode_cls_token", "max": "pooling_mode_max_tokens"}
        active = [k for k, v in pooling.items() if k.startswith("pooling_mode_") and v]
        modes = [mode for mode, key in mode_keys.items() if key in active]
        if len(active) != 1 or len(modes) != 1:
            raise UnsupportedNativeModel(f"unsupported pooling {active}")
        st_config = _read_model_json(model_name, "sentence_bert_config.json") or {}

        self.torch = torch
        self.pooling = modes[0]
        self.normalize = "Normalize" in kinds
        self.max_length = int(st_config.get("max_seq_length") or 512)
        self.do_lower_case = bool(st_config.get("do_lower_case", False))
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True, **_: object) -> np.ndarray:
        torch = self.torch
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if self.do_lower_case:
                batch = [t.lower() for t in batch]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                tokens = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
            if self.pooling == "mean":
                emb = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            elif self.pooling == "cls":
                emb = tokens[:, 0]
            else:
                emb = tokens.masked_fill(mask == 0, -1e9).max(dim=1).values
            if self.normalize or normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outputs.append(emb.float().cpu().numpy())
        return np.vstack(outputs)


class UnsupportedNativeModel(ValueError):
    pass


def _read_model_json(model_name: str, filename: str):
    """Read a JSON file from a local model directory or the Hugging Face cache/Hub; None if it does not exist."""
    local = Path(model_name)
    if local.is_dir():
        path = local / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(model_name, filename)).read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_backend(backend: Optional[str], mock_mode: Optional[bool]) -> str:
    """Explicit backend > mock_mode flag > configured backend. mock_mode=False means "a real model" (ONNX if the
    configured backend is the mock one, so real-model tests work under the mock test configuration)."""
    if backend:
        return backend.lower()
    if mock_mode is True:
        return "mock"
    if mock_mode is False and settings.embedding_backend == "mock":
        return "onnx"
    return settings.embedding_backend


class TextEmbeddingModel:
    """Text embedding model shared process-wide via the model registry.

    Backends: "onnx" (default; ONNX Runtime, no PyTorch), "transformers" (PyTorch; `encoder="native"` runs the
    model with plain transformers, `encoder="sentence-transformers"` uses that library) and "mock" (tests).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: int = 384,
        mock_mode: Optional[bool] = None,
        batch_size: Optional[int] = None,
        encoder: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.text_model_name
        self.device = device or settings.device
        self.dimension = dimension
        self.backend = resolve_backend(backend, mock_mode)
        self.mock_mode = self.backend == "mock"
        self.batch_size = batch_size or settings.text_batch_size
        self.encoder = (encoder or settings.text_encoder).lower()
        self.onnx_repo = settings.text_onnx_repo
        self.onnx_file = settings.text_onnx_file

    @property
    def registry_key(self) -> str:
        if self.backend == "onnx":
            return f"{self.onnx_repo}/{self.onnx_file}|onnx|text"
        return f"{self.model_name}|{self.device}|text|{self.encoder}"

    @property
    def identity(self) -> str:
        if self.mock_mode:
            return f"mock-hash-{self.dimension}"
        if self.backend == "onnx":
            return f"{self.onnx_repo}/{self.onnx_file}"
        return self.model_name

    @property
    def is_loaded(self) -> bool:
        return self.mock_mode or registry.is_loaded(self.registry_key)

    # Backwards compatibility: the API used to inspect `_model` directly.
    @property
    def _model(self):
        state = registry.state(self.registry_key)
        return state.objects[0] if state and state.loaded else None

    def _load(self):
        def loader():
            if self.backend == "onnx":
                from drivedocs.onnx_models import OnnxTextEncoder

                return (OnnxTextEncoder(self.onnx_repo, self.onnx_file, max_length=settings.text_max_length),)
            if self.encoder == "native":
                try:
                    return (NativeSentenceEncoder(self.model_name, self.device),)
                except UnsupportedNativeModel as exc:
                    logger.info("Native encoder unsupported for %s (%s); using sentence-transformers.", self.model_name, exc)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    f"Text model '{self.model_name}' needs the optional sentence-transformers package "
                    "(pip install 'drivedocs[sentence-transformers]'); the bundled native encoder only supports "
                    "Transformer + Pooling (+ Normalize) models such as all-MiniLM-L6-v2."
                ) from exc

            return (SentenceTransformer(self.model_name, device=self.device),)

        return registry.get(self.registry_key, loader)[0]

    def warm_up(self) -> None:
        if not self.mock_mode:
            self._load()

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into float32 embeddings (L2-normalized by default)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.mock_mode:
            return mock_text_embedding(texts, self.dimension)
        model = self._load()
        with registry.inference_slot():
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Text model '{self.model_name}' produced {vectors.shape[1]}-d vectors; expected {self.dimension}."
            )
        return vectors


class TextFAISSIndex:
    """FAISS IndexFlatIP over L2-normalized vectors (inner product == cosine similarity)."""

    INDEX_FILE = "text_index.faiss"
    METADATA_FILE = "text_metadata.json"

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_model: Optional[TextEmbeddingModel] = None,
    ) -> None:
        self.dimension = dimension or settings.text_dimension
        self.embedding_model = embedding_model or TextEmbeddingModel(dimension=self.dimension)
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[TextChunk] = []
        self._lock = threading.RLock()

    def count(self) -> int:
        return len(self.chunks)

    def encode_chunks(self, chunks: List[TextChunk]) -> np.ndarray:
        vectors = self.embedding_model.encode([c.embedding_text() for c in chunks], normalize=True)
        faiss.normalize_L2(vectors)
        return vectors

    def add_embedded(self, chunks: List[TextChunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError("vector/chunk count mismatch")
        with self._lock:
            self.index.add(vectors)
            self.chunks.extend(chunks)
        return len(chunks)

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Embed and append chunks. Embedding happens before the index is touched (no partial updates)."""
        if not chunks:
            return 0
        return self.add_embedded(chunks, self.encode_chunks(chunks))

    def remove_doc(self, doc_id: str) -> int:
        """Remove all chunks of a document by rebuilding the flat index from the kept vectors (no re-embedding)."""
        with self._lock:
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            removed = len(self.chunks) - len(keep)
            if removed == 0:
                return 0
            vectors = self.index.reconstruct_n(0, self.index.ntotal) if self.index.ntotal else None
            new_index = faiss.IndexFlatIP(self.dimension)
            if keep and vectors is not None:
                new_index.add(np.ascontiguousarray(vectors[keep]))
            self.index = new_index
            self.chunks = [self.chunks[i] for i in keep]
            return removed

    def reembed_all(self) -> None:
        """Recompute every vector (e.g. after the configured text model changed)."""
        with self._lock:
            chunks = list(self.chunks)
            new_index = faiss.IndexFlatIP(self.dimension)
            if chunks:
                new_index.add(self.encode_chunks(chunks))
            self.index = new_index

    def search(self, query: str, top_k: int = 4) -> List[TextSearchResult]:
        """Return the top-k chunks by cosine similarity."""
        if not query or not query.strip() or self.count() == 0:
            return []
        q_vec = self.embedding_model.encode([query], normalize=True)
        faiss.normalize_L2(q_vec)
        with self._lock:
            k = min(max(1, top_k), self.count())
            scores, indices = self.index.search(q_vec, k)
            chunks = self.chunks
            results: List[TextSearchResult] = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx < 0 or idx >= len(chunks):
                    continue
                results.append(
                    TextSearchResult(chunk=chunks[idx], score=float(np.clip(score, 0.0, 1.0)), rank=rank)
                )
        return results

    def save(self, directory: Path | str) -> None:
        target_dir = Path(directory)
        with self._lock:
            write_faiss_atomic(self.index, target_dir / self.INDEX_FILE)
            atomic_write_json(target_dir / self.METADATA_FILE, [c.model_dump() for c in self.chunks])

    def load(self, directory: Path | str) -> bool:
        """Load the index; returns False if files are missing or inconsistent with each other."""
        target_dir = Path(directory)
        faiss_file = target_dir / self.INDEX_FILE
        metadata_file = target_dir / self.METADATA_FILE
        if not faiss_file.exists() or not metadata_file.exists():
            return False
        index = faiss.read_index(str(faiss_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            chunks = [TextChunk(**item) for item in json.load(f)]
        if index.d != self.dimension or index.ntotal != len(chunks):
            return False
        with self._lock:
            self.index = index
            self.chunks = chunks
        return True
