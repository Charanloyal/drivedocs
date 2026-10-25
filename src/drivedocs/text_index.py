"""Text embedding and FAISS vector index module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import faiss

from drivedocs.config import settings
from drivedocs.schemas import TextChunk, TextSearchResult


class TextEmbeddingModel:
    """Wrapper around Hugging Face SentenceTransformer with CPU/GPU device handling."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: int = 384,
        mock_mode: bool = False,
    ) -> None:
        self.model_name = model_name or settings.text_model_name
        self.device = device or settings.device
        self.dimension = dimension
        self.mock_mode = mock_mode
        self._model = None

    def _get_model(self):
        if self.mock_mode:
            return None
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode a list of text strings into float32 dense embeddings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.mock_mode:
            # Deterministic, repeatable pseudo-embeddings for fast isolated unit tests
            embeddings = []
            for t in texts:
                # Seed based on text hash to ensure consistent similarity behavior in tests
                import hashlib
                h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
                rng = np.random.RandomState(h % (2**31))
                vec = rng.randn(self.dimension).astype(np.float32)
                if normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                embeddings.append(vec)
            return np.vstack(embeddings).astype(np.float32)

        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=settings.text_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return vectors.astype(np.float32)


class TextFAISSIndex:
    """FAISS-based vector index for semantic document chunk retrieval."""

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_model: Optional[TextEmbeddingModel] = None,
    ) -> None:
        self.dimension = dimension or settings.text_dimension
        self.embedding_model = embedding_model or TextEmbeddingModel(dimension=self.dimension)
        # Using IndexFlatIP for Inner Product. On L2-normalized vectors, IP == Cosine Similarity.
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[TextChunk] = []

    def count(self) -> int:
        """Total number of indexed text chunks."""
        return len(self.chunks)

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Embed and append new text chunks into the FAISS index."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.embedding_model.encode(texts, normalize=True)

        # Ensure unit L2 normalization for exact cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[TextSearchResult]:
        """Search top-k text chunks semantically similar to the query string."""
        if not query.strip() or self.count() == 0:
            return []

        k = min(top_k, self.count())
        q_vec = self.embedding_model.encode([query], normalize=True)
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, k)

        results: List[TextSearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            # Clip cosine similarity score to [0.0, 1.0] for clean reporting
            norm_score = float(np.clip(score, 0.0, 1.0))
            results.append(
                TextSearchResult(
                    chunk=chunk,
                    score=norm_score,
                    rank=rank,
                )
            )

        return results

    def save(self, directory: Path | str) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_file = target_dir / "text_index.faiss"
        faiss.write_index(self.index, str(faiss_file))

        metadata_file = target_dir / "text_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self.chunks], f, indent=2)

    def load(self, directory: Path | str) -> bool:
        """Load FAISS index and metadata from disk if available."""
        target_dir = Path(directory)
        faiss_file = target_dir / "text_index.faiss"
        metadata_file = target_dir / "text_metadata.json"

        if not faiss_file.exists() or not metadata_file.exists():
            return False

        self.index = faiss.read_index(str(faiss_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = [TextChunk(**item) for item in data]
        return True
