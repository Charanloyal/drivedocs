"""Image embedding and FAISS vector index module using Hugging Face CLIP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import faiss

from drivedocs.config import settings
from drivedocs.schemas import ImageItem, ImageSearchResult


class CLIPEmbeddingModel:
    """Wrapper around Hugging Face CLIP model for multimodal image & text encoding."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: int = 512,
        mock_mode: bool = False,
    ) -> None:
        self.model_name = model_name or settings.image_model_name
        self.device = device or settings.device
        self.dimension = dimension
        self.mock_mode = mock_mode
        self._model = None
        self._processor = None

    def _load_model(self):
        if self.mock_mode:
            return
        if self._model is None or self._processor is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def encode_images(self, images: List[Union[Image.Image, str, Path]]) -> np.ndarray:
        """Encode a batch of PIL Images or image filepaths into normalized 512-dim vectors."""
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Open PIL images if filepaths passed
        pil_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        if self.mock_mode:
            # Deterministic pseudo-embeddings for fast isolated unit tests
            embeddings = []
            for img in pil_images:
                # Seed with image size and sample pixels
                w, h = img.size
                rng = np.random.RandomState((w * 31 + h * 17) % (2**31))
                vec = rng.randn(self.dimension).astype(np.float32)
                vec /= np.linalg.norm(vec)
                embeddings.append(vec)
            return np.vstack(embeddings).astype(np.float32)

        self._load_model()
        import torch

        inputs = self._processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            elif not isinstance(image_features, torch.Tensor):
                image_features = image_features[0]
            # L2 normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            vectors = image_features.cpu().numpy().astype(np.float32)

        return vectors

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text prompts with CLIP text encoder into the joint multimodal embedding space."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.mock_mode:
            embeddings = []
            for t in texts:
                import hashlib
                h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
                rng = np.random.RandomState(h % (2**31))
                vec = rng.randn(self.dimension).astype(np.float32)
                vec /= np.linalg.norm(vec)
                embeddings.append(vec)
            return np.vstack(embeddings).astype(np.float32)

        self._load_model()
        import torch

        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif not isinstance(text_features, torch.Tensor):
                text_features = text_features[0]
            # L2 normalize
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            vectors = text_features.cpu().numpy().astype(np.float32)

        return vectors


class ImageFAISSIndex:
    """FAISS-based vector index for multimodal image retrieval."""

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_model: Optional[CLIPEmbeddingModel] = None,
    ) -> None:
        self.dimension = dimension or settings.image_dimension
        self.embedding_model = embedding_model or CLIPEmbeddingModel(dimension=self.dimension)
        # Using IndexFlatIP for Inner Product (Cosine similarity on unit vectors)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.images: List[ImageItem] = []

    def count(self) -> int:
        """Total number of indexed images."""
        return len(self.images)

    def add_images(self, images: List[ImageItem]) -> int:
        """Embed and append new images into the FAISS index."""
        if not images:
            return 0

        paths = [item.image_path for item in images]
        vectors = self.embedding_model.encode_images(paths)

        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.images.extend(images)
        return len(images)

    def search_by_text(self, text_query: str, top_k: int = 2) -> List[ImageSearchResult]:
        """Retrieve images matching a text description using CLIP text encoder."""
        if not text_query.strip() or self.count() == 0:
            return []

        k = min(top_k, self.count())
        q_vec = self.embedding_model.encode_text([text_query])
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, k)
        return self._format_results(scores[0], indices[0])

    def search_by_image(
        self,
        query_image: Union[Image.Image, str, Path],
        top_k: int = 2,
    ) -> List[ImageSearchResult]:
        """Retrieve images visually similar to an uploaded query image."""
        if self.count() == 0:
            return []

        k = min(top_k, self.count())
        q_vec = self.embedding_model.encode_images([query_image])
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, k)
        return self._format_results(scores[0], indices[0])

    def _format_results(self, scores: np.ndarray, indices: np.ndarray) -> List[ImageSearchResult]:
        results: List[ImageSearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(self.images):
                continue
            item = self.images[idx]
            norm_score = float(np.clip(score, 0.0, 1.0))
            results.append(
                ImageSearchResult(
                    image=item,
                    score=norm_score,
                    rank=rank,
                )
            )
        return results

    def save(self, directory: Path | str) -> None:
        """Persist FAISS index and image metadata to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_file = target_dir / "image_index.faiss"
        faiss.write_index(self.index, str(faiss_file))

        metadata_file = target_dir / "image_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump([img.model_dump() for img in self.images], f, indent=2)

    def load(self, directory: Path | str) -> bool:
        """Load FAISS index and metadata from disk if available."""
        target_dir = Path(directory)
        faiss_file = target_dir / "image_index.faiss"
        metadata_file = target_dir / "image_metadata.json"

        if not faiss_file.exists() or not metadata_file.exists():
            return False

        self.index = faiss.read_index(str(faiss_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.images = [ImageItem(**item) for item in data]
        return True
