"""Image embedding (CLIP) and FAISS image index.

CLIP maps images and text into one shared 512-d space, which enables text→image and image→image search. This space
is unrelated to the MiniLM text space, so CLIP vectors live in their own index and their scores are never compared
numerically with MiniLM scores.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
from PIL import Image

from drivedocs.config import settings
from drivedocs.models import mock_image_embedding, mock_text_embedding, registry
from drivedocs.schemas import ImageItem, ImageSearchResult
from drivedocs.storage import atomic_write_json
from drivedocs.text_index import resolve_backend, write_faiss_atomic

ImageInput = Union[Image.Image, str, Path]


def _as_tensor(features):
    """transformers versions differ in whether get_*_features returns a tensor or a model output."""
    import torch

    if isinstance(features, torch.Tensor):
        return features
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(features, attr, None)
        if value is not None:
            return value
    return features[0]


class CLIPEmbeddingModel:
    """CLIP text + image towers; shared process-wide via the model registry. Backends: onnx (default),
    transformers (PyTorch) or mock (tests)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: int = 512,
        mock_mode: Optional[bool] = None,
        batch_size: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.image_model_name
        self.device = device or settings.device
        self.dimension = dimension
        self.backend = resolve_backend(backend, mock_mode)
        self.mock_mode = self.backend == "mock"
        self.batch_size = batch_size or settings.image_batch_size
        self.onnx_repo = settings.image_onnx_repo
        self.onnx_text_file = settings.image_onnx_text_file
        self.onnx_vision_file = settings.image_onnx_vision_file

    @property
    def registry_key(self) -> str:
        if self.backend == "onnx":
            return f"{self.onnx_repo}/{self.onnx_text_file}+{self.onnx_vision_file}|onnx|clip"
        return f"{self.model_name}|{self.device}|clip"

    @property
    def identity(self) -> str:
        if self.mock_mode:
            return f"mock-clip-{self.dimension}"
        if self.backend == "onnx":
            return f"{self.onnx_repo}/{self.onnx_text_file}+{self.onnx_vision_file}"
        return self.model_name

    @property
    def is_loaded(self) -> bool:
        return self.mock_mode or registry.is_loaded(self.registry_key)

    @property
    def _model(self):
        state = registry.state(self.registry_key)
        return state.objects[0] if state and state.loaded else None

    def _load(self):
        def loader():
            if self.backend == "onnx":
                from drivedocs.onnx_models import OnnxClipEncoder

                return (
                    OnnxClipEncoder(
                        self.onnx_repo, self.onnx_text_file, self.onnx_vision_file,
                        idle_unload_seconds=settings.vision_idle_unload_seconds,
                    ),
                    None,
                )
            from transformers import CLIPModel, CLIPProcessor

            processor = CLIPProcessor.from_pretrained(self.model_name)
            model = CLIPModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            return (model, processor)

        return registry.get(self.registry_key, loader)

    def warm_up(self) -> None:
        if not self.mock_mode:
            self._load()

    @property
    def vision_state(self) -> str:
        """'loaded', 'idle' (ONNX vision tower released until the next image is embedded) or 'not loaded'."""
        if self.mock_mode:
            return "loaded"
        model = self._model
        if model is None:
            return "not loaded"
        loaded = getattr(model, "vision_loaded", True)
        return "loaded" if loaded else "idle"

    def unload_vision(self) -> None:
        """Release the ONNX vision tower now (it reloads on the next image); no-op for other backends."""
        model = None if self.mock_mode else self._model
        if model is not None and hasattr(model, "unload_vision"):
            model.unload_vision()

    @staticmethod
    def _open(images: List[ImageInput]) -> List[Image.Image]:
        pil_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, (str, Path)):
                with Image.open(img) as opened:
                    pil_images.append(opened.convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))
        return pil_images

    def encode_images(self, images: List[ImageInput]) -> np.ndarray:
        """Encode images into L2-normalized 512-d vectors, in batches of `batch_size`."""
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        outputs: List[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = self._open(images[start : start + self.batch_size])
            if self.mock_mode:
                outputs.append(mock_image_embedding(batch, self.dimension))
                continue
            if self.backend == "onnx":
                encoder, _ = self._load()
                with registry.inference_slot():
                    outputs.append(encoder.encode_images(batch))
                continue
            import torch

            model, processor = self._load()
            with registry.inference_slot(), torch.no_grad():
                inputs = processor(images=batch, return_tensors="pt").to(self.device)
                feats = _as_tensor(model.get_image_features(**inputs))
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                outputs.append(feats.cpu().numpy().astype(np.float32))
        return np.ascontiguousarray(np.vstack(outputs), dtype=np.float32)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text prompts with the CLIP text tower (same space as CLIP image vectors)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.mock_mode:
            return mock_text_embedding(texts, self.dimension)
        if self.backend == "onnx":
            encoder, _ = self._load()
            with registry.inference_slot():
                return np.ascontiguousarray(encoder.encode_text(texts), dtype=np.float32)
        import torch

        model, processor = self._load()
        with registry.inference_slot(), torch.no_grad():
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = _as_tensor(model.get_text_features(**inputs))
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return np.ascontiguousarray(feats.cpu().numpy(), dtype=np.float32)


class ImageFAISSIndex:
    """FAISS IndexFlatIP over L2-normalized CLIP image vectors."""

    INDEX_FILE = "image_index.faiss"
    METADATA_FILE = "image_metadata.json"

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_model: Optional[CLIPEmbeddingModel] = None,
        images_dir: Optional[Path] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.dimension = dimension or settings.image_dimension
        self.embedding_model = embedding_model or CLIPEmbeddingModel(dimension=self.dimension)
        self.images_dir = Path(images_dir or settings.images_dir)
        # When disabled, images are still registered (viewable) but not embedded or searchable.
        self.enabled = settings.enable_image_search if enabled is None else enabled
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        self.images: List[ImageItem] = []
        self.vectors_valid = True
        self._lock = threading.RLock()

    def count(self) -> int:
        return len(self.images)

    def resolve_path(self, item: ImageItem) -> Path:
        p = Path(item.image_path)
        if not p.is_absolute():
            return self.images_dir / p
        if p.exists():
            return p
        return self.images_dir / p.name

    def encode_items(self, images: List[ImageItem]) -> np.ndarray:
        if not self.enabled:
            return np.zeros((len(images), self.dimension), dtype=np.float32)
        vectors = self.embedding_model.encode_images([self.resolve_path(i) for i in images])
        faiss.normalize_L2(vectors)
        return vectors

    def add_embedded(self, images: List[ImageItem], vectors: np.ndarray) -> int:
        if not images:
            return 0
        if vectors.shape != (len(images), self.dimension):
            raise ValueError("vector/image count mismatch")
        with self._lock:
            self.index.add(vectors)
            self.images.extend(images)
        return len(images)

    def add_images(self, images: List[ImageItem]) -> int:
        if not images:
            return 0
        return self.add_embedded(images, self.encode_items(images))

    def remove_doc(self, doc_id: str) -> List[ImageItem]:
        with self._lock:
            keep = [i for i, img in enumerate(self.images) if img.doc_id != doc_id]
            removed = [img for img in self.images if img.doc_id == doc_id]
            if not removed:
                return []
            vectors = self.index.reconstruct_n(0, self.index.ntotal) if self.index.ntotal else None
            new_index = faiss.IndexFlatIP(self.dimension)
            if keep and vectors is not None:
                new_index.add(np.ascontiguousarray(vectors[keep]))
            self.index = new_index
            self.images = [self.images[i] for i in keep]
            return removed

    def reembed_all(self) -> None:
        with self._lock:
            items = list(self.images)
            new_index = faiss.IndexFlatIP(self.dimension)
            if items:
                new_index.add(self.encode_items(items))
            self.index = new_index
            self.vectors_valid = self.enabled

    def search_by_text(self, text_query: str, top_k: int = 2) -> List[ImageSearchResult]:
        if not self.enabled or not text_query or not text_query.strip() or self.count() == 0:
            return []
        q_vec = self.embedding_model.encode_text([text_query])
        faiss.normalize_L2(q_vec)
        return self._search(q_vec, top_k)

    def search_by_image(self, query_image: ImageInput, top_k: int = 2) -> List[ImageSearchResult]:
        if not self.enabled or self.count() == 0:
            return []
        q_vec = self.embedding_model.encode_images([query_image])
        faiss.normalize_L2(q_vec)
        return self._search(q_vec, top_k)

    def _search(self, q_vec: np.ndarray, top_k: int) -> List[ImageSearchResult]:
        with self._lock:
            if not self.vectors_valid:
                return []
            k = min(max(1, top_k), self.count())
            scores, indices = self.index.search(q_vec, k)
            items = self.images
            results: List[ImageSearchResult] = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx < 0 or idx >= len(items):
                    continue
                results.append(ImageSearchResult(image=items[idx], score=float(np.clip(score, 0.0, 1.0)), rank=rank))
        return results

    def _portable(self, item: ImageItem) -> dict:
        data = item.model_dump()
        path = Path(item.image_path)
        if path.is_absolute():
            try:
                data["image_path"] = str(path.resolve().relative_to(self.images_dir.resolve())).replace("\\", "/")
            except ValueError:
                data["image_path"] = str(path)
        return data

    def save(self, directory: Path | str) -> None:
        target_dir = Path(directory)
        with self._lock:
            write_faiss_atomic(self.index, target_dir / self.INDEX_FILE)
            atomic_write_json(target_dir / self.METADATA_FILE, [self._portable(img) for img in self.images])

    def load(self, directory: Path | str) -> bool:
        target_dir = Path(directory)
        faiss_file = target_dir / self.INDEX_FILE
        metadata_file = target_dir / self.METADATA_FILE
        if not faiss_file.exists() or not metadata_file.exists():
            return False
        index = faiss.read_index(str(faiss_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            items = [ImageItem(**item) for item in json.load(f)]
        if index.d != self.dimension or index.ntotal != len(items):
            return False
        with self._lock:
            self.index = index
            self.images = items
        return True
