"""Configuration loader and environment manager for DriveDocs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

# Load any .env present in working directory or project root
load_dotenv()


class Settings:
    """Application settings loaded from YAML config with environment variable overrides."""

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self.project_root: Path = Path(__file__).resolve().parent.parent.parent

        if config_path is None:
            default_path = self.project_root / "configs" / "config.yaml"
            if default_path.exists():
                config_path = default_path
            else:
                config_path = Path("configs/config.yaml")

        self.config_path: Path = Path(config_path)
        self.raw_config: Dict[str, Any] = self._load_yaml()

        # Application
        app_cfg = self.raw_config.get("app", {})
        self.app_name: str = app_cfg.get("name", "DriveDocs")
        self.app_version: str = app_cfg.get("version", "0.1.0")
        self.host: str = os.getenv("DRIVEDOCS_HOST", app_cfg.get("host", "0.0.0.0"))
        self.port: int = int(os.getenv("DRIVEDOCS_PORT", app_cfg.get("port", 8000)))
        self.debug: bool = os.getenv("DRIVEDOCS_DEBUG", "false").lower() == "true"

        # Hardware
        hw_cfg = self.raw_config.get("hardware", {})
        requested_device = os.getenv("DEVICE", hw_cfg.get("device", "cpu")).lower()
        self.device: str = self._resolve_device(requested_device)

        # Storage
        storage_cfg = self.raw_config.get("storage", {})
        storage_base = os.getenv("DRIVEDOCS_STORAGE_DIR", storage_cfg.get("base_dir", "data/storage"))
        self.storage_dir: Path = self._resolve_path(storage_base)
        self.images_dir: Path = self._resolve_path(storage_cfg.get("images_dir", "data/storage/images"))
        self.indexes_dir: Path = self._resolve_path(storage_cfg.get("indexes_dir", "data/storage/indexes"))
        self.max_upload_size_bytes: int = int(storage_cfg.get("max_upload_size_bytes", 15 * 1024 * 1024))
        self.allowed_extensions: List[str] = storage_cfg.get(
            "allowed_extensions", [".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"]
        )

        # Ensure storage directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        # Chunking
        chunk_cfg = self.raw_config.get("chunking", {})
        self.chunk_size: int = int(chunk_cfg.get("chunk_size", 500))
        self.chunk_overlap: int = int(chunk_cfg.get("chunk_overlap", 100))
        self.separators: List[str] = chunk_cfg.get(
            "separators", ["\n\n", "\n", ". ", " ", ""]
        )

        # Embeddings
        emb_cfg = self.raw_config.get("embeddings", {})
        text_emb = emb_cfg.get("text", {})
        self.text_model_name: str = text_emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.text_dimension: int = int(text_emb.get("dimension", 384))
        self.text_batch_size: int = int(text_emb.get("batch_size", 32))
        self.normalize_embeddings: bool = bool(text_emb.get("normalize", True))

        image_emb = emb_cfg.get("image", {})
        self.image_model_name: str = image_emb.get("model_name", "openai/clip-vit-base-patch32")
        self.image_dimension: int = int(image_emb.get("dimension", 512))
        self.image_batch_size: int = int(image_emb.get("batch_size", 16))

        # Retrieval
        ret_cfg = self.raw_config.get("retrieval", {})
        self.default_top_k_text: int = int(ret_cfg.get("default_top_k_text", 4))
        self.default_top_k_images: int = int(ret_cfg.get("default_top_k_images", 2))
        self.min_similarity_threshold: float = float(ret_cfg.get("min_similarity_threshold", 0.25))
        self.text_weight: float = float(ret_cfg.get("text_weight", 0.7))
        self.image_weight: float = float(ret_cfg.get("image_weight", 0.3))

        # Generation
        gen_cfg = self.raw_config.get("generation", {})
        self.llm_provider: str = os.getenv("LLM_PROVIDER", gen_cfg.get("provider", "extractive")).lower()
        self.gemini_model: str = gen_cfg.get("gemini_model", "gemini-1.5-flash")
        self.openai_model: str = gen_cfg.get("openai_model", "gpt-4o-mini")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.temperature: float = float(gen_cfg.get("temperature", 0.2))
        self.max_tokens: int = int(gen_cfg.get("max_tokens", 512))

    def _load_yaml(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _resolve_device(self, requested: str) -> str:
        if requested == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return "cpu"


# Singleton instance
settings = Settings()
