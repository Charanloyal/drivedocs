"""Configuration loader and environment manager for DriveDocs.

Values come from ``configs/config.yaml`` and can be overridden with environment variables. All storage locations
are derived from a single storage root (``DRIVEDOCS_STORAGE_DIR``) so a persistent disk can be mounted in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


class Settings:
    """Application settings loaded from YAML config with environment variable overrides."""

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self.project_root: Path = self._resolve_project_root()

        if config_path is None:
            env_config = _env_str("DRIVEDOCS_CONFIG", None)
            default_path = self.project_root / "configs" / "config.yaml"
            if env_config:
                config_path = Path(env_config)
            elif default_path.exists():
                config_path = default_path
            else:
                config_path = Path("configs/config.yaml")

        self.config_path: Path = Path(config_path)
        self.raw_config: Dict[str, Any] = self._load_yaml()

        # Application
        app_cfg = self.raw_config.get("app", {})
        self.app_name: str = app_cfg.get("name", "DriveDocs")
        self.app_version: str = str(app_cfg.get("version", "0.2.0"))
        self.host: str = _env_str("DRIVEDOCS_HOST", app_cfg.get("host", "0.0.0.0"))
        self.port: int = int(os.getenv("PORT", os.getenv("DRIVEDOCS_PORT", app_cfg.get("port", 8000))))
        self.debug: bool = _env_bool("DRIVEDOCS_DEBUG", bool(app_cfg.get("debug", False)))
        self.log_level: str = (_env_str("DRIVEDOCS_LOG_LEVEL", "INFO") or "INFO").upper()
        cors_raw = _env_str("DRIVEDOCS_CORS_ORIGINS", "") or ""
        self.cors_origins: List[str] = [o.strip() for o in cors_raw.split(",") if o.strip()]

        # Hardware: CPU unless CUDA is explicitly requested *and* available.
        hw_cfg = self.raw_config.get("hardware", {})
        requested_device = (_env_str("DEVICE", hw_cfg.get("device", "cpu")) or "cpu").lower()
        self.requested_device: str = requested_device
        self.device: str = self._resolve_device(requested_device)

        # Storage: every path is derived from the storage root.
        storage_cfg = self.raw_config.get("storage", {})
        storage_base = _env_str("DRIVEDOCS_STORAGE_DIR", storage_cfg.get("base_dir", "data/storage"))
        self.storage_dir: Path = self._resolve_path(storage_base)
        self.images_dir: Path = self.storage_dir / "images"
        self.indexes_dir: Path = self.storage_dir / "indexes"
        self.uploads_dir: Path = self.storage_dir / "uploads"
        self.registry_path: Path = self.storage_dir / "documents.json"
        self.persistent_storage: bool = _env_bool(
            "DRIVEDOCS_PERSISTENT_STORAGE", bool(storage_cfg.get("persistent", False))
        )
        self.keep_original_uploads: bool = _env_bool(
            "DRIVEDOCS_KEEP_UPLOADS", bool(storage_cfg.get("keep_original_uploads", True))
        )

        max_mb_env = _env_str("DRIVEDOCS_MAX_UPLOAD_MB", None)
        if max_mb_env is not None:
            self.max_upload_size_bytes: int = int(float(max_mb_env) * 1024 * 1024)
        else:
            self.max_upload_size_bytes = int(storage_cfg.get("max_upload_size_bytes", 15 * 1024 * 1024))
        self.allowed_extensions: List[str] = [
            e.lower() for e in storage_cfg.get("allowed_extensions", [".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"])
        ]
        self.max_pdf_pages: int = int(storage_cfg.get("max_pdf_pages", 300))
        self.max_images_per_document: int = int(storage_cfg.get("max_images_per_document", 50))
        self.max_image_pixels: int = int(storage_cfg.get("max_image_pixels", 40_000_000))
        self.max_stored_image_side: int = int(storage_cfg.get("max_stored_image_side", 1600))
        self.min_extracted_image_side: int = int(storage_cfg.get("min_extracted_image_side", 40))
        self.max_documents: int = int(_env_str("DRIVEDOCS_MAX_DOCUMENTS", str(storage_cfg.get("max_documents", 500))))

        # Chunking
        chunk_cfg = self.raw_config.get("chunking", {})
        self.chunk_size: int = int(_env_str("DRIVEDOCS_CHUNK_SIZE", str(chunk_cfg.get("chunk_size", 500))))
        self.chunk_overlap: int = int(_env_str("DRIVEDOCS_CHUNK_OVERLAP", str(chunk_cfg.get("chunk_overlap", 100))))
        self.separators: List[str] = chunk_cfg.get("separators", ["\n\n", "\n", ". ", " ", ""])

        # Embeddings: "onnx" (default; ONNX Runtime, no PyTorch, lowest memory), "transformers" (PyTorch, full
        # precision; needs the optional torch extra) or "mock" (deterministic hashing vectors for tests only).
        emb_cfg = self.raw_config.get("embeddings", {})
        self.embedding_backend: str = (
            _env_str("DRIVEDOCS_EMBEDDING_BACKEND", emb_cfg.get("backend", "onnx")) or "onnx"
        ).lower()
        text_emb = emb_cfg.get("text", {})
        self.text_model_name: str = _env_str(
            "DRIVEDOCS_TEXT_MODEL", text_emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.text_onnx_repo: str = _env_str("DRIVEDOCS_TEXT_ONNX_REPO", text_emb.get("onnx_repo", "Xenova/all-MiniLM-L6-v2"))
        self.text_onnx_file: str = _env_str(
            "DRIVEDOCS_TEXT_ONNX_FILE", text_emb.get("onnx_file", "onnx/model_quantized.onnx")
        )
        self.text_max_length: int = int(text_emb.get("max_length", 256))
        self.text_dimension: int = int(text_emb.get("dimension", 384))
        self.text_batch_size: int = int(text_emb.get("batch_size", 32))
        # "native" (plain transformers, lower memory) or "sentence-transformers" (the full library)
        self.text_encoder: str = (_env_str("DRIVEDOCS_TEXT_ENCODER", text_emb.get("encoder", "native")) or "native").lower()
        self.normalize_embeddings: bool = bool(text_emb.get("normalize", True))

        image_emb = emb_cfg.get("image", {})
        self.image_model_name: str = _env_str(
            "DRIVEDOCS_IMAGE_MODEL", image_emb.get("model_name", "openai/clip-vit-base-patch32")
        )
        self.image_onnx_repo: str = _env_str(
            "DRIVEDOCS_IMAGE_ONNX_REPO", image_emb.get("onnx_repo", "Xenova/clip-vit-base-patch32")
        )
        self.image_onnx_text_file: str = _env_str(
            "DRIVEDOCS_IMAGE_ONNX_TEXT_FILE", image_emb.get("onnx_text_file", "onnx/text_model_q4.onnx")
        )
        self.image_onnx_vision_file: str = _env_str(
            "DRIVEDOCS_IMAGE_ONNX_VISION_FILE", image_emb.get("onnx_vision_file", "onnx/vision_model_q4.onnx")
        )
        self.image_dimension: int = int(image_emb.get("dimension", 512))
        self.image_batch_size: int = int(image_emb.get("batch_size", 16))
        self.enable_image_search: bool = _env_bool(
            "DRIVEDOCS_ENABLE_IMAGE_SEARCH", bool(image_emb.get("enabled", True))
        )

        # Runtime / performance
        runtime_cfg = self.raw_config.get("runtime", {})
        self.preload_models: bool = _env_bool("DRIVEDOCS_PRELOAD_MODELS", bool(runtime_cfg.get("preload_models", True)))
        self.seed_sample_data: bool = _env_bool(
            "DRIVEDOCS_SEED_SAMPLE_DATA", bool(runtime_cfg.get("seed_sample_data", True))
        )
        self.max_concurrent_inference: int = int(
            _env_str("DRIVEDOCS_MAX_CONCURRENT_INFERENCE", str(runtime_cfg.get("max_concurrent_inference", 2)))
        )
        # ONNX CLIP vision tower: released after this many idle seconds (0 = keep loaded). It is only needed to embed
        # new images; text-to-image search uses the always-loaded text tower.
        self.vision_idle_unload_seconds: float = float(
            _env_str("DRIVEDOCS_VISION_IDLE_UNLOAD_SECONDS", str(runtime_cfg.get("vision_idle_unload_seconds", 300)))
        )
        self.ready_wait_seconds: float = float(
            _env_str("DRIVEDOCS_READY_WAIT_SECONDS", str(runtime_cfg.get("ready_wait_seconds", 60)))
        )
        self.torch_threads: Optional[int] = (
            int(os.getenv("DRIVEDOCS_TORCH_THREADS")) if os.getenv("DRIVEDOCS_TORCH_THREADS") else None
        )
        self.sample_dir: Path = self.project_root / "data" / "sample"
        self.sample_files: List[str] = runtime_cfg.get(
            "sample_files",
            [
                "aurora_s5_owner_manual.pdf",
                "maintenance_schedule.md",
                "roadside_emergency_guide.txt",
                "service_record_ro58213.txt",
                "engine_bay_diagram.png",
            ],
        )
        self.reports_dir: Path = self.project_root / "reports"
        # Demo index built at deploy time (scripts/build_demo_index.py). It lives inside the project so it always ships
        # with the build output, whatever DRIVEDOCS_STORAGE_DIR points to; it is copied into an empty storage root.
        self.prebuilt_index_dir: Path = self._resolve_path(
            _env_str("DRIVEDOCS_PREBUILT_DIR", runtime_cfg.get("prebuilt_index_dir", "data/prebuilt"))
        )
        self.security_dir: Path = self.project_root / "data" / "security"

        # Security guardrails
        sec_cfg = self.raw_config.get("security", {})
        self.redact_pii: bool = _env_bool("DRIVEDOCS_REDACT_PII", bool(sec_cfg.get("redact_pii", True)))
        self.block_pii_requests: bool = _env_bool(
            "DRIVEDOCS_BLOCK_PII_REQUESTS", bool(sec_cfg.get("block_pii_requests", True))
        )
        self.detect_prompt_injection: bool = _env_bool(
            "DRIVEDOCS_DETECT_PROMPT_INJECTION", bool(sec_cfg.get("detect_prompt_injection", True))
        )

        # LangChain agent
        agent_cfg = self.raw_config.get("agent", {})
        self.agent_enabled: bool = _env_bool("DRIVEDOCS_AGENT_ENABLED", bool(agent_cfg.get("enabled", True)))
        # "auto": an LLM tool-calling planner when a provider key and its LangChain integration are available,
        # otherwise the deterministic offline router. "router" forces the offline router.
        self.agent_planner: str = (_env_str("DRIVEDOCS_AGENT_PLANNER", agent_cfg.get("planner", "auto")) or "auto").lower()
        self.agent_max_steps: int = int(agent_cfg.get("max_steps", 4))
        self.agent_max_subquestions: int = int(agent_cfg.get("max_subquestions", 3))

        # Retrieval
        ret_cfg = self.raw_config.get("retrieval", {})
        self.default_top_k_text: int = int(ret_cfg.get("default_top_k_text", 4))
        self.default_top_k_images: int = int(ret_cfg.get("default_top_k_images", 2))
        # Kept for backwards compatibility: used as the text threshold when text_min_score is absent.
        self.min_similarity_threshold: float = float(ret_cfg.get("min_similarity_threshold", 0.30))
        self.text_min_score: float = float(ret_cfg.get("text_min_score", self.min_similarity_threshold))
        self.text_context_min_score: float = float(ret_cfg.get("text_context_min_score", 0.20))
        self.image_min_score: float = float(ret_cfg.get("image_min_score", 0.30))
        self.image_min_score_visual_query: float = float(ret_cfg.get("image_min_score_visual_query", 0.25))
        self.image_min_score_colocated: float = float(ret_cfg.get("image_min_score_colocated", 0.25))
        self.text_weight: float = float(ret_cfg.get("text_weight", 0.7))
        self.image_weight: float = float(ret_cfg.get("image_weight", 0.3))
        self.rrf_k: int = int(ret_cfg.get("rrf_k", 60))

        # Generation
        gen_cfg = self.raw_config.get("generation", {})
        self.llm_provider: str = (_env_str("LLM_PROVIDER", gen_cfg.get("provider", "extractive")) or "extractive").lower()
        self.gemini_model: str = _env_str("GEMINI_MODEL", gen_cfg.get("gemini_model", "gemini-1.5-flash"))
        self.openai_model: str = _env_str("OPENAI_MODEL", gen_cfg.get("openai_model", "gpt-4o-mini"))
        self.gemini_api_key: Optional[str] = _env_str("GEMINI_API_KEY", None)
        self.openai_api_key: Optional[str] = _env_str("OPENAI_API_KEY", None)
        self.temperature: float = float(gen_cfg.get("temperature", 0.2))
        self.max_tokens: int = int(gen_cfg.get("max_tokens", 512))
        self.llm_timeout_seconds: float = float(gen_cfg.get("timeout_seconds", 30))
        self.max_answer_sentences: int = int(gen_cfg.get("max_answer_sentences", 3))

    def ensure_storage_dirs(self) -> None:
        """Create storage directories. Called by the storage layer, not at import time."""
        for d in (self.storage_dir, self.images_dir, self.indexes_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _load_yaml(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _resolve_project_root() -> Path:
        """DRIVEDOCS_HOME, else the source checkout containing configs/, else the current directory."""
        env_home = _env_str("DRIVEDOCS_HOME", None)
        if env_home:
            return Path(env_home).resolve()
        checkout = Path(__file__).resolve().parent.parent.parent
        if (checkout / "configs").is_dir():
            return checkout
        return Path.cwd()

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Use CUDA only when explicitly requested and actually available; never auto-select a GPU."""
        if requested == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"


settings = Settings()
