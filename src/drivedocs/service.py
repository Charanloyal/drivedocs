"""DriveDocs application service: the business logic behind the HTTP routes.

Responsibilities
* start-up: load registry + indexes, reconcile them, re-embed when the configured model changed, seed demo data,
  warm models (in a background thread so the HTTP port binds immediately);
* ingestion: parse → embed → add to indexes → register → persist, all under one write lock, so a failure leaves
  the previous state intact;
* deletion, listing, search, chat, health and readiness reporting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from drivedocs.agent import DriveDocsAgent
from drivedocs.citations import CitationVerifier
from drivedocs.config import Settings, settings as default_settings
from drivedocs.generation import GroundedMultimodalGenerator
from drivedocs.image_index import CLIPEmbeddingModel, ImageFAISSIndex
from drivedocs.ingestion import (
    DocumentIngestionManager,
    DuplicateDocumentError,
    IngestionError,
    MalformedDocumentError,
    ParsedDocument,
)
from drivedocs.models import ModelLoadError, registry
from drivedocs.retrieval import MultimodalRetriever
from drivedocs.schemas import (
    ChatRequest,
    ChatResponse,
    ComponentStatus,
    DeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadata,
    HealthResponse,
    PublicImage,
    PublicImageSearchResult,
    ReadinessResponse,
    SystemInfoResponse,
    TextSearchResult,
    UploadResponse,
)
from drivedocs.storage import LocalStorage, atomic_write_json
from drivedocs.text_index import TextEmbeddingModel, TextFAISSIndex

logger = logging.getLogger("drivedocs.service")

MANIFEST_FILE = "index_manifest.json"


class ServiceUnavailableError(RuntimeError):
    """Raised when the service is still starting or failed to start."""


class NotFoundError(LookupError):
    pass


class InvalidQueryImageError(ValueError):
    pass


def _rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024, 1)
    except Exception:
        return None


def _release_memory() -> None:
    """Return memory freed after start-up (PDF parsing, image batches) to the OS; glibc keeps it otherwise."""
    import gc

    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # pragma: no cover - non-glibc Linux (e.g. musl)
            pass


class DriveDocsService:
    def __init__(self, config: Optional[Settings] = None, storage: Optional[LocalStorage] = None) -> None:
        self.config = config or default_settings
        self.storage = storage or LocalStorage(config=self.config)
        self.text_model = TextEmbeddingModel(dimension=self.config.text_dimension)
        self.image_model = CLIPEmbeddingModel(dimension=self.config.image_dimension)
        self.text_index = TextFAISSIndex(dimension=self.config.text_dimension, embedding_model=self.text_model)
        self.image_index = ImageFAISSIndex(
            dimension=self.config.image_dimension,
            embedding_model=self.image_model,
            images_dir=self.storage.images_dir,
            enabled=self.config.enable_image_search,
        )
        self.retriever = MultimodalRetriever(text_index=self.text_index, image_index=self.image_index)
        self.ingestion = DocumentIngestionManager(
            storage_dir=self.storage.root,
            images_dir=self.storage.images_dir,
        )
        self.verifier = CitationVerifier(lookup_document=self.ingestion.find_by_filename)
        self.generator = GroundedMultimodalGenerator(verifier=self.verifier)
        self.agent = DriveDocsAgent(self.retriever, self.generator, config=self.config)

        self._write_lock = threading.RLock()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.phase = "created"
        self.startup_error: Optional[str] = None
        self.consistency_problems: List[str] = []
        self.timings_ms: Dict[str, float] = {}
        self.registry_loaded = False

    # Lifecycle -------------------------------------------------------------------------------------------------

    def start(self, background: bool = True) -> None:
        if background:
            self._thread = threading.Thread(target=self.initialize, name="drivedocs-init", daemon=True)
            self._thread.start()
        else:
            self.initialize()

    def initialize(self) -> None:
        started = time.perf_counter()
        try:
            self.phase = "loading_indexes"
            t0 = time.perf_counter()
            self.storage.ensure_dirs()
            self._load_state()
            self.timings_ms["load_state"] = round((time.perf_counter() - t0) * 1000, 1)

            if self.config.preload_models:
                self.phase = "loading_models"
                t0 = time.perf_counter()
                self.text_model.warm_up()
                self.timings_ms["text_model_load"] = round((time.perf_counter() - t0) * 1000, 1)
                if self.config.enable_image_search:
                    t0 = time.perf_counter()
                    try:
                        self.image_model.warm_up()
                    except ModelLoadError as exc:
                        # Image search is optional: degrade instead of failing the whole service.
                        logger.error("Image model unavailable, disabling image search: %s", exc)
                        self.image_index.enabled = False
                    self.timings_ms["image_model_load"] = round((time.perf_counter() - t0) * 1000, 1)

            self._reembed_if_model_changed()

            if self.config.seed_sample_data and not any(
                d.status == "ready" for d in self.ingestion.documents_by_id.values()
            ):
                self.phase = "seeding"
                t0 = time.perf_counter()
                self.seed_samples()
                self.timings_ms["seed_samples"] = round((time.perf_counter() - t0) * 1000, 1)

            self.timings_ms["startup_total"] = round((time.perf_counter() - started) * 1000, 1)
            _release_memory()
            self.phase = "ready"
            logger.info(
                "DriveDocs ready in %.0f ms: %d documents, %d chunks, %d images, storage=%s",
                self.timings_ms["startup_total"],
                len(self.ingestion.documents_by_id),
                self.text_index.count(),
                self.image_index.count(),
                self.storage.mode,
            )
        except Exception as exc:  # pragma: no cover - surfaced through /ready
            self.phase = "failed"
            self.startup_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.exception("DriveDocs failed to start")
        finally:
            self._ready.set()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        self._ready.wait(self.config.ready_wait_seconds if timeout is None else timeout)
        return self.phase == "ready"

    def require_ready(self) -> None:
        if not self.wait_ready():
            if self.phase == "failed":
                raise ServiceUnavailableError("The service failed to start. Check the server logs.")
            raise ServiceUnavailableError("The service is still starting (loading models/indexes). Retry shortly.")

    def _load_state(self) -> None:
        payload = self.storage.load_registry()
        loaded_text = self.text_index.load(self.storage.indexes_dir)
        loaded_image = self.image_index.load(self.storage.indexes_dir)
        if payload is None:
            if loaded_text or loaded_image:
                self.consistency_problems.append("Indexes exist without a document registry; they were discarded.")
                logger.warning("Discarding legacy indexes that have no document registry.")
                self._reset_indexes()
            self.registry_loaded = True
            return
        problems = self.ingestion.load_registry_payload(payload, self.text_index.chunks, self.image_index.images)
        self.registry_loaded = True
        if problems:
            self.consistency_problems.extend(problems)
            logger.warning("Registry/index inconsistency detected (%d problems); rebuilding.", len(problems))
            self.rebuild_from_originals()

    def _reset_indexes(self) -> None:
        self.text_index.index.reset()
        self.text_index.chunks = []
        self.image_index.index.reset()
        self.image_index.images = []

    def rebuild_from_originals(self) -> None:
        """Rebuild indexes from private copies of the original uploads (and bundled samples)."""
        with self._write_lock:
            docs = [d for d in self.ingestion.documents_by_id.values() if d.status == "ready"]
            self._reset_indexes()
            fresh = DocumentIngestionManager(storage_dir=self.storage.root, images_dir=self.storage.images_dir)
            for meta in docs:
                original = self._original_bytes(meta)
                if original is None:
                    logger.warning("Cannot rebuild %s: original file not available; dropping it.", meta.doc_id)
                    continue
                try:
                    parsed = fresh.parse_bytes(meta.filename, original)
                    self.retriever.index_parsed_document(parsed)
                    fresh.register(parsed, source=meta.source)
                    parsed.metadata.created_at = meta.created_at
                except Exception as exc:
                    logger.warning("Rebuild of %s failed: %s", meta.doc_id, type(exc).__name__)
            for meta in self.ingestion.documents_by_id.values():
                if meta.status == "failed":
                    fresh.documents_by_id[meta.doc_id] = meta
            self.ingestion.documents_by_id = fresh.documents_by_id
            self.ingestion.hash_to_doc_id = fresh.hash_to_doc_id
            self.ingestion.chunks_by_doc_id = fresh.chunks_by_doc_id
            self.ingestion.images_by_doc_id = fresh.images_by_doc_id
            self.persist()

    def _original_bytes(self, meta: DocumentMetadata) -> Optional[bytes]:
        candidate = self.storage.uploads_dir / f"{meta.doc_id}.{meta.file_type}"
        if candidate.exists():
            return candidate.read_bytes()
        sample = self.config.sample_dir / meta.filename
        if meta.source == "sample" and sample.exists():
            return sample.read_bytes()
        return None

    def _manifest(self) -> dict:
        return {
            "version": 1,
            "text_model": self.text_model.identity,
            "text_dimension": self.text_index.dimension,
            "image_model": self.image_model.identity if self.image_index.enabled else None,
            "image_dimension": self.image_index.dimension,
        }

    def _reembed_if_model_changed(self) -> None:
        path = self.storage.indexes_dir / MANIFEST_FILE
        if not path.exists():
            if self.text_index.count() or self.image_index.count():
                self.persist()
            return
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
        current = self._manifest()
        changed = False
        if old.get("text_model") != current["text_model"] and self.text_index.count():
            logger.info("Text model changed (%s -> %s); re-embedding chunks.", old.get("text_model"), current["text_model"])
            self.text_index.reembed_all()
            changed = True
        if old.get("image_model") != current["image_model"] and self.image_index.count():
            logger.info("Image model changed; re-embedding images.")
            self.image_index.reembed_all()
            changed = True
        if changed:
            self.persist()

    def persist(self) -> None:
        """Atomically write indexes, the manifest and then the registry."""
        with self._write_lock:
            self.retriever.save(self.storage.indexes_dir)
            atomic_write_json(self.storage.indexes_dir / MANIFEST_FILE, self._manifest())
            self.storage.save_registry(self.ingestion.registry_payload())

    def seed_samples(self) -> List[str]:
        seeded: List[str] = []
        for name in self.config.sample_files:
            path = self.config.sample_dir / name
            if not path.exists():
                logger.warning("Sample file missing: %s", name)
                continue
            try:
                result = self.ingest(name, path.read_bytes(), source="sample", keep_original=False)
                seeded.append(result.document.doc_id)
            except DuplicateDocumentError:
                continue
            except IngestionError as exc:
                logger.warning("Could not seed %s: %s", name, exc)
        return seeded

    # Ingestion -------------------------------------------------------------------------------------------------

    def ingest(
        self, filename: str, content: bytes, source: str = "upload", keep_original: Optional[bool] = None
    ) -> UploadResponse:
        keep = self.config.keep_original_uploads if keep_original is None else keep_original
        with self._write_lock:
            try:
                parsed = self.ingestion.parse_bytes(filename, content)
            except MalformedDocumentError as exc:
                self.ingestion.record_failure(filename, content, str(exc))
                self._persist_quietly()
                raise
            try:
                self._index_and_register(parsed, source)
                if keep and source == "upload":
                    self.storage.save_original(parsed.metadata.doc_id, parsed.metadata.file_type, content)
                self.persist()
            except Exception:
                self._rollback(parsed)
                raise
        logger.info(
            "Ingested %s (%s): %d chunks, %d images",
            parsed.metadata.doc_id,
            parsed.metadata.file_type,
            len(parsed.chunks),
            len(parsed.images),
        )
        return UploadResponse(
            message=f"Document '{parsed.metadata.filename}' ingested successfully.",
            document=parsed.metadata,
            chunks_indexed=len(parsed.chunks),
            images_indexed=len(parsed.images) if self.image_index.enabled else 0,
            warnings=parsed.metadata.warnings,
        )

    def _index_and_register(self, parsed: ParsedDocument, source: str) -> None:
        self.retriever.index_parsed_document(parsed)
        self.ingestion.register(parsed, source=source)

    def _rollback(self, parsed: ParsedDocument) -> None:
        doc_id = parsed.metadata.doc_id
        self.retriever.remove_document(doc_id)
        self.ingestion.remove_document(doc_id)
        self.ingestion.discard(parsed)
        self.storage.delete_original(doc_id)
        logger.warning("Rolled back ingestion of %s", doc_id)

    def _persist_quietly(self) -> None:
        try:
            self.persist()
        except Exception:
            logger.exception("Failed to persist registry")

    def delete_document(self, doc_id: str) -> DeleteResponse:
        with self._write_lock:
            meta = self.ingestion.get_document(doc_id)
            if meta is None:
                raise NotFoundError(f"Document '{doc_id}' was not found.")
            removed_chunks = self.text_index.remove_doc(doc_id)
            removed_images = self.image_index.remove_doc(doc_id)
            for item in self.ingestion.get_images(doc_id) or removed_images:
                self.storage.delete_image(item.image_path)
            self.ingestion.remove_document(doc_id)
            self.storage.delete_original(doc_id)
            self.persist()
        logger.info("Deleted %s", doc_id)
        return DeleteResponse(
            message=f"Document '{meta.filename}' was deleted.",
            doc_id=doc_id,
            chunks_removed=removed_chunks,
            images_removed=len(removed_images),
        )

    # Queries ---------------------------------------------------------------------------------------------------

    def list_documents(self) -> DocumentListResponse:
        docs = self.ingestion.list_documents()
        return DocumentListResponse(
            documents=docs,
            total_documents=sum(1 for d in docs if d.status == "ready"),
            total_chunks=self.text_index.count(),
            total_images=self.image_index.count(),
            failed_documents=sum(1 for d in docs if d.status == "failed"),
        )

    def document_details(self, doc_id: str) -> DocumentDetailResponse:
        meta = self.ingestion.get_document(doc_id)
        if meta is None:
            raise NotFoundError(f"Document '{doc_id}' was not found.")
        return DocumentDetailResponse(
            document=meta,
            chunks=self.ingestion.get_chunks(doc_id),
            images=[PublicImage.from_item(i) for i in self.ingestion.get_images(doc_id)],
        )

    def image_file(self, image_name: str) -> Optional[Path]:
        image_id = image_name[:-4] if image_name.endswith(".png") else image_name
        if not any(img.image_id == image_id for img in self.image_index.images):
            return None
        return self.storage.image_file(image_name)

    def search_text(self, query: str, top_k: int) -> List[TextSearchResult]:
        results = self.retriever.dedupe_text(self.text_index.search(query, top_k=top_k + 2))[:top_k]
        for r in results:
            r.accepted = r.score >= self.retriever.text_threshold
        return results

    def search_images_by_text(self, query: str, top_k: int) -> List[PublicImageSearchResult]:
        if not self.image_index.enabled:
            return []
        results = self.image_index.search_by_text(query, top_k=top_k)
        out = []
        for r in results:
            # The user explicitly asked for images, so the visual-query rule applies.
            reason = self.retriever.accept_image(r, visual_intent=True, text_pages=set())
            out.append(
                PublicImageSearchResult(
                    image=PublicImage.from_item(r.image), score=round(r.score, 4), rank=r.rank, match_reason=reason
                )
            )
        return out

    def search_images_by_image(self, content: bytes, top_k: int) -> List[PublicImageSearchResult]:
        if not self.image_index.enabled:
            return []
        if len(content) > self.config.max_upload_size_bytes:
            raise InvalidQueryImageError("The query image is too large.")
        try:
            img = self.ingestion._load_image(content)
        except Exception as exc:
            raise InvalidQueryImageError("The query image could not be decoded.") from exc
        if img is None:
            raise InvalidQueryImageError("The query image exceeds the maximum pixel count.")
        results = self.image_index.search_by_image(img, top_k=top_k)
        return [
            PublicImageSearchResult(
                image=PublicImage.from_item(r.image), score=round(r.score, 4), rank=r.rank,
                match_reason="image_similarity",
            )
            for r in results
        ]

    def chat(self, req: ChatRequest) -> ChatResponse:
        use_agent = req.mode == "agent" and self.config.agent_enabled
        return self.agent.run(
            req.query,
            top_k_text=req.top_k_text,
            top_k_images=req.top_k_images,
            include_debug=bool(req.include_debug),
            use_agent=use_agent,
        )

    # Status ----------------------------------------------------------------------------------------------------

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="unhealthy" if self.phase == "failed" else "healthy",
            app_name=self.config.app_name,
            version=self.config.app_version,
            device=self.config.device,
            ready=self.phase == "ready",
            phase=self.phase,
            indexed_documents=sum(1 for d in self.ingestion.documents_by_id.values() if d.status == "ready"),
            indexed_text_chunks=self.text_index.count(),
            indexed_images=self.image_index.count(),
            models_loaded={
                "text_embedding": self.text_model.is_loaded,
                "image_clip": self.image_index.enabled and self.image_model.is_loaded,
            },
        )

    def readiness(self) -> ReadinessResponse:
        text_state = registry.state(self.text_model.registry_key)
        image_state = registry.state(self.image_model.registry_key)

        def model_status(model, state, enabled: bool = True) -> ComponentStatus:
            if not enabled:
                return ComponentStatus(ready=True, detail="disabled by configuration")
            if model.mock_mode:
                return ComponentStatus(ready=True, detail="mock embedding backend (tests only)")
            if state and state.loaded:
                return ComponentStatus(ready=True, detail=f"loaded in {state.load_seconds:.1f}s")
            if state and state.error:
                return ComponentStatus(ready=False, detail="failed to load")
            if state and state.loading:
                return ComponentStatus(ready=False, detail="loading")
            return ComponentStatus(ready=False, detail="not loaded yet (loads on first use)")

        components = {
            "storage": ComponentStatus(
                ready=self.storage.is_writable(), detail=f"{self.storage.mode} local filesystem"
            ),
            "registry": ComponentStatus(
                ready=self.registry_loaded,
                detail=f"{len(self.ingestion.documents_by_id)} records"
                + (f"; {len(self.consistency_problems)} repaired inconsistencies" if self.consistency_problems else ""),
            ),
            "text_index": ComponentStatus(ready=self.registry_loaded, detail=f"{self.text_index.count()} chunks"),
            "image_index": ComponentStatus(
                ready=self.registry_loaded,
                detail=f"{self.image_index.count()} images" + ("" if self.image_index.enabled else " (search disabled)"),
            ),
            "text_model": model_status(self.text_model, text_state),
            "image_model": model_status(self.image_model, image_state, self.image_index.enabled),
        }
        essential = ("storage", "registry", "text_index", "text_model")
        ready = self.phase == "ready" and all(components[name].ready for name in essential)
        return ReadinessResponse(
            ready=ready,
            phase=self.phase if not self.startup_error else f"failed: {self.startup_error.split(':')[0]}",
            components=components,
            storage_mode=self.storage.mode,  # type: ignore[arg-type]
            image_search_enabled=self.image_index.enabled,
            embedding_backend=self.config.embedding_backend,
            timings_ms=self.timings_ms,
            memory_rss_mb=_rss_mb(),
        )

    def system_info(self) -> SystemInfoResponse:
        persistent = self.storage.persistent
        notice = (
            "Documents are stored on persistent storage and survive restarts."
            if persistent
            else "Demo storage is ephemeral: uploaded documents are lost when the server restarts or redeploys. "
            "Only the bundled sample documents are restored automatically."
        )
        return SystemInfoResponse(
            app_name=self.config.app_name,
            version=self.config.app_version,
            device=self.config.device,
            llm_provider=self.generator.provider if self.generator.llm_configured else "extractive",
            llm_configured=self.generator.llm_configured,
            storage_mode="persistent" if persistent else "ephemeral",
            storage_notice=notice,
            image_search_enabled=self.image_index.enabled,
            max_upload_size_bytes=self.config.max_upload_size_bytes,
            allowed_extensions=sorted(self.ingestion.allowed_extensions),
            text_model=self.text_model.identity,
            image_model=self.image_model.identity,
            thresholds={
                "text_min_score": self.retriever.text_threshold,
                "text_context_min_score": self.retriever.text_context_min_score,
                "image_min_score": self.retriever.image_min_score,
                "image_min_score_visual_query": self.retriever.image_min_score_visual_query,
                "image_min_score_colocated": self.retriever.image_min_score_colocated,
            },
            agent=self.agent.describe(),
            security={
                "redact_pii": self.config.redact_pii,
                "block_pii_requests": self.config.block_pii_requests,
                "detect_prompt_injection": self.config.detect_prompt_injection,
            },
        )

    def evaluation_report(self) -> Optional[dict]:
        path = self.config.reports_dir / "evaluation_report.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
