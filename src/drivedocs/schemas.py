"""Pydantic data schemas and API contracts for DriveDocs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core domain models (also persisted to disk)
# ---------------------------------------------------------------------------


class TextChunk(BaseModel):
    """An extracted and segmented piece of text from a document."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. doc_ab12_p1_c0")
    doc_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Original (sanitized) filename")
    page_number: int = Field(1, description="Source page number (1-indexed)")
    chunk_index: int = Field(0, description="Sequential index within the document")
    text: str = Field(..., description="Text content of the chunk")
    char_count: int = Field(0, description="Character length of text")
    section: Optional[str] = Field(None, description="Heading path for Markdown documents, e.g. 'Guide > Setup'")

    def embedding_text(self) -> str:
        """Text sent to the embedding model; heading context improves retrieval of short sections."""
        if self.section:
            return f"{self.section}\n{self.text}"
        return self.text


class ImageItem(BaseModel):
    """An image item, either uploaded directly or extracted from a PDF."""

    image_id: str = Field(..., description="Unique image identifier, e.g. img_doc_ab12_p2_0")
    doc_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Original source filename")
    page_number: Optional[int] = Field(None, description="Page number if extracted from a document")
    image_path: str = Field(
        ...,
        description="Internal storage path (file name relative to the images directory). Never returned by the API.",
    )
    url_path: str = Field(..., description="URL path to retrieve the image via the API")
    width: int = Field(0, description="Image width in pixels")
    height: int = Field(0, description="Image height in pixels")
    format: str = Field("PNG", description="Stored image format")


DocumentStatus = Literal["processing", "ready", "failed"]


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested (or failed) document."""

    doc_id: str = Field(..., description="Stable identifier derived from the content hash")
    filename: str = Field(..., description="Original sanitized filename")
    file_type: str = Field(..., description="Extension/format: pdf, txt, md, png, jpg, jpeg")
    file_size_bytes: int = Field(..., description="Size of file in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of file content for deduplication")
    page_count: int = Field(1, description="Total pages parsed")
    chunk_count: int = Field(0, description="Total text chunks created")
    image_count: int = Field(0, description="Total images extracted or registered")
    created_at: str = Field(default_factory=_utc_now)
    status: DocumentStatus = Field("ready", description="Processing status")
    error: Optional[str] = Field(None, description="Reason processing failed (status=failed)")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal ingestion warnings")
    pages_with_text: Optional[int] = Field(None, description="Pages that yielded extractable text (PDF)")
    source: Literal["upload", "sample"] = Field("upload", description="Uploaded by a user or bundled demo data")


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


class TextSearchResult(BaseModel):
    """A single retrieved text chunk with similarity score."""

    chunk: TextChunk
    score: float = Field(..., description="MiniLM cosine similarity clipped to [0, 1]")
    rank: int = Field(..., description="1-indexed retrieval rank")
    accepted: bool = Field(True, description="Score passed the text relevance threshold")


class ImageSearchResult(BaseModel):
    """A single retrieved image with similarity score."""

    image: ImageItem
    score: float = Field(..., description="CLIP cosine similarity clipped to [0, 1]")
    rank: int = Field(..., description="1-indexed retrieval rank")
    accepted: bool = Field(True, description="Image passed one of the image relevance rules")
    match_reason: Optional[str] = Field(None, description="Relevance rule that accepted the image")


class FusedEvidence(BaseModel):
    """One entry of the hybrid (Reciprocal Rank Fusion) evidence list."""

    modality: Literal["text", "image"]
    item_id: str = Field(..., description="chunk_id or image_id")
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    modality_rank: int = Field(..., description="Rank inside its own modality list")
    raw_score: float = Field(..., description="Cosine similarity in the modality's own embedding space")
    fused_score: float = Field(..., description="weight / (rrf_k + modality_rank); comparable across modalities")
    accepted: bool = Field(..., description="Whether the item passed its modality's relevance rule")


class MultimodalEvidence(BaseModel):
    """Unified evidence container combining retrieved text and image matches."""

    query: str
    text_results: List[TextSearchResult] = Field(default_factory=list)
    image_results: List[ImageSearchResult] = Field(default_factory=list)
    fused_results: List[FusedEvidence] = Field(default_factory=list)
    has_sufficient_evidence: bool = Field(True, description="Whether any evidence passed its modality threshold")
    top_confidence: float = Field(0.0, description="Top text score if text evidence passed, else top image score")
    top_text_score: float = 0.0
    top_image_score: float = 0.0
    visual_intent: bool = Field(False, description="Query explicitly asks for a diagram, chart, figure, ...")


# ---------------------------------------------------------------------------
# Answers and citations
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """Attribution citation pointing to a specific document source and page."""

    filename: str
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    snippet: str = Field("", description="Relevant text snippet from evidence")
    score: Optional[float] = Field(None, description="Retrieval similarity of the cited passage")
    verified: bool = Field(False, description="Document, page and passage were verified against the registry")
    support: Optional[float] = Field(
        None, description="Fraction of the cited claim's content words found in the cited passage (0-1)"
    )


class ImageReference(BaseModel):
    """Visual evidence reference returned to the user."""

    filename: str
    page_number: Optional[int] = None
    image_id: str
    image_url: str
    doc_id: Optional[str] = None
    score: float = 0.0
    description: Optional[str] = None
    match_reason: Optional[str] = Field(None, description="Which relevance rule accepted this image")


class PassageEvidence(BaseModel):
    """A retrieved passage exposed for inspection in the UI."""

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    section: Optional[str] = None
    text: str
    score: float
    rank: int
    accepted: bool = Field(..., description="Score passed the text relevance threshold")
    cited: bool = Field(False, description="Passage is cited in the answer")


class CitationCheck(BaseModel):
    """Summary of citation verification for one answer."""

    total: int = 0
    verified: int = 0
    removed: int = 0
    weakly_supported: int = 0
    all_verified: bool = True


class ChatResponse(BaseModel):
    """Response payload for multimodal question answering."""

    answer: str = Field(..., description="Grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="Verified text source attributions")
    images: List[ImageReference] = Field(default_factory=list, description="Relevant visual evidence")
    evidence_count: int = Field(0, description="Number of text chunks retrieved")
    confidence: float = Field(0.0, description="Retrieval confidence (see MultimodalEvidence.top_confidence)")
    answer_type: Literal["answer", "visual_only", "insufficient_evidence"] = "answer"
    provider: str = "extractive"
    warnings: List[str] = Field(default_factory=list)
    passages: List[PassageEvidence] = Field(default_factory=list)
    citation_check: CitationCheck = Field(default_factory=CitationCheck)
    latency_ms: Optional[float] = None
    debug_info: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class SearchTextRequest(BaseModel):
    """Request payload for text retrieval."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=2000, description="Search query string")
    top_k: int = Field(4, ge=1, le=20, description="Number of text chunks to retrieve")


class SearchImageRequest(BaseModel):
    """Request payload for text-to-image retrieval."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=500, description="Search prompt to match images")
    top_k: int = Field(2, ge=1, le=10, description="Number of images to retrieve")


class ChatRequest(BaseModel):
    """Request payload for multimodal question answering."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k_text: Optional[int] = Field(4, ge=1, le=10)
    top_k_images: Optional[int] = Field(2, ge=0, le=5)
    include_debug: Optional[bool] = Field(False, description="Include similarity scores and timing debug info")


# ---------------------------------------------------------------------------
# Public API responses (never expose filesystem paths)
# ---------------------------------------------------------------------------


class PublicImage(BaseModel):
    image_id: str
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    url_path: str
    width: int = 0
    height: int = 0
    format: str = "PNG"

    @classmethod
    def from_item(cls, item: ImageItem) -> "PublicImage":
        return cls(**item.model_dump(exclude={"image_path"}))


class PublicImageSearchResult(BaseModel):
    image: PublicImage
    score: float
    rank: int
    match_reason: Optional[str] = None


class HealthResponse(BaseModel):
    """Lightweight liveness response; never loads models."""

    status: str = "healthy"
    app_name: str
    version: str
    device: str
    ready: bool = False
    phase: str = "starting"
    indexed_documents: int
    indexed_text_chunks: int
    indexed_images: int
    models_loaded: Dict[str, bool]


class ComponentStatus(BaseModel):
    ready: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    ready: bool
    phase: str
    components: Dict[str, ComponentStatus]
    storage_mode: Literal["persistent", "ephemeral"]
    image_search_enabled: bool
    embedding_backend: str
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    memory_rss_mb: Optional[float] = None


class SystemInfoResponse(BaseModel):
    app_name: str
    version: str
    device: str
    llm_provider: str
    llm_configured: bool
    storage_mode: Literal["persistent", "ephemeral"]
    storage_notice: str
    image_search_enabled: bool
    max_upload_size_bytes: int
    allowed_extensions: List[str]
    text_model: str
    image_model: str
    thresholds: Dict[str, float]
    authentication: bool = False


class UploadResponse(BaseModel):
    """Response returned upon document ingestion."""

    message: str
    document: DocumentMetadata
    chunks_indexed: int
    images_indexed: int
    duplicate: bool = False
    warnings: List[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """List of all registered documents in the system."""

    documents: List[DocumentMetadata]
    total_documents: int
    total_chunks: int
    total_images: int
    failed_documents: int = 0


class DocumentDetailResponse(BaseModel):
    document: DocumentMetadata
    chunks: List[TextChunk] = Field(default_factory=list)
    images: List[PublicImage] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    message: str
    doc_id: str
    chunks_removed: int
    images_removed: int


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorBody
