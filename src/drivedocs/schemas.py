"""Pydantic data schemas and contracts for DriveDocs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Represents an extracted and segmented piece of text from a document."""
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. doc1_p1_c0")
    doc_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(1, description="Source page number (1-indexed)")
    chunk_index: int = Field(0, description="Sequential index within the document")
    text: str = Field(..., description="Text content of the chunk")
    char_count: int = Field(0, description="Character length of text")


class ImageItem(BaseModel):
    """Represents an image item, either uploaded directly or extracted from a PDF."""
    image_id: str = Field(..., description="Unique image identifier, e.g. img_doc1_p2_0")
    doc_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Original source filename")
    page_number: Optional[int] = Field(None, description="Page number if extracted from document")
    image_path: str = Field(..., description="Filesystem absolute or relative path to image")
    url_path: str = Field(..., description="Web URL path to retrieve image via API")
    width: int = Field(0, description="Image width in pixels")
    height: int = Field(0, description="Image height in pixels")
    format: str = Field("PNG", description="Image format, e.g. PNG, JPEG")


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested document."""
    doc_id: str = Field(..., description="Stable unique identifier (hash-derived or uuid)")
    filename: str = Field(..., description="Original sanitized filename")
    file_type: str = Field(..., description="Extension/format: pdf, txt, md, image")
    file_size_bytes: int = Field(..., description="Size of file in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of file content for deduplication")
    page_count: int = Field(1, description="Total pages parsed")
    chunk_count: int = Field(0, description="Total text chunks created")
    image_count: int = Field(0, description="Total images extracted or registered")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TextSearchResult(BaseModel):
    """A single retrieved text chunk with similarity score."""
    chunk: TextChunk
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    rank: int = Field(..., description="1-indexed retrieval rank")


class ImageSearchResult(BaseModel):
    """A single retrieved image with similarity score."""
    image: ImageItem
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    rank: int = Field(..., description="1-indexed retrieval rank")


class MultimodalEvidence(BaseModel):
    """Unified evidence container combining retrieved text and image matches."""
    query: str
    text_results: List[TextSearchResult] = Field(default_factory=list)
    image_results: List[ImageSearchResult] = Field(default_factory=list)
    has_sufficient_evidence: bool = Field(True, description="Whether evidence exceeds minimum similarity threshold")
    top_confidence: float = Field(0.0, description="Highest similarity score among retrieved items")


class Citation(BaseModel):
    """Attribution citation pointing to specific document source and page."""
    filename: str
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: str = Field("", description="Relevant text snippet from evidence")


class ImageReference(BaseModel):
    """Visual evidence reference returned to user."""
    filename: str
    page_number: Optional[int] = None
    image_id: str
    image_url: str
    score: float = 0.0
    description: Optional[str] = None


class SearchTextRequest(BaseModel):
    """Request payload for text retrieval."""
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(4, ge=1, le=20, description="Number of text chunks to retrieve")


class SearchImageRequest(BaseModel):
    """Request payload for text-to-image retrieval."""
    query: str = Field(..., min_length=1, description="Search prompt to match images")
    top_k: int = Field(2, ge=1, le=10, description="Number of images to retrieve")


class ChatRequest(BaseModel):
    """Request payload for multimodal question answering."""
    query: str = Field(..., min_length=1, description="User question")
    top_k_text: Optional[int] = Field(4, ge=1, le=10)
    top_k_images: Optional[int] = Field(2, ge=0, le=5)
    include_debug: Optional[bool] = Field(False, description="Include similarity scores and prompt debug info")


class ChatResponse(BaseModel):
    """Response payload for multimodal question answering."""
    answer: str = Field(..., description="Synthesized grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="Text source attributions")
    images: List[ImageReference] = Field(default_factory=list, description="Relevant visual evidence")
    evidence_count: int = Field(0, description="Number of text chunks retrieved")
    confidence: float = Field(0.0, description="Retrieval confidence score")
    debug_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """System health and index status response."""
    status: str = "healthy"
    app_name: str
    version: str
    device: str
    indexed_documents: int
    indexed_text_chunks: int
    indexed_images: int
    models_loaded: Dict[str, bool]


class UploadResponse(BaseModel):
    """Response returned upon document ingestion."""
    message: str
    document: DocumentMetadata
    chunks_indexed: int
    images_indexed: int


class DocumentListResponse(BaseModel):
    """List of all registered documents in the system."""
    documents: List[DocumentMetadata]
    total_documents: int
    total_chunks: int
    total_images: int
