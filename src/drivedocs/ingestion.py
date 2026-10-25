"""Document ingestion, parsing, image extraction, and security validation module."""

from __future__ import annotations

import hashlib
import io
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image

from drivedocs.config import settings
from drivedocs.chunking import TextChunker
from drivedocs.schemas import DocumentMetadata, ImageItem, TextChunk


class IngestionError(Exception):
    """Base exception for document ingestion failures."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file format is not supported."""
    pass


class FileSizeLimitExceededError(IngestionError):
    """Raised when an uploaded file exceeds the configured size limit."""
    pass


class MalformedDocumentError(IngestionError):
    """Raised when a document is corrupted or unreadable."""
    pass


class DuplicateDocumentError(IngestionError):
    """Raised when a document with identical content hash has already been ingested."""
    def __init__(self, message: str, existing_doc: DocumentMetadata) -> None:
        super().__init__(message)
        self.existing_doc = existing_doc


@dataclass
class ParsedDocument:
    """Container holding extracted document metadata, text chunks, and images."""
    metadata: DocumentMetadata
    chunks: List[TextChunk] = field(default_factory=list)
    images: List[ImageItem] = field(default_factory=list)


def check_ocr_support() -> Tuple[bool, str]:
    """Check if optional OCR (e.g. pytesseract) is installed and operational."""
    try:
        import pytesseract  # type: ignore
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract OCR is available (version {version})"
    except Exception as e:
        return (
            False,
            "Optional OCR is not available. Scanned image-only PDFs will yield visual evidence "
            "via CLIP image retrieval, but optical text character extraction requires 'pytesseract' "
            "and the Tesseract OCR engine binary installed on the host system."
        )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks and illegal characters."""
    # Strip path components
    basename = os.path.basename(filename)
    # Remove control characters, null bytes, backslashes
    cleaned = re.sub(r"[^\w\s\.-]", "_", basename)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)  # prevent ../
    cleaned = cleaned.strip()
    if not cleaned or cleaned.startswith("."):
        return f"doc_{uuid.uuid4().hex[:8]}.txt"
    return cleaned[:100]


def compute_content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file byte content for deduplication and integrity."""
    return hashlib.sha256(data).hexdigest()


class DocumentIngestionManager:
    """Manages parsing, image extraction, deduplication, and persistence of documents."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        chunker: Optional[TextChunker] = None,
        max_file_size: Optional[int] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir or settings.storage_dir)
        self.images_dir = Path(images_dir or settings.images_dir)
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators,
        )
        self.max_file_size = max_file_size or settings.max_upload_size_bytes
        self.allowed_extensions = set(settings.allowed_extensions)

        # In-memory document & hash registries
        self.documents_by_id: Dict[str, DocumentMetadata] = {}
        self.hash_to_doc_id: Dict[str, str] = {}
        self.chunks_by_doc_id: Dict[str, List[TextChunk]] = {}
        self.images_by_doc_id: Dict[str, List[ImageItem]] = {}

        self.images_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: str, content: bytes) -> str:
        """Validate file size, extension, and content."""
        if len(content) == 0:
            raise MalformedDocumentError(f"File '{filename}' is empty (0 bytes).")

        if len(content) > self.max_file_size:
            raise FileSizeLimitExceededError(
                f"File size ({len(content)} bytes) exceeds the maximum allowed limit of {self.max_file_size} bytes."
            )

        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"File type '{ext}' is not supported. Allowed formats: {sorted(list(self.allowed_extensions))}"
            )
        return ext

    def ingest_bytes(
        self,
        filename: str,
        content: bytes,
        allow_duplicates: bool = False,
    ) -> ParsedDocument:
        """Parse raw file bytes, perform chunking and image extraction, and register the document."""
        safe_filename = sanitize_filename(filename)
        ext = self.validate_file(safe_filename, content)
        content_hash = compute_content_hash(content)

        # Duplicate check
        if not allow_duplicates and content_hash in self.hash_to_doc_id:
            existing_id = self.hash_to_doc_id[content_hash]
            existing_doc = self.documents_by_id.get(existing_id)
            if existing_doc:
                raise DuplicateDocumentError(
                    f"Document with identical content hash '{content_hash[:12]}...' already exists as ID '{existing_id}' ('{existing_doc.filename}').",
                    existing_doc=existing_doc,
                )

        doc_id = f"doc_{content_hash[:12]}"

        # Dispatch based on extension
        if ext == ".pdf":
            parsed = self._parse_pdf(doc_id, safe_filename, content, content_hash)
        elif ext in [".txt", ".md"]:
            parsed = self._parse_text(doc_id, safe_filename, content, content_hash, ext[1:])
        elif ext in [".png", ".jpg", ".jpeg"]:
            parsed = self._parse_standalone_image(doc_id, safe_filename, content, content_hash, ext[1:])
        else:
            raise UnsupportedFileTypeError(f"Unhandled file extension: {ext}")

        # Register document
        self.documents_by_id[doc_id] = parsed.metadata
        self.hash_to_doc_id[content_hash] = doc_id
        self.chunks_by_doc_id[doc_id] = parsed.chunks
        self.images_by_doc_id[doc_id] = parsed.images

        return parsed

    def ingest_file(self, file_path: str | Path, allow_duplicates: bool = False) -> ParsedDocument:
        """Ingest a file from local filesystem."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            content = f.read()
        return self.ingest_bytes(path.name, content, allow_duplicates=allow_duplicates)

    def _parse_text(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_hash: str,
        file_type: str,
    ) -> ParsedDocument:
        """Parse plain text or Markdown document."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except Exception as e:
                raise MalformedDocumentError(f"Unable to decode text file '{filename}': {str(e)}")

        chunks = self.chunker.chunk_page_text(
            text=text,
            doc_id=doc_id,
            filename=filename,
            page_number=1,
            start_chunk_idx=0,
        )

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(content),
            content_hash=content_hash,
            page_count=1,
            chunk_count=len(chunks),
            image_count=0,
        )

        return ParsedDocument(metadata=metadata, chunks=chunks, images=[])

    def _parse_pdf(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_hash: str,
    ) -> ParsedDocument:
        """Parse PDF document with page-level text extraction and embedded image extraction."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
        except Exception as e:
            raise MalformedDocumentError(f"Failed to parse PDF '{filename}': {str(e)}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise MalformedDocumentError(f"PDF '{filename}' has 0 pages.")

        all_chunks: List[TextChunk] = []
        all_images: List[ImageItem] = []
        chunk_counter = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1

            # 1. Extract text with page numbers
            page_text = ""
            try:
                extracted = page.extract_text()
                if extracted:
                    page_text = extracted
            except Exception:
                page_text = ""

            if page_text.strip():
                page_chunks = self.chunker.chunk_page_text(
                    text=page_text,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_num,
                    start_chunk_idx=chunk_counter,
                )
                chunk_counter += len(page_chunks)
                all_chunks.extend(page_chunks)

            # 2. Extract embedded images from this page
            try:
                for img_idx, img_obj in enumerate(page.images):
                    try:
                        img_bytes = img_obj.data
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        width, height = pil_img.size

                        # Filter out tiny icon artifacts (e.g. < 40x40 pixels)
                        if width < 40 or height < 40:
                            continue

                        image_id = f"img_{doc_id}_p{page_num}_{img_idx}"
                        image_filename = f"{image_id}.png"
                        out_path = self.images_dir / image_filename
                        pil_img.save(out_path, format="PNG")

                        all_images.append(
                            ImageItem(
                                image_id=image_id,
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page_num,
                                image_path=str(out_path.resolve()),
                                url_path=f"/static/images/{image_filename}",
                                width=width,
                                height=height,
                                format="PNG",
                            )
                        )
                    except Exception:
                        continue
            except Exception:
                pass

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            file_size_bytes=len(content),
            content_hash=content_hash,
            page_count=total_pages,
            chunk_count=len(all_chunks),
            image_count=len(all_images),
        )

        return ParsedDocument(metadata=metadata, chunks=all_chunks, images=all_images)

    def _parse_standalone_image(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_hash: str,
        file_type: str,
    ) -> ParsedDocument:
        """Parse standalone uploaded image (PNG/JPEG) and store for indexing."""
        try:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            width, height = pil_img.size
        except Exception as e:
            raise MalformedDocumentError(f"Corrupt or unreadable image '{filename}': {str(e)}")

        image_id = f"img_{doc_id}"
        image_filename = f"{image_id}.png"
        out_path = self.images_dir / image_filename
        pil_img.save(out_path, format="PNG")

        image_item = ImageItem(
            image_id=image_id,
            doc_id=doc_id,
            filename=filename,
            page_number=1,
            image_path=str(out_path.resolve()),
            url_path=f"/static/images/{image_filename}",
            width=width,
            height=height,
            format="PNG",
        )

        # Image documents have 0 text chunks and 1 image item
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(content),
            content_hash=content_hash,
            page_count=1,
            chunk_count=0,
            image_count=1,
        )

        return ParsedDocument(metadata=metadata, chunks=[], images=[image_item])

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        return self.documents_by_id.get(doc_id)

    def list_documents(self) -> List[DocumentMetadata]:
        return list(self.documents_by_id.values())

    def get_chunks(self, doc_id: str) -> List[TextChunk]:
        return self.chunks_by_doc_id.get(doc_id, [])

    def get_images(self, doc_id: str) -> List[ImageItem]:
        return self.images_by_doc_id.get(doc_id, [])
