"""Document ingestion: validation, parsing, image extraction and the document registry.

Uploaded files are treated as untrusted input:

* file names are sanitized and never used to build storage paths (IDs are derived from the SHA-256 content hash);
* extensions must match the file's magic bytes / decodability;
* size, page, image-count and pixel limits bound the work done per upload;
* parsing is separated from registration so the service layer can index first and only then commit.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from drivedocs.chunking import TextChunker, reflow_lines
from drivedocs.config import settings
from drivedocs.guardrails import find_pii, is_injection, redact_pii
from drivedocs.schemas import DocumentMetadata, ImageItem, TextChunk

logger = logging.getLogger("drivedocs.ingestion")
# pypdf logs every recoverable structural oddity at WARNING; keep logs readable (failures still raise).
logging.getLogger("pypdf").setLevel(logging.ERROR)

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_FAILED_RECORDS = 100


class IngestionError(Exception):
    """Base exception for document ingestion failures."""

    code = "ingestion_error"


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file format is not supported or its content does not match its extension."""

    code = "unsupported_file_type"


class FileSizeLimitExceededError(IngestionError):
    """Raised when an uploaded file exceeds the configured size limit."""

    code = "file_too_large"


class MalformedDocumentError(IngestionError):
    """Raised when a document is empty, corrupted or unreadable."""

    code = "malformed_document"


class NoExtractableContentError(MalformedDocumentError):
    """Raised when a document parses but yields neither text nor images."""

    code = "no_extractable_content"


class StorageLimitError(IngestionError):
    """Raised when the configured maximum number of documents is reached."""

    code = "storage_limit_reached"


class DuplicateDocumentError(IngestionError):
    """Raised when a document with an identical content hash has already been ingested."""

    code = "duplicate_document"

    def __init__(self, message: str, existing_doc: DocumentMetadata) -> None:
        super().__init__(message)
        self.existing_doc = existing_doc


@dataclass
class ParsedDocument:
    """Container holding extracted document metadata, text chunks, and images."""

    metadata: DocumentMetadata
    chunks: List[TextChunk] = field(default_factory=list)
    images: List[ImageItem] = field(default_factory=list)
    duplicate: bool = False


def check_ocr_support() -> Tuple[bool, str]:
    """Check if optional OCR (pytesseract + Tesseract binary) is available. OCR is not used by default."""
    try:
        import pytesseract  # type: ignore

        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract OCR is available (version {version})"
    except Exception:
        return (
            False,
            "Optional OCR is not available. Scanned image-only PDFs will yield visual evidence "
            "via CLIP image retrieval, but optical text character extraction requires 'pytesseract' "
            "and the Tesseract OCR engine binary installed on the host system.",
        )


_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def sanitize_filename(filename: str, max_length: int = 120) -> str:
    """Return a display-safe file name: no directories, control characters or traversal sequences.

    The result is only used for display and citations; storage paths are always derived from content hashes.
    """
    name = unicodedata.normalize("NFKC", filename or "")
    # Strip any directory component, regardless of the separator style or host OS.
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"[^\w\s.\-()]", "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if not stem or stem.lower() in _WINDOWS_RESERVED:
        return f"doc_{uuid.uuid4().hex[:8]}" + (f".{ext}" if ext else ".txt")
    if len(name) > max_length:
        keep = max_length - (len(ext) + 1 if ext else 0)
        name = stem[:keep].rstrip() + (f".{ext}" if ext else "")
    return name


def compute_content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file byte content for deduplication and integrity."""
    return hashlib.sha256(data).hexdigest()


def doc_id_for_hash(content_hash: str) -> str:
    return f"doc_{content_hash[:12]}"


def decode_text(content: bytes, filename: str) -> str:
    """Decode a text upload, rejecting binary content disguised as text."""
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = content.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise MalformedDocumentError(f"'{filename}' is not valid UTF-16 text.") from exc
    else:
        if b"\x00" in content:
            raise UnsupportedFileTypeError(f"'{filename}' appears to be a binary file, not text.")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content.decode("cp1252")
            except UnicodeDecodeError as exc:
                raise MalformedDocumentError(
                    f"'{filename}' could not be decoded as UTF-8 or Windows-1252 text."
                ) from exc
    if text:
        control = sum(1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\t\n\r\f")
        if control / len(text) > 0.02:
            raise UnsupportedFileTypeError(f"'{filename}' contains binary data, not text.")
    return text


def sniff_matches_extension(ext: str, content: bytes) -> bool:
    if ext == ".pdf":
        return b"%PDF-" in content[:1024]
    if ext in IMAGE_EXTENSIONS:
        return content.startswith(b"\x89PNG\r\n\x1a\n") or content.startswith(b"\xff\xd8\xff")
    return True  # text formats are validated by decode_text


class DocumentIngestionManager:
    """Parses documents and keeps the in-memory document registry."""

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
        self.allowed_extensions = {e.lower() for e in settings.allowed_extensions}
        self.max_pdf_pages = settings.max_pdf_pages
        self.max_images_per_document = settings.max_images_per_document
        self.max_image_pixels = settings.max_image_pixels
        self.max_stored_image_side = settings.max_stored_image_side
        self.min_extracted_image_side = settings.min_extracted_image_side
        self.max_documents = settings.max_documents

        self.documents_by_id: Dict[str, DocumentMetadata] = {}
        self.hash_to_doc_id: Dict[str, str] = {}
        self.chunks_by_doc_id: Dict[str, List[TextChunk]] = {}
        self.images_by_doc_id: Dict[str, List[ImageItem]] = {}

        self.images_dir.mkdir(parents=True, exist_ok=True)

    # Validation ------------------------------------------------------------------------------------------------

    def validate_file(self, filename: str, content: bytes) -> str:
        """Validate size, extension and content signature. Returns the normalized extension."""
        if len(content) == 0:
            raise MalformedDocumentError(f"File '{filename}' is empty (0 bytes).")
        if len(content) > self.max_file_size:
            raise FileSizeLimitExceededError(
                f"File size ({len(content)} bytes) exceeds the maximum allowed limit of {self.max_file_size} bytes."
            )
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"File type '{ext or '(none)'}' is not supported. Allowed formats: {sorted(self.allowed_extensions)}"
            )
        if not sniff_matches_extension(ext, content):
            raise UnsupportedFileTypeError(f"The content of '{filename}' does not match its '{ext}' extension.")
        return ext

    # Parsing ---------------------------------------------------------------------------------------------------

    def parse_bytes(self, filename: str, content: bytes, allow_duplicates: bool = False) -> ParsedDocument:
        """Validate and parse a file without registering it.

        If an identical document already exists, raises DuplicateDocumentError, or (allow_duplicates=True) returns
        the existing document with ``duplicate=True``. Duplicates are never parsed or indexed twice.
        """
        safe_filename = sanitize_filename(filename)
        ext = self.validate_file(safe_filename, content)
        content_hash = compute_content_hash(content)

        existing_id = self.hash_to_doc_id.get(content_hash)
        existing = self.documents_by_id.get(existing_id) if existing_id else None
        if existing is not None and existing.status == "ready":
            if allow_duplicates:
                return ParsedDocument(
                    metadata=existing,
                    chunks=list(self.chunks_by_doc_id.get(existing.doc_id, [])),
                    images=list(self.images_by_doc_id.get(existing.doc_id, [])),
                    duplicate=True,
                )
            raise DuplicateDocumentError(
                f"Document with identical content hash '{content_hash[:12]}...' already exists as ID "
                f"'{existing.doc_id}' ('{existing.filename}').",
                existing_doc=existing,
            )

        ready_docs = sum(1 for d in self.documents_by_id.values() if d.status == "ready")
        if ready_docs >= self.max_documents:
            raise StorageLimitError(f"The document limit ({self.max_documents}) has been reached.")

        doc_id = doc_id_for_hash(content_hash)
        if ext == ".pdf":
            parsed = self._parse_pdf(doc_id, safe_filename, content, content_hash)
        elif ext in TEXT_EXTENSIONS:
            parsed = self._parse_text(doc_id, safe_filename, content, content_hash, ext[1:])
        elif ext in IMAGE_EXTENSIONS:
            parsed = self._parse_standalone_image(doc_id, safe_filename, content, content_hash, ext[1:])
        else:
            raise UnsupportedFileTypeError(f"Unhandled file extension: {ext}")
        return self._apply_guardrails(parsed)

    @staticmethod
    def _apply_guardrails(parsed: "ParsedDocument") -> "ParsedDocument":
        """Redact personal data before anything is embedded or indexed, and flag instruction-like text."""
        pii_values: Dict[str, set] = {}
        injected_pages: List[int] = []
        for chunk in parsed.chunks:
            if settings.redact_pii:
                for kind, value in find_pii(chunk.text):
                    pii_values.setdefault(kind, set()).add(value)
                chunk.text, _ = redact_pii(chunk.text)
                if chunk.section:
                    chunk.section, _ = redact_pii(chunk.section)
                chunk.char_count = len(chunk.text)
            if settings.detect_prompt_injection:
                if is_injection(chunk.text) and chunk.page_number not in injected_pages:
                    injected_pages.append(chunk.page_number)
                if chunk.section and is_injection(chunk.section):
                    # Headings are prepended to embedded text and quoted with list items: drop injected parts.
                    parts = [p for p in chunk.section.split(" > ") if not is_injection(p)]
                    chunk.section = " > ".join(parts) or None
                    if chunk.page_number not in injected_pages:
                        injected_pages.append(chunk.page_number)
        if pii_values:
            summary = ", ".join(f"{len(v)} {k.lower()}" for k, v in sorted(pii_values.items()))
            parsed.metadata.warnings.append(
                f"Personal data was redacted before indexing ({summary}); it cannot be retrieved or returned."
            )
        if injected_pages:
            pages = ", ".join(str(p) for p in injected_pages)
            parsed.metadata.warnings.append(
                f"Instruction-like text (possible prompt injection) was found on page(s) {pages}. It is never "
                "followed, quoted in answers or sent to a language model."
            )
        return parsed

    def ingest_bytes(self, filename: str, content: bytes, allow_duplicates: bool = False) -> ParsedDocument:
        """Parse and register a document (no vector indexing; see DriveDocsService for the full pipeline)."""
        parsed = self.parse_bytes(filename, content, allow_duplicates=allow_duplicates)
        if not parsed.duplicate:
            self.register(parsed)
        return parsed

    def ingest_file(self, file_path: str | Path, allow_duplicates: bool = False) -> ParsedDocument:
        """Ingest a file from the local filesystem."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path.name}")
        return self.ingest_bytes(path.name, path.read_bytes(), allow_duplicates=allow_duplicates)

    def _parse_text(
        self, doc_id: str, filename: str, content: bytes, content_hash: str, file_type: str
    ) -> ParsedDocument:
        text = decode_text(content, filename)
        if not text.strip():
            raise NoExtractableContentError(f"'{filename}' contains no text.")

        if file_type == "md":
            chunks = self.chunker.chunk_markdown(text, doc_id=doc_id, filename=filename)
            if not chunks:  # e.g. a document consisting only of headings
                chunks = self.chunker.chunk_page_text(text, doc_id, filename, page_number=1)
        else:
            chunks = self.chunker.chunk_page_text(text, doc_id, filename, page_number=1)
        if not chunks:
            raise NoExtractableContentError(f"'{filename}' contains no indexable text.")

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(content),
            content_hash=content_hash,
            page_count=1,
            chunk_count=len(chunks),
            image_count=0,
            pages_with_text=1,
        )
        return ParsedDocument(metadata=metadata, chunks=chunks, images=[])

    def _parse_pdf(self, doc_id: str, filename: str, content: bytes, content_hash: str) -> ParsedDocument:
        try:
            import pypdf

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = pypdf.PdfReader(io.BytesIO(content))
                if reader.is_encrypted:
                    try:
                        if not reader.decrypt(""):
                            raise MalformedDocumentError(f"PDF '{filename}' is password-protected.")
                    except MalformedDocumentError:
                        raise
                    except Exception as exc:
                        raise MalformedDocumentError(f"PDF '{filename}' is encrypted and cannot be read.") from exc
                total_pages = len(reader.pages)
        except MalformedDocumentError:
            raise
        except Exception as exc:
            raise MalformedDocumentError(f"'{filename}' is not a readable PDF (the file may be corrupted).") from exc

        if total_pages == 0:
            raise MalformedDocumentError(f"PDF '{filename}' has 0 pages.")
        if total_pages > self.max_pdf_pages:
            raise MalformedDocumentError(
                f"PDF '{filename}' has {total_pages} pages; the limit is {self.max_pdf_pages}."
            )

        doc_warnings: List[str] = []
        all_chunks: List[TextChunk] = []
        all_images: List[ImageItem] = []
        seen_image_hashes: set[str] = set()
        pages_with_text = 0
        failed_text_pages = 0
        skipped_images = 0
        image_limit_hit = False

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
                failed_text_pages += 1

            if page_text.strip():
                page_chunks = self.chunker.chunk_page_text(
                    text=reflow_lines(page_text),
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_num,
                    start_chunk_idx=len(all_chunks),
                )
                if page_chunks:
                    pages_with_text += 1
                    all_chunks.extend(page_chunks)

            if image_limit_hit:
                continue
            try:
                page_images = list(page.images)
            except Exception:
                page_images = []
            for img_idx, img_obj in enumerate(page_images):
                if len(all_images) >= self.max_images_per_document:
                    image_limit_hit = True
                    break
                try:
                    pil_img = self._load_image(img_obj.data)
                except Exception:
                    skipped_images += 1
                    continue
                if pil_img is None:
                    skipped_images += 1
                    continue
                width, height = pil_img.size
                if width < self.min_extracted_image_side or height < self.min_extracted_image_side:
                    continue  # icons, bullets and other decorative artifacts
                png_bytes = self._to_png_bytes(pil_img)
                digest = hashlib.sha256(png_bytes).hexdigest()
                if digest in seen_image_hashes:
                    continue  # the same embedded image repeated (e.g. a logo on every page)
                seen_image_hashes.add(digest)
                image_id = f"img_{doc_id}_p{page_num}_{img_idx}"
                all_images.append(
                    self._store_image(image_id, doc_id, filename, page_num, png_bytes, pil_img.size)
                )

        if failed_text_pages:
            doc_warnings.append(f"Text extraction failed on {failed_text_pages} page(s).")
        if skipped_images:
            doc_warnings.append(f"{skipped_images} embedded image(s) could not be decoded and were skipped.")
        if image_limit_hit:
            doc_warnings.append(
                f"Only the first {self.max_images_per_document} images were extracted (per-document limit)."
            )
        if not all_chunks and not all_images:
            raise NoExtractableContentError(
                f"No extractable text or images were found in '{filename}'. It may be a scanned PDF; "
                "OCR is not enabled in this deployment."
            )
        if not all_chunks:
            doc_warnings.append("No extractable text layer (scanned PDF?). Only image search is available for it.")
        elif pages_with_text < total_pages:
            doc_warnings.append(f"{total_pages - pages_with_text} of {total_pages} page(s) had no extractable text.")

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            file_size_bytes=len(content),
            content_hash=content_hash,
            page_count=total_pages,
            chunk_count=len(all_chunks),
            image_count=len(all_images),
            warnings=doc_warnings,
            pages_with_text=pages_with_text,
        )
        return ParsedDocument(metadata=metadata, chunks=all_chunks, images=all_images)

    def _parse_standalone_image(
        self, doc_id: str, filename: str, content: bytes, content_hash: str, file_type: str
    ) -> ParsedDocument:
        try:
            pil_img = self._load_image(content)
        except Exception as exc:
            raise MalformedDocumentError(f"'{filename}' is a corrupt or unreadable image.") from exc
        if pil_img is None:
            raise MalformedDocumentError(
                f"'{filename}' exceeds the maximum image size of {self.max_image_pixels} pixels."
            )

        image_id = f"img_{doc_id}"
        item = self._store_image(image_id, doc_id, filename, 1, self._to_png_bytes(pil_img), pil_img.size)
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
        return ParsedDocument(metadata=metadata, chunks=[], images=[item])

    # Image helpers ---------------------------------------------------------------------------------------------

    def _load_image(self, data: bytes) -> Optional[Image.Image]:
        """Open and normalize an image (EXIF orientation, transparency → white, RGB, bounded size).

        Returns None if the image exceeds the pixel limit (decompression-bomb guard).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                img = Image.open(io.BytesIO(data))
                if img.size[0] * img.size[1] > self.max_image_pixels:
                    return None
                img.load()
            except (Image.DecompressionBombError, Image.DecompressionBombWarning):
                return None
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
        if max(img.size) > self.max_stored_image_side:
            img.thumbnail((self.max_stored_image_side, self.max_stored_image_side))
        return img

    @staticmethod
    def _to_png_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _store_image(
        self, image_id: str, doc_id: str, filename: str, page_num: int, png_bytes: bytes, size: Tuple[int, int]
    ) -> ImageItem:
        image_filename = f"{image_id}.png"
        out_path = self.images_dir / image_filename
        self.images_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)
        return ImageItem(
            image_id=image_id,
            doc_id=doc_id,
            filename=filename,
            page_number=page_num,
            image_path=str(out_path.resolve()),
            url_path=f"/static/images/{image_filename}",
            width=size[0],
            height=size[1],
            format="PNG",
        )

    def discard(self, parsed: ParsedDocument) -> None:
        """Delete image files written while parsing a document that will not be committed."""
        if parsed.duplicate:
            return
        for item in parsed.images:
            path = Path(item.image_path)
            if not path.is_absolute():
                path = self.images_dir / path
            try:
                path.unlink()
            except OSError:
                pass

    # Registry --------------------------------------------------------------------------------------------------

    def register(self, parsed: ParsedDocument, source: str = "upload") -> None:
        meta = parsed.metadata
        if meta.source != source:
            meta.source = source  # type: ignore[assignment]
        meta.status = "ready"
        meta.error = None
        self.documents_by_id[meta.doc_id] = meta
        self.hash_to_doc_id[meta.content_hash] = meta.doc_id
        self.chunks_by_doc_id[meta.doc_id] = list(parsed.chunks)
        self.images_by_doc_id[meta.doc_id] = list(parsed.images)

    def record_failure(self, filename: str, content: bytes, error: str) -> Optional[DocumentMetadata]:
        """Record a document that failed processing so it is visible instead of silently dropped."""
        if not content:
            return None
        safe_filename = sanitize_filename(filename)
        content_hash = compute_content_hash(content)
        doc_id = doc_id_for_hash(content_hash)
        existing = self.documents_by_id.get(doc_id)
        if existing is not None and existing.status == "ready":
            return existing
        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=safe_filename,
            file_type=Path(safe_filename).suffix.lower().lstrip(".") or "unknown",
            file_size_bytes=len(content),
            content_hash=content_hash,
            page_count=0,
            chunk_count=0,
            image_count=0,
            status="failed",
            error=error,
        )
        self.documents_by_id[doc_id] = meta
        failed = sorted(
            (d for d in self.documents_by_id.values() if d.status == "failed"), key=lambda d: d.created_at
        )
        for old in failed[: max(0, len(failed) - MAX_FAILED_RECORDS)]:
            self.documents_by_id.pop(old.doc_id, None)
        return meta

    def remove_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        meta = self.documents_by_id.pop(doc_id, None)
        if meta is None:
            return None
        if self.hash_to_doc_id.get(meta.content_hash) == doc_id:
            self.hash_to_doc_id.pop(meta.content_hash, None)
        self.chunks_by_doc_id.pop(doc_id, None)
        self.images_by_doc_id.pop(doc_id, None)
        return meta

    def registry_payload(self) -> dict:
        return {"version": 1, "documents": [d.model_dump() for d in self.documents_by_id.values()]}

    def load_registry_payload(
        self, payload: dict, chunks: List[TextChunk], images: List[ImageItem]
    ) -> List[str]:
        """Restore the registry from disk, reconciling it with the chunks/images present in the indexes.

        Returns a list of human-readable consistency problems (empty when consistent).
        """
        problems: List[str] = []
        self.documents_by_id.clear()
        self.hash_to_doc_id.clear()
        self.chunks_by_doc_id.clear()
        self.images_by_doc_id.clear()

        for raw in payload.get("documents", []):
            meta = DocumentMetadata(**raw)
            self.documents_by_id[meta.doc_id] = meta
            if meta.status == "ready":
                self.hash_to_doc_id[meta.content_hash] = meta.doc_id
                self.chunks_by_doc_id[meta.doc_id] = []
                self.images_by_doc_id[meta.doc_id] = []

        for chunk in chunks:
            if chunk.doc_id in self.chunks_by_doc_id:
                self.chunks_by_doc_id[chunk.doc_id].append(chunk)
            else:
                problems.append(f"Index contains chunks for unregistered document {chunk.doc_id}.")
        for image in images:
            if image.doc_id in self.images_by_doc_id:
                self.images_by_doc_id[image.doc_id].append(image)
            else:
                problems.append(f"Index contains images for unregistered document {image.doc_id}.")

        for doc_id, meta in self.documents_by_id.items():
            if meta.status != "ready":
                continue
            n_chunks = len(self.chunks_by_doc_id.get(doc_id, []))
            n_images = len(self.images_by_doc_id.get(doc_id, []))
            if n_chunks != meta.chunk_count or n_images != meta.image_count:
                problems.append(
                    f"Document {doc_id} expects {meta.chunk_count} chunks/{meta.image_count} images but the index "
                    f"has {n_chunks}/{n_images}."
                )
        return problems

    # Queries ---------------------------------------------------------------------------------------------------

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        return self.documents_by_id.get(doc_id)

    def list_documents(self) -> List[DocumentMetadata]:
        return sorted(self.documents_by_id.values(), key=lambda d: d.created_at)

    def get_chunks(self, doc_id: str) -> List[TextChunk]:
        return self.chunks_by_doc_id.get(doc_id, [])

    def get_images(self, doc_id: str) -> List[ImageItem]:
        return self.images_by_doc_id.get(doc_id, [])

    def find_by_filename(self, filename: str) -> List[DocumentMetadata]:
        return [d for d in self.documents_by_id.values() if d.filename == filename and d.status == "ready"]
