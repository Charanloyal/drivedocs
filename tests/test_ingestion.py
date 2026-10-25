"""Unit tests for document parsing, image extraction, and security checks."""

import io
from pathlib import Path
from PIL import Image
import pytest

from drivedocs.ingestion import (
    DocumentIngestionManager,
    DuplicateDocumentError,
    FileSizeLimitExceededError,
    MalformedDocumentError,
    UnsupportedFileTypeError,
    check_ocr_support,
    compute_content_hash,
    sanitize_filename,
)


@pytest.fixture
def temp_ingestion_mgr(tmp_path):
    storage_dir = tmp_path / "storage"
    images_dir = tmp_path / "images"
    return DocumentIngestionManager(storage_dir=storage_dir, images_dir=images_dir, max_file_size=1024 * 1024)


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert sanitize_filename("clean_doc.pdf") == "clean_doc.pdf"
    assert sanitize_filename("   spaced name .txt  ") == "spaced name .txt"


def test_compute_content_hash():
    data1 = b"Hello DriveDocs"
    data2 = b"Hello DriveDocs"
    data3 = b"Different Content"
    assert compute_content_hash(data1) == compute_content_hash(data2)
    assert compute_content_hash(data1) != compute_content_hash(data3)


def test_ingest_text_and_markdown(temp_ingestion_mgr):
    txt_content = b"This is a sample document for retrieval testing."
    parsed_txt = temp_ingestion_mgr.ingest_bytes("sample.txt", txt_content)
    assert parsed_txt.metadata.filename == "sample.txt"
    assert parsed_txt.metadata.file_type == "txt"
    assert len(parsed_txt.chunks) == 1
    assert parsed_txt.chunks[0].text == "This is a sample document for retrieval testing."
    assert len(parsed_txt.images) == 0

    md_content = b"# Architecture\n\nDriveDocs uses dual FAISS indexes."
    parsed_md = temp_ingestion_mgr.ingest_bytes("arch.md", md_content)
    assert parsed_md.metadata.file_type == "md"
    assert len(parsed_md.chunks) >= 1


def test_ingest_standalone_image(temp_ingestion_mgr):
    img = Image.new("RGB", (120, 80), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    parsed = temp_ingestion_mgr.ingest_bytes("diagram.png", raw_bytes)
    assert parsed.metadata.file_type == "png"
    assert parsed.metadata.chunk_count == 0
    assert parsed.metadata.image_count == 1
    assert len(parsed.images) == 1
    assert parsed.images[0].width == 120
    assert parsed.images[0].height == 80
    assert Path(parsed.images[0].image_path).exists()


def test_duplicate_document_rejection(temp_ingestion_mgr):
    content = b"Exact duplicate testing bytes content 12345"
    temp_ingestion_mgr.ingest_bytes("first.txt", content)

    with pytest.raises(DuplicateDocumentError) as exc_info:
        temp_ingestion_mgr.ingest_bytes("second.txt", content, allow_duplicates=False)

    assert "already exists" in str(exc_info.value)
    assert exc_info.value.existing_doc.filename == "first.txt"


def test_unsupported_file_type(temp_ingestion_mgr):
    with pytest.raises(UnsupportedFileTypeError):
        temp_ingestion_mgr.ingest_bytes("malicious.exe", b"binary content")


def test_empty_file_validation(temp_ingestion_mgr):
    with pytest.raises(MalformedDocumentError):
        temp_ingestion_mgr.ingest_bytes("empty.txt", b"")


def test_file_size_limit_exceeded(temp_ingestion_mgr):
    large_content = b"A" * (1024 * 1024 + 50)
    with pytest.raises(FileSizeLimitExceededError):
        temp_ingestion_mgr.ingest_bytes("large.txt", large_content)


def test_ocr_support_check():
    avail, msg = check_ocr_support()
    assert isinstance(avail, bool)
    assert isinstance(msg, str)
    assert len(msg) > 10
