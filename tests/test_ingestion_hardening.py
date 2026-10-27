"""Ingestion hardening: content sniffing, malformed files, PDFs, images, filenames and the registry."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from drivedocs.ingestion import (
    DocumentIngestionManager,
    DuplicateDocumentError,
    MalformedDocumentError,
    NoExtractableContentError,
    StorageLimitError,
    UnsupportedFileTypeError,
    decode_text,
    sanitize_filename,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture
def mgr(tmp_path):
    return DocumentIngestionManager(storage_dir=tmp_path / "s", images_dir=tmp_path / "img", max_file_size=2_000_000)


def _png(size=(120, 90), color=(10, 120, 200), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _pdf(pages_text, image: bytes | None = None) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages_text:
        if text:
            c.drawString(72, 720, text)
        if image is not None:
            c.drawImage(ImageReader(io.BytesIO(image)), 72, 400, width=200, height=150)
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\calc.exe", "calc.exe"),
        ("report<script>.pdf", "report_script_.pdf"),
        ("a\x00b\x1f.txt", "ab.txt"),
    ],
)
def test_sanitize_filename_strips_paths_and_controls(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_reserved_and_long_names():
    assert sanitize_filename("CON.txt").startswith("doc_")
    long_name = "x" * 300 + ".pdf"
    cleaned = sanitize_filename(long_name)
    assert len(cleaned) <= 120 and cleaned.endswith(".pdf")


def test_extension_must_match_content(mgr):
    with pytest.raises(UnsupportedFileTypeError):
        mgr.ingest_bytes("fake.pdf", b"this is not a pdf at all")
    with pytest.raises(UnsupportedFileTypeError):
        mgr.ingest_bytes("fake.png", b"GIF89a....")


def test_binary_disguised_as_text_is_rejected(mgr):
    with pytest.raises(UnsupportedFileTypeError):
        mgr.ingest_bytes("blob.txt", b"\x00\x01\x02binary\x00data")
    with pytest.raises(UnsupportedFileTypeError):
        decode_text(bytes(range(1, 32)) * 10, "ctrl.txt")


def test_text_encodings(mgr):
    assert decode_text("héllo wörld".encode("utf-16"), "u16.txt") == "héllo wörld"
    assert decode_text("caf\xe9 menu".encode("cp1252"), "win.txt") == "café menu"
    assert decode_text(b"\xef\xbb\xbfBOM text", "bom.txt") == "BOM text"


def test_whitespace_only_text_is_rejected(mgr):
    with pytest.raises(NoExtractableContentError):
        mgr.ingest_bytes("blank.md", b"   \n\n\t  ")


def test_corrupted_pdf_is_rejected(mgr):
    good = (SAMPLE_DIR / "quantum_photonics_spec.pdf").read_bytes()
    corrupted = good[:200] + b"\x00garbage" * 50
    with pytest.raises(MalformedDocumentError):
        mgr.ingest_bytes("broken.pdf", corrupted)


def test_pdf_without_text_or_images_is_rejected(mgr):
    with pytest.raises(NoExtractableContentError):
        mgr.ingest_bytes("empty.pdf", _pdf([""]))


def test_pdf_with_only_images_is_accepted_with_warning(mgr):
    parsed = mgr.ingest_bytes("scan.pdf", _pdf([""], image=_png((300, 200))))
    assert parsed.metadata.chunk_count == 0
    assert parsed.metadata.image_count == 1
    assert any("No extractable text" in w for w in parsed.metadata.warnings)


def test_pdf_page_metadata_and_image_dedupe(mgr):
    img = _png((300, 200), (200, 30, 30))
    parsed = mgr.ingest_bytes("three.pdf", _pdf(["alpha page one", "beta page two", "gamma page three"], image=img))
    assert parsed.metadata.page_count == 3
    assert [c.page_number for c in parsed.chunks] == [1, 2, 3]
    assert "beta" in parsed.chunks[1].text
    # The same embedded image on every page is stored once.
    assert parsed.metadata.image_count == 1
    assert parsed.images[0].page_number == 1


def test_pdf_page_limit(mgr):
    mgr.max_pdf_pages = 2
    with pytest.raises(MalformedDocumentError, match="limit"):
        mgr.ingest_bytes("long.pdf", _pdf(["a", "b", "c"]))


def test_sample_pdf_extraction():
    m = DocumentIngestionManager(images_dir=Path(__import__("tempfile").mkdtemp()))
    parsed = m.ingest_file(SAMPLE_DIR / "quantum_photonics_spec.pdf")
    assert parsed.metadata.page_count == 2
    assert {c.page_number for c in parsed.chunks} == {1, 2}
    assert any("0.12 dB/cm" in c.text and c.page_number == 1 for c in parsed.chunks)
    assert any("98.4%" in c.text and c.page_number == 2 for c in parsed.chunks)
    assert parsed.images and parsed.images[0].page_number == 2


def test_markdown_heading_context(mgr):
    md = b"# Guide\n\n## Setup\n\nInstall the package first.\n\n```\n# not a heading\n```\n\n## Usage\n\nRun it."
    parsed = mgr.ingest_bytes("guide.md", md)
    sections = [c.section for c in parsed.chunks]
    assert "Guide > Setup" in sections and "Guide > Usage" in sections
    setup = next(c for c in parsed.chunks if c.section == "Guide > Setup")
    assert "# not a heading" in setup.text
    assert setup.embedding_text().startswith("Guide > Setup\n")


def test_image_normalization_transparency_and_size(mgr):
    mgr.max_stored_image_side = 256
    rgba = _png((1000, 500), (0, 0, 0, 0), mode="RGBA")
    parsed = mgr.ingest_bytes("transparent.png", rgba)
    stored = Image.open(parsed.images[0].image_path)
    assert stored.mode == "RGB"
    assert max(stored.size) == 256
    assert stored.getpixel((10, 10)) == (255, 255, 255)  # transparent → white, not black


def test_decompression_bomb_guard(mgr):
    mgr.max_image_pixels = 100 * 100
    with pytest.raises(MalformedDocumentError, match="maximum image size"):
        mgr.ingest_bytes("huge.png", _png((200, 200)))


def test_duplicate_allowed_returns_existing_without_reparse(mgr):
    first = mgr.ingest_bytes("one.txt", b"same content for both uploads")
    again = mgr.ingest_bytes("two.txt", b"same content for both uploads", allow_duplicates=True)
    assert again.duplicate is True
    assert again.metadata.doc_id == first.metadata.doc_id
    assert len(mgr.list_documents()) == 1
    with pytest.raises(DuplicateDocumentError):
        mgr.ingest_bytes("three.txt", b"same content for both uploads")


def test_document_limit(mgr):
    mgr.max_documents = 1
    mgr.ingest_bytes("a.txt", b"first document text")
    with pytest.raises(StorageLimitError):
        mgr.ingest_bytes("b.txt", b"second document text")


def test_failure_record_and_registry_round_trip(mgr):
    failed = mgr.record_failure("bad.pdf", b"%PDF-broken", "not readable")
    assert failed is not None and failed.status == "failed" and failed.error == "not readable"
    ok = mgr.ingest_bytes("ok.txt", b"registry round trip text")
    payload = mgr.registry_payload()

    other = DocumentIngestionManager(images_dir=mgr.images_dir)
    problems = other.load_registry_payload(payload, ok.chunks, [])
    assert problems == []
    assert other.get_document(ok.metadata.doc_id).status == "ready"
    assert other.get_document(failed.doc_id).status == "failed"
    # Failed documents do not block a later successful upload of the same content.
    assert failed.content_hash not in other.hash_to_doc_id


def test_registry_detects_inconsistency(mgr):
    ok = mgr.ingest_bytes("ok.txt", b"consistency check text")
    other = DocumentIngestionManager(images_dir=mgr.images_dir)
    problems = other.load_registry_payload(mgr.registry_payload(), [], [])
    assert problems and ok.metadata.doc_id in problems[0]
