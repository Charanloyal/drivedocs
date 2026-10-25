"""Unit tests for text chunking and page boundary preservation."""

import pytest
from drivedocs.chunking import TextChunker
from drivedocs.schemas import TextChunk


def test_chunker_basic_split():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "Silicon photonics integrates optical devices onto silicon substrates. "
        "Waveguides route photons with low transmission loss. "
        "Single-photon detectors operate at cryogenic temperatures."
    )
    chunks = chunker.chunk_page_text(
        text=text,
        doc_id="test_doc",
        filename="report.pdf",
        page_number=1,
    )

    assert len(chunks) >= 1
    for i, c in enumerate(chunks):
        assert isinstance(c, TextChunk)
        assert c.doc_id == "test_doc"
        assert c.filename == "report.pdf"
        assert c.page_number == 1
        assert c.chunk_index == i
        assert len(c.text) > 0
        assert c.char_count == len(c.text)


def test_chunker_empty_and_whitespace():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_page_text("", "d1", "f.txt", 1) == []
    assert chunker.chunk_page_text("   \n\n   \t  ", "d1", "f.txt", 1) == []


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap"):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_chunker_page_number_isolation():
    chunker = TextChunker(chunk_size=80, chunk_overlap=10)
    c1 = chunker.chunk_page_text("Page one content is here.", "d1", "doc.pdf", page_number=1)
    c2 = chunker.chunk_page_text("Page two content is here.", "d1", "doc.pdf", page_number=2)

    assert len(c1) == 1
    assert len(c2) == 1
    assert c1[0].page_number == 1
    assert c2[0].page_number == 2
    assert "p1" in c1[0].chunk_id
    assert "p2" in c2[0].chunk_id
