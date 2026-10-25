"""Unit tests for GroundedMultimodalGenerator and citation attribution."""

import pytest
from drivedocs.generation import GroundedMultimodalGenerator
from drivedocs.schemas import (
    ImageItem,
    ImageSearchResult,
    MultimodalEvidence,
    TextChunk,
    TextSearchResult,
)


@pytest.fixture
def sample_evidence():
    chunk1 = TextChunk(
        chunk_id="chk1",
        doc_id="d1",
        filename="specs.pdf",
        page_number=2,
        chunk_index=0,
        text="The peak quantum efficiency is 98.4% at 1550 nm telecommunication wavelength.",
        char_count=77,
    )
    img1 = ImageItem(
        image_id="img1",
        doc_id="d1",
        filename="specs.pdf",
        page_number=2,
        image_path="/path/chart.png",
        url_path="/static/images/chart.png",
        width=400,
        height=200,
        format="PNG",
    )
    return MultimodalEvidence(
        query="What is the peak quantum efficiency?",
        text_results=[TextSearchResult(chunk=chunk1, score=0.65, rank=1)],
        image_results=[ImageSearchResult(image=img1, score=0.45, rank=1)],
        has_sufficient_evidence=True,
        top_confidence=0.65,
    )


def test_generation_grounded_answer_and_citations(sample_evidence):
    generator = GroundedMultimodalGenerator(provider="extractive")
    resp = generator.generate(
        query="What is the peak quantum efficiency?",
        evidence=sample_evidence,
        include_debug=True,
    )

    assert "98.4%" in resp.answer
    assert "[Source: specs.pdf, Page 2]" in resp.answer
    assert len(resp.citations) >= 1
    assert resp.citations[0].filename == "specs.pdf"
    assert resp.citations[0].page_number == 2
    assert len(resp.images) == 1
    assert resp.images[0].filename == "specs.pdf"
    assert resp.debug_info is not None
    assert resp.debug_info["provider"] == "extractive"


def test_generation_insufficient_evidence():
    generator = GroundedMultimodalGenerator(provider="extractive")
    empty_evidence = MultimodalEvidence(
        query="What is the recipe for cookies?",
        text_results=[],
        image_results=[],
        has_sufficient_evidence=False,
        top_confidence=0.15,
    )

    resp = generator.generate("What is the recipe for cookies?", empty_evidence)
    assert resp.answer == GroundedMultimodalGenerator.INSUFFICIENT_EVIDENCE_MSG
    assert len(resp.citations) == 0
    assert len(resp.images) == 0


def test_generation_missing_credentials_fallback(sample_evidence):
    # Setting provider to gemini with dummy/missing key should gracefully fallback to extractive
    generator = GroundedMultimodalGenerator(provider="gemini", gemini_api_key=None)
    resp = generator.generate(
        query="What is the peak quantum efficiency?",
        evidence=sample_evidence,
    )
    # Must succeed via extractive fallback without raising uncaught exceptions
    assert "98.4%" in resp.answer
    assert len(resp.citations) >= 1
    assert resp.citations[0].filename == "specs.pdf"
