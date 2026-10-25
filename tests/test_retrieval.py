"""Unit tests for MultimodalRetriever orchestrator."""

import pytest
from drivedocs.ingestion import ParsedDocument
from drivedocs.retrieval import MultimodalRetriever
from drivedocs.schemas import DocumentMetadata, ImageItem, TextChunk
from drivedocs.text_index import TextFAISSIndex
from drivedocs.image_index import ImageFAISSIndex


@pytest.fixture
def multimodal_retriever():
    return MultimodalRetriever(min_similarity_threshold=0.25)


def test_retriever_unified_evidence(multimodal_retriever, tmp_path):
    img_path = tmp_path / "diagram.png"
    from PIL import Image
    Image.new("RGB", (60, 60), color=(50, 100, 150)).save(img_path)

    doc = ParsedDocument(
        metadata=DocumentMetadata(
            doc_id="doc_m1",
            filename="photonics.pdf",
            file_type="pdf",
            file_size_bytes=1000,
            content_hash="abc12345",
            page_count=1,
            chunk_count=1,
            image_count=1,
        ),
        chunks=[
            TextChunk(
                chunk_id="c_m1",
                doc_id="doc_m1",
                filename="photonics.pdf",
                page_number=1,
                chunk_index=0,
                text="The silicon nitride waveguide achieves 0.12 dB/cm optical attenuation.",
                char_count=71,
            )
        ],
        images=[
            ImageItem(
                image_id="img_m1",
                doc_id="doc_m1",
                filename="photonics.pdf",
                page_number=1,
                image_path=str(img_path),
                url_path="/static/images/diagram.png",
                width=60,
                height=60,
                format="PNG",
            )
        ],
    )

    multimodal_retriever.index_parsed_document(doc)

    evidence = multimodal_retriever.retrieve("waveguide optical attenuation", top_k_text=2, top_k_images=1)
    assert evidence.has_sufficient_evidence is True
    assert len(evidence.text_results) == 1
    assert evidence.text_results[0].chunk.filename == "photonics.pdf"
    assert len(evidence.image_results) == 1
    assert evidence.image_results[0].image.image_id == "img_m1"
    assert evidence.top_confidence >= 0.25


def test_retriever_insufficient_evidence(multimodal_retriever):
    # Empty retriever
    evidence = multimodal_retriever.retrieve("What is quantum entanglement?")
    assert evidence.has_sufficient_evidence is False
    assert len(evidence.text_results) == 0
    assert len(evidence.image_results) == 0
