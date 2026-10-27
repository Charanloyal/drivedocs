"""Unit tests for TextFAISSIndex and semantic text retrieval (real MiniLM weights)."""

import os
import numpy as np
import pytest
from drivedocs.schemas import TextChunk
from drivedocs.text_index import TextEmbeddingModel, TextFAISSIndex

pytestmark = pytest.mark.real_models


def real_index() -> TextFAISSIndex:
    return TextFAISSIndex(embedding_model=TextEmbeddingModel(backend=os.environ["DRIVEDOCS_TEST_REAL_BACKEND"]))


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id="c1",
            doc_id="d1",
            filename="photonics.pdf",
            page_number=1,
            chunk_index=0,
            text="Silicon nitride waveguides have an optical loss of 0.12 dB/cm.",
            char_count=60,
        ),
        TextChunk(
            chunk_id="c2",
            doc_id="d1",
            filename="photonics.pdf",
            page_number=2,
            chunk_index=1,
            text="Superconducting nanowire single photon detectors operate at 0.85 Kelvin.",
            char_count=72,
        ),
        TextChunk(
            chunk_id="c3",
            doc_id="d2",
            filename="raft.txt",
            page_number=1,
            chunk_index=0,
            text="Raft cluster leaders periodically send heartbeats every 150 milliseconds.",
            char_count=73,
        ),
    ]


def test_text_index_insertion_and_search(sample_chunks):
    # Using real SentenceTransformer model (already cached in virtual environment)
    index = real_index()
    assert index.count() == 0

    added = index.add_chunks(sample_chunks)
    assert added == 3
    assert index.count() == 3

    # Search for photonics
    results = index.search("waveguide transmission loss", top_k=2)
    assert len(results) == 2
    top = results[0]
    assert top.rank == 1
    assert "waveguide" in top.chunk.text.lower()
    assert top.chunk.filename == "photonics.pdf"
    assert top.chunk.page_number == 1
    assert 0.0 <= top.score <= 1.0


def test_text_index_persistence(sample_chunks, tmp_path):
    index = real_index()
    index.add_chunks(sample_chunks)

    save_dir = tmp_path / "faiss_text"
    index.save(save_dir)

    assert (save_dir / "text_index.faiss").exists()
    assert (save_dir / "text_metadata.json").exists()

    new_index = real_index()
    loaded = new_index.load(save_dir)
    assert loaded is True
    assert new_index.count() == 3

    results = new_index.search("heartbeats leader election", top_k=1)
    assert len(results) == 1
    assert "heartbeat" in results[0].chunk.text.lower()


def test_text_index_empty_query(sample_chunks):
    index = real_index()
    index.add_chunks(sample_chunks)
    assert index.search("", top_k=3) == []
    assert index.search("   ", top_k=3) == []
