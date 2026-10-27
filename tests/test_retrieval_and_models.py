"""Retrieval rules, hybrid fusion, chunk overlap, FAISS maintenance and model management (mock backend)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from drivedocs.chunking import TextChunker, reflow_lines
from drivedocs.image_index import ImageFAISSIndex
from drivedocs.models import ModelLoadError, ModelRegistry, mock_text_embedding
from drivedocs.retrieval import MultimodalRetriever, has_visual_intent
from drivedocs.schemas import ImageItem, ImageSearchResult, TextChunk, TextSearchResult
from drivedocs.text_index import TextFAISSIndex


def _chunk(text, doc="doc_aaaaaaaaaaaa", page=1, idx=0):
    return TextChunk(chunk_id=f"{doc}_p{page}_c{idx}", doc_id=doc, filename=f"{doc}.txt", page_number=page,
                     chunk_index=idx, text=text, char_count=len(text))


def _image_result(score, doc="doc_bbbbbbbbbbbb", page=2):
    item = ImageItem(image_id=f"img_{doc}_p{page}_0", doc_id=doc, filename="spec.pdf", page_number=page,
                     image_path="x.png", url_path="/static/images/x.png")
    return ImageSearchResult(image=item, score=score, rank=1)


def test_chunk_overlap_is_carried_between_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    text = " ".join(f"word{i}" for i in range(80))
    chunks = chunker.split_text(text)
    assert len(chunks) > 2
    assert all(len(c) <= 100 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()  # the tail of one chunk starts the next


def test_reflow_joins_wrapped_pdf_lines_but_keeps_headings():
    text = "Short Title\nThis is a long wrapped line of body text that continues on the\nnext line and ends here.\n- item"
    out = reflow_lines(text)
    assert "continues on the next line" in out
    assert out.startswith("Short Title\n")
    assert out.endswith("\n- item")


def test_visual_intent_detection():
    assert has_visual_intent("Show the architecture diagram")
    assert has_visual_intent("efficiency curve versus wavelength")
    assert not has_visual_intent("What is the heartbeat interval?")


@pytest.mark.parametrize(
    "score,visual,pages,expected",
    [
        (0.31, False, set(), "strong_clip_match"),
        (0.27, True, set(), "visual_query_match"),
        (0.27, False, {("doc_bbbbbbbbbbbb", 2)}, "same_page_as_text_evidence"),
        (0.27, False, {("doc_bbbbbbbbbbbb", 1)}, None),
        (0.20, True, {("doc_bbbbbbbbbbbb", 2)}, None),
    ],
)
def test_image_acceptance_rules(score, visual, pages, expected):
    retriever = MultimodalRetriever(text_index=TextFAISSIndex(), image_index=ImageFAISSIndex(enabled=False))
    assert retriever.accept_image(_image_result(score), visual, pages) == expected


def test_rrf_fusion_is_rank_based_and_never_compares_raw_scores():
    retriever = MultimodalRetriever(text_index=TextFAISSIndex(), image_index=ImageFAISSIndex(enabled=False),
                                    text_weight=0.7, image_weight=0.3, rrf_k=60)
    text = [TextSearchResult(chunk=_chunk("a"), score=0.31, rank=1, accepted=True)]
    image = [_image_result(0.99)]
    fused = retriever.fuse(text, image, accepted_image_ids={image[0].image.image_id})
    # A CLIP score of 0.99 does not outrank the first text hit: fusion uses ranks and weights only.
    assert [f.modality for f in fused] == ["text", "image"]
    assert fused[0].fused_score == pytest.approx(0.7 / 61, abs=1e-6)
    assert fused[1].fused_score == pytest.approx(0.3 / 61, abs=1e-6)
    assert fused[1].raw_score == 0.99


def test_two_level_text_threshold_and_dedupe():
    index = TextFAISSIndex()
    index.add_chunks([
        _chunk("raft leader heartbeat interval is 150 milliseconds", idx=0),
        _chunk("raft leader heartbeat interval is 150 milliseconds", doc="doc_cccccccccccc", idx=0),
        _chunk("completely unrelated banana bread recipe", doc="doc_dddddddddddd", idx=0),
    ])
    retriever = MultimodalRetriever(text_index=index, image_index=ImageFAISSIndex(enabled=False),
                                    min_similarity_threshold=0.3, text_context_min_score=0.2)
    ev = retriever.retrieve("raft heartbeat interval", top_k_text=3, top_k_images=0)
    texts = [r.chunk.text for r in ev.text_results]
    assert len(texts) == len(set(texts))  # identical passages appear once
    assert ev.has_sufficient_evidence
    assert ev.text_results[0].accepted
    banana = [r for r in ev.text_results if "banana" in r.chunk.text]
    assert not banana or not banana[0].accepted

    empty = retriever.retrieve("quantum chromodynamics lattice", top_k_text=3, top_k_images=0)
    assert not empty.has_sufficient_evidence
    assert not any(r.accepted for r in empty.text_results)


def test_faiss_remove_doc_keeps_other_vectors():
    index = TextFAISSIndex()
    a = _chunk("alpha photonics waveguide", doc="doc_aaaaaaaaaaaa")
    b = _chunk("beta consensus heartbeat", doc="doc_bbbbbbbbbbbb")
    index.add_chunks([a, b])
    assert index.remove_doc("doc_aaaaaaaaaaaa") == 1
    assert index.count() == 1 and index.index.ntotal == 1
    assert index.search("consensus heartbeat", 1)[0].chunk.doc_id == "doc_bbbbbbbbbbbb"


def test_index_load_rejects_mismatched_files(tmp_path):
    index = TextFAISSIndex()
    index.add_chunks([_chunk("one"), _chunk("two", idx=1)])
    index.save(tmp_path)
    (tmp_path / "text_metadata.json").write_text("[]")
    assert TextFAISSIndex().load(tmp_path) is False


def test_image_index_persists_portable_relative_paths(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    path = images_dir / "img_doc_aaaaaaaaaaaa.png"
    Image.new("RGB", (50, 50), (5, 5, 5)).save(path)
    item = ImageItem(image_id="img_doc_aaaaaaaaaaaa", doc_id="doc_aaaaaaaaaaaa", filename="a.png", page_number=1,
                     image_path=str(path), url_path="/static/images/img_doc_aaaaaaaaaaaa.png")
    index = ImageFAISSIndex(images_dir=images_dir, enabled=True)
    index.add_images([item])
    index.save(tmp_path / "idx")
    saved = (tmp_path / "idx" / "image_metadata.json").read_text()
    assert str(tmp_path) not in saved and "img_doc_aaaaaaaaaaaa.png" in saved
    loaded = ImageFAISSIndex(images_dir=images_dir, enabled=True)
    assert loaded.load(tmp_path / "idx") and loaded.resolve_path(loaded.images[0]) == path


def test_disabled_image_search_returns_nothing(tmp_path):
    index = ImageFAISSIndex(enabled=False)
    assert index.search_by_text("diagram") == []


def test_mock_text_embeddings_are_deterministic_and_normalized():
    a = mock_text_embedding(["silicon waveguide loss"], 384)
    b = mock_text_embedding(["silicon waveguide loss"], 384)
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)


def test_model_registry_loads_once_under_concurrency():
    registry = ModelRegistry(max_concurrent_inference=1)
    calls = []

    def loader():
        calls.append(1)
        time.sleep(0.05)
        return ("model",)

    threads = [threading.Thread(target=registry.get, args=("m|cpu|text", loader)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    state = registry.state("m|cpu|text")
    assert state.loaded and state.load_seconds is not None


def test_model_registry_records_load_errors():
    registry = ModelRegistry()

    def broken():
        raise OSError("weights missing")

    with pytest.raises(ModelLoadError):
        registry.get("bad|cpu|text", broken)
    assert registry.state("bad|cpu|text").error.startswith("OSError")
    assert not registry.is_loaded("bad|cpu|text")


def test_device_never_auto_selects_gpu():
    from drivedocs.config import Settings

    assert Settings._resolve_device("cpu") == "cpu"
    assert Settings._resolve_device("auto") == "cpu"
