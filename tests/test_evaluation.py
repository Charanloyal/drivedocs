"""Evaluation metric definitions, checked with stub retrieval/generation output."""

from __future__ import annotations

from drivedocs.evaluation import DriveDocsEvaluator, EvaluationQuery
from drivedocs.generation import GroundedMultimodalGenerator
from drivedocs.schemas import ChatResponse, Citation, MultimodalEvidence, TextChunk, TextSearchResult


def _result(filename, page, rank):
    chunk = TextChunk(chunk_id=f"doc_aaaaaaaaaaaa_p{page}_c{rank}", doc_id="doc_aaaaaaaaaaaa", filename=filename,
                      page_number=page, chunk_index=rank, text="text", char_count=4)
    return TextSearchResult(chunk=chunk, score=0.5, rank=rank)


class StubRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k_text=5, top_k_images=3):
        return MultimodalEvidence(query=query, text_results=self.results, has_sufficient_evidence=True,
                                  top_confidence=0.5, top_text_score=0.5)


class StubGenerator:
    provider = "extractive"
    llm_configured = False

    def __init__(self, response):
        self.response = response

    def generate(self, query, evidence, include_debug=False):
        return self.response


def _evaluate(results, response, **query):
    item = EvaluationQuery(id="q", query="question", modality="text", **query)
    return DriveDocsEvaluator(StubRetriever(results), StubGenerator(response)).evaluate_query(item)


def test_recall_counts_relevant_files_not_retrieved_pages():
    results = [_result("a.pdf", 1, 1), _result("a.pdf", 2, 2)]
    answer = ChatResponse(answer="x [Source: a.pdf, Page 1]", citations=[Citation(filename="a.pdf", page_number=1)])
    res = _evaluate(results, answer, relevant_filenames=["a.pdf", "b.pdf"])
    assert res.recall_at_k == 0.5  # b.pdf was never retrieved
    assert res.reciprocal_rank == 1.0


def test_page_level_hits_and_citation_precision():
    results = [_result("a.pdf", 1, 1), _result("a.pdf", 2, 2)]
    answer = ChatResponse(answer="98.4% [Source: a.pdf, Page 1]", citations=[Citation(filename="a.pdf", page_number=1)])
    res = _evaluate(results, answer, relevant_filenames=["a.pdf"], relevant_pages=[2], expected_facts=["98.4%"])
    assert res.reciprocal_rank == 0.5
    assert res.precision_at_k == 0.5
    assert res.citation_precision == 0.0  # cited the wrong page
    assert res.fact_coverage == 1.0
    assert res.passed is False


def test_negative_query_requires_clean_refusal():
    refusal = ChatResponse(answer=GroundedMultimodalGenerator.INSUFFICIENT_EVIDENCE_MSG)
    item = EvaluationQuery(id="n", query="q", is_unsupported=True)
    res = DriveDocsEvaluator(StubRetriever([]), StubGenerator(refusal)).evaluate_query(item)
    assert res.negative_handled is True and res.passed is True

    answered = ChatResponse(answer="Something [Source: a.pdf, Page 1]", citations=[Citation(filename="a.pdf")])
    res = DriveDocsEvaluator(StubRetriever([]), StubGenerator(answered)).evaluate_query(item)
    assert res.negative_handled is False
