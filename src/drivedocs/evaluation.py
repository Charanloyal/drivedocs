"""Evaluation harness for retrieval, citations, answer quality and negative-query handling.

Metric definitions (all computed from actual system output; nothing is hard-coded):

* Text Recall@K / Precision@K / MRR: over the top-K *text* results, a hit is a chunk whose (filename, page) is in
  the query's relevant (filename, page) pairs. Evaluated on text queries only.
* Image Recall@K / MRR: same, over the top-K *image* results. Evaluated on visual queries only.
* Hybrid document MRR: first relevant item in the evidence list ordered by weighted Reciprocal Rank Fusion, compared
  with the legacy ordering that sorted MiniLM and CLIP scores together (raw-score mixing).
* Citation precision (page-level): share of answer citations (text) or returned images (visual answers) whose
  (filename, page) is relevant. Computed only over answers that cite something.
* Citation verification rate: share of citation tags that passed the CitationVerifier.
* Cited-claim lexical support: mean content-word overlap between each cited claim and its passage.
* Fact coverage: share of the expected fact strings that occur in the answer (a recall proxy, not a proof of
  groundedness).
* Answer rate: share of answerable queries that were answered instead of refused.
* Negative handling: share of out-of-corpus queries refused (or, for hard negatives, explicitly flagged as not
  mentioned) without citations.
* Latency: wall-clock retrieve + generate time per query with warm models.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field

from drivedocs.config import settings
from drivedocs.generation import GroundedMultimodalGenerator
from drivedocs.retrieval import MultimodalRetriever
from drivedocs.schemas import ChatResponse, MultimodalEvidence

Pair = Tuple[str, Optional[int]]


class EvaluationQuery(BaseModel):
    """One benchmark query."""

    id: str
    query: str
    modality: Optional[Literal["text", "image", "none"]] = None
    relevant_doc_ids: List[str] = Field(default_factory=list)
    relevant_filenames: List[str] = Field(default_factory=list)
    relevant_pages: List[int] = Field(default_factory=list)
    relevant_image_ids: List[str] = Field(default_factory=list)
    expected_facts: List[str] = Field(default_factory=list)
    is_unsupported: bool = False
    hard_negative: bool = False
    notes: Optional[str] = None

    @property
    def resolved_modality(self) -> str:
        if self.modality:
            return self.modality
        if self.is_unsupported:
            return "none"
        return "image" if self.relevant_image_ids else "text"

    def relevant_pairs(self) -> Set[Pair]:
        pages = self.relevant_pages or [None]  # type: ignore[list-item]
        return {(f, p) for f in self.relevant_filenames for p in pages}


class QueryEvaluationResult(BaseModel):
    query_id: str
    query: str
    modality: str
    is_unsupported: bool
    hard_negative: bool
    top_text_score: float
    top_image_score: float
    retrieved_text: List[str]
    retrieved_images: List[str]
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    reciprocal_rank: Optional[float] = None
    hybrid_rr: Optional[float] = None
    raw_mix_rr: Optional[float] = None
    answered: bool
    answer_type: str
    citation_precision: Optional[float] = None
    citations_total: int
    citations_verified: int
    citation_support: Optional[float] = None
    fact_coverage: Optional[float] = None
    negative_handled: Optional[bool] = None
    latency_ms: float
    generated_answer: str
    passed: bool


class BenchmarkReport(BaseModel):
    timestamp: str
    environment: Dict[str, Any]
    dataset: Dict[str, Any]
    summary: Dict[str, Optional[float]]
    comparison: Dict[str, Optional[float]]
    query_results: List[QueryEvaluationResult]


def _pair_hit(pair: Pair, relevant: Set[Pair]) -> bool:
    filename, page = pair
    return (filename, page) in relevant or (filename, None) in relevant


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


class DriveDocsEvaluator:
    """Runs benchmark queries through the real retriever and generator and scores the output."""

    def __init__(
        self,
        retriever: MultimodalRetriever,
        generator: Optional[GroundedMultimodalGenerator] = None,
        k: int = 5,
        k_images: int = 3,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or GroundedMultimodalGenerator(provider="extractive")
        self.k = k
        self.k_images = k_images

    def _run(self, query: str) -> Tuple[MultimodalEvidence, ChatResponse, float]:
        t0 = time.perf_counter()
        evidence = self.retriever.retrieve(query, top_k_text=self.k, top_k_images=self.k_images)
        response = self.generator.generate(query, evidence, include_debug=False)
        return evidence, response, round((time.perf_counter() - t0) * 1000.0, 2)

    def evaluate_query(self, item: EvaluationQuery) -> QueryEvaluationResult:
        evidence, resp, latency = self._run(item.query)
        relevant = item.relevant_pairs()
        modality = item.resolved_modality

        text_pairs: List[Pair] = [(r.chunk.filename, r.chunk.page_number) for r in evidence.text_results[: self.k]]
        image_pairs: List[Pair] = [(r.image.filename, r.image.page_number) for r in evidence.image_results]
        refused = GroundedMultimodalGenerator.INSUFFICIENT_EVIDENCE_MSG in resp.answer
        flagged = resp.answer.startswith(GroundedMultimodalGenerator.NOT_MENTIONED_PREFIX)
        answered = not refused

        result: Dict[str, Any] = dict(
            query_id=item.id,
            query=item.query,
            modality=modality,
            is_unsupported=item.is_unsupported,
            hard_negative=item.hard_negative,
            top_text_score=evidence.top_text_score,
            top_image_score=evidence.top_image_score,
            retrieved_text=[f"{f} p{p}" for f, p in text_pairs],
            retrieved_images=[f"{f} p{p}" for f, p in image_pairs],
            answered=answered,
            answer_type=resp.answer_type,
            citations_total=resp.citation_check.total,
            citations_verified=resp.citation_check.verified,
            latency_ms=latency,
            generated_answer=resp.answer,
        )
        supports = [c.support for c in resp.citations if c.support is not None]
        result["citation_support"] = _mean(supports)

        if item.is_unsupported:
            refused_cleanly = refused and not resp.citations
            # Hard negatives are near-domain; flagging the unknown term ("not mentioned") is also acceptable.
            handled = refused_cleanly or (item.hard_negative and flagged)
            result["negative_handled"] = handled
            result["passed"] = handled
            return QueryEvaluationResult(**result)

        ranked = text_pairs if modality == "text" else image_pairs
        k = self.k if modality == "text" else self.k_images
        top = ranked[:k]
        hits = [p for p in top if _pair_hit(p, relevant)]
        # Recall counts relevant files covered by the top-K (page-aware); relevant_pages apply to every listed file.
        covered = {f for f in item.relevant_filenames if any(_pair_hit(p, relevant) and p[0] == f for p in top)}
        result["recall_at_k"] = round(len(covered) / max(1, len(item.relevant_filenames)), 4)
        result["precision_at_k"] = round(len(hits) / len(top), 4) if top else 0.0
        result["reciprocal_rank"] = next(
            (1.0 / i for i, p in enumerate(ranked[:k], start=1) if _pair_hit(p, relevant)), 0.0
        )

        fused_pairs = [(f.filename, f.page_number) for f in evidence.fused_results]
        result["hybrid_rr"] = next((1.0 / i for i, p in enumerate(fused_pairs, 1) if _pair_hit(p, relevant)), 0.0)
        raw_mix = sorted(
            [(r.score, (r.chunk.filename, r.chunk.page_number)) for r in evidence.text_results]
            + [(r.score, (r.image.filename, r.image.page_number)) for r in evidence.image_results],
            key=lambda x: x[0],
            reverse=True,
        )
        result["raw_mix_rr"] = next((1.0 / i for i, (_, p) in enumerate(raw_mix, 1) if _pair_hit(p, relevant)), 0.0)

        cited: List[Pair] = [(c.filename, c.page_number) for c in resp.citations]
        if not cited and resp.images:
            cited = [(img.filename, img.page_number) for img in resp.images]
        if cited:
            result["citation_precision"] = round(sum(_pair_hit(p, relevant) for p in cited) / len(cited), 4)

        if item.expected_facts:
            answer_lower = resp.answer.lower()
            found = sum(1 for fact in item.expected_facts if fact.lower() in answer_lower)
            result["fact_coverage"] = round(found / len(item.expected_facts), 4) if answered else 0.0

        result["passed"] = bool(
            answered
            and (result["reciprocal_rank"] or 0) > 0
            and (result.get("citation_precision") or 0) >= 0.5
            and (result.get("fact_coverage") is None or result["fact_coverage"] >= 0.5)
        )
        return QueryEvaluationResult(**result)

    def run_benchmark(
        self, eval_set: List[EvaluationQuery], dataset_path: Optional[str] = None, warmup: bool = True
    ) -> BenchmarkReport:
        if not eval_set:
            raise ValueError("Evaluation set is empty.")
        if warmup:
            self._run(eval_set[0].query)  # exclude one-time model/kernel warm-up from latency figures

        results = [self.evaluate_query(item) for item in eval_set]
        text_q = [r for r in results if r.modality == "text" and not r.is_unsupported]
        image_q = [r for r in results if r.modality == "image" and not r.is_unsupported]
        answerable = text_q + image_q
        negatives = [r for r in results if r.is_unsupported and not r.hard_negative]
        hard = [r for r in results if r.is_unsupported and r.hard_negative]
        latencies = [r.latency_ms for r in results]
        cit_total = sum(r.citations_total for r in results)
        cit_verified = sum(r.citations_verified for r in results)

        def m(rows, attr):
            return _mean([getattr(r, attr) for r in rows if getattr(r, attr) is not None])

        summary = {
            "text_recall_at_k": m(text_q, "recall_at_k"),
            "text_precision_at_k": m(text_q, "precision_at_k"),
            "text_mrr": m(text_q, "reciprocal_rank"),
            "image_recall_at_k": m(image_q, "recall_at_k"),
            "image_mrr": m(image_q, "reciprocal_rank"),
            "hybrid_document_mrr": m(answerable, "hybrid_rr"),
            "citation_precision": m(answerable, "citation_precision"),
            "citation_verification_rate": round(cit_verified / cit_total, 4) if cit_total else None,
            "cited_claim_lexical_support": m(answerable, "citation_support"),
            "fact_coverage": m(answerable, "fact_coverage"),
            "answer_rate": _mean([1.0 if r.answered else 0.0 for r in answerable]),
            "negative_rejection_rate": _mean([1.0 if r.negative_handled else 0.0 for r in negatives]),
            "hard_negative_handling_rate": _mean([1.0 if r.negative_handled else 0.0 for r in hard]),
            "pass_rate": _mean([1.0 if r.passed else 0.0 for r in results]),
            "latency_mean_ms": round(statistics.mean(latencies), 2),
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
        }
        comparison = {
            "hybrid_rrf_document_mrr": m(answerable, "hybrid_rr"),
            "raw_score_mixing_document_mrr": m(answerable, "raw_mix_rr"),
        }
        environment = {
            "text_model": self.retriever.text_index.embedding_model.identity,
            "image_model": self.retriever.image_index.embedding_model.identity,
            "image_search_enabled": self.retriever.image_index.enabled,
            "provider": self.generator.provider if self.generator.llm_configured else "extractive",
            "k_text": self.k,
            "k_images": self.k_images,
            "thresholds": {
                "text_min_score": self.retriever.text_threshold,
                "text_context_min_score": self.retriever.text_context_min_score,
                "image_min_score": self.retriever.image_min_score,
                "image_min_score_visual_query": self.retriever.image_min_score_visual_query,
                "image_min_score_colocated": self.retriever.image_min_score_colocated,
            },
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "device": settings.device,
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
        }
        dataset = {
            "path": dataset_path,
            "size": len(results),
            "text_queries": len(text_q),
            "visual_queries": len(image_q),
            "negative_queries": len(negatives),
            "hard_negative_queries": len(hard),
        }
        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            environment=environment,
            dataset=dataset,
            summary=summary,
            comparison=comparison,
            query_results=results,
        )

    def write_reports(self, report: BenchmarkReport, json_output_path: Path | str, md_output_path: Path | str) -> None:
        json_path, md_path = Path(json_output_path), Path(md_output_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_markdown_report(report), encoding="utf-8")


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown_report(report: BenchmarkReport) -> str:
    s, d, e, c = report.summary, report.dataset, report.environment, report.comparison
    rows = []
    for r in report.query_results:
        kind = ("Hard negative" if r.hard_negative else "Negative") if r.is_unsupported else r.modality.title()
        q = r.query.replace("|", "/")
        rr = "—" if r.reciprocal_rank is None else f"{r.reciprocal_rank:.2f}"
        cp = "—" if r.citation_precision is None else f"{r.citation_precision * 100:.0f}%"
        fc = "—" if r.fact_coverage is None else f"{r.fact_coverage * 100:.0f}%"
        rows.append(
            f"| `{r.query_id}` | {q} | {kind} | {r.top_text_score:.2f} / {r.top_image_score:.2f} | {rr} | {cp} | "
            f"{fc} | {r.answer_type} | {r.latency_ms:.0f} | {'PASS' if r.passed else 'FAIL'} |"
        )
    failing = [r for r in report.query_results if not r.passed]
    failures = "\n".join(
        f"- `{r.query_id}` ({r.answer_type}): {r.generated_answer[:220].replace(chr(10), ' ')}" for r in failing
    ) or "- None"

    return f"""# DriveDocs Evaluation Report

Generated by `scripts/run_eval.py` on **{report.timestamp}**. All numbers below were computed from actual system output.

**Configuration:** text model `{e['text_model']}`, image model `{e['image_model']}`, provider `{e['provider']}`,
device `{e['device']}`, K(text)={e['k_text']}, K(images)={e['k_images']}, chunk size {e['chunk_size']} / overlap
{e['chunk_overlap']}, thresholds `{json.dumps(e['thresholds'])}`, Python {e['python']} on {e['platform']}.

**Dataset:** `{d['path']}`: {d['size']} queries ({d['text_queries']} text, {d['visual_queries']} visual,
{d['negative_queries']} negative, {d['hard_negative_queries']} hard negative) over the bundled sample corpus (4 files).

> The dataset is small and was written by the project authors together with the sample corpus. Treat these figures as
> a regression check for this demo corpus, not as a general benchmark of retrieval quality.

## Summary

| Metric | Value | Definition |
| :--- | :---: | :--- |
| Text Recall@{e['k_text']} | {_fmt_pct(s['text_recall_at_k'])} | Relevant (file, page) in the top-{e['k_text']} text results |
| Text Precision@{e['k_text']} | {_fmt_pct(s['text_precision_at_k'])} | Share of top-{e['k_text']} text results that are relevant (low by construction for single-page answers) |
| Text MRR | {_fmt_num(s['text_mrr'])} | Mean reciprocal rank of the first relevant text chunk |
| Image Recall@{e['k_images']} | {_fmt_pct(s['image_recall_at_k'])} | Relevant (file, page) in the top-{e['k_images']} CLIP image results |
| Image MRR | {_fmt_num(s['image_mrr'])} | Mean reciprocal rank of the first relevant image |
| Hybrid document MRR (RRF) | {_fmt_num(s['hybrid_document_mrr'])} | First relevant item in the fused evidence list |
| Citation precision (page-level) | {_fmt_pct(s['citation_precision'])} | Cited (file, page) pairs that are relevant |
| Citation verification rate | {_fmt_pct(s['citation_verification_rate'])} | Citation tags that passed the verifier |
| Cited-claim lexical support | {_fmt_pct(s['cited_claim_lexical_support'])} | Content-word overlap between cited claims and passages |
| Fact coverage | {_fmt_pct(s['fact_coverage'])} | Expected fact strings present in the answer |
| Answer rate | {_fmt_pct(s['answer_rate'])} | Answerable queries that were not refused |
| Negative rejection | {_fmt_pct(s['negative_rejection_rate'])} | Out-of-corpus queries refused without citations |
| Hard-negative handling | {_fmt_pct(s['hard_negative_handling_rate'])} | Near-domain unanswerable queries refused or flagged as not mentioned |
| Pass rate | {_fmt_pct(s['pass_rate'])} | Per-query pass criteria (see methodology) |
| Latency mean / p50 / p95 | {_fmt_num(s['latency_mean_ms'], 1)} / {_fmt_num(s['latency_p50_ms'], 1)} / {_fmt_num(s['latency_p95_ms'], 1)} ms | Retrieve + generate, warm models, this machine |

### Hybrid ranking vs. baseline

| Ordering of the combined text + image evidence | Document MRR |
| :--- | :---: |
| Weighted Reciprocal Rank Fusion (current) | {_fmt_num(c['hybrid_rrf_document_mrr'])} |
| Raw-score mixing of MiniLM and CLIP cosines (previous evaluation) | {_fmt_num(c['raw_score_mixing_document_mrr'])} |

## Per-query results

| ID | Question | Type | Top text / image score | RR | Citation precision | Fact coverage | Answer type | Latency (ms) | Result |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
{chr(10).join(rows)}

## Failing queries

{failures}

## Methodology

See [docs/EVALUATION_METHODOLOGY.md](../docs/EVALUATION_METHODOLOGY.md) for metric definitions, pass criteria and known
limitations.
"""
