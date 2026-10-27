"""Multimodal retrieval over two separate vector spaces.

* Text questions are embedded with MiniLM and searched in the text index.
* The same question is embedded with the CLIP text tower and searched in the image index.
* Each modality is judged against its own relevance rule; raw scores from the two spaces are never compared.
* A hybrid evidence list is produced with weighted Reciprocal Rank Fusion (rank-based, so it is scale-free).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from drivedocs.config import settings
from drivedocs.ingestion import ParsedDocument
from drivedocs.schemas import FusedEvidence, ImageSearchResult, MultimodalEvidence, TextSearchResult
from drivedocs.text_index import TextFAISSIndex
from drivedocs.image_index import ImageFAISSIndex

VISUAL_INTENT = re.compile(
    r"\b(diagrams?|charts?|figures?|fig\.|images?|pictures?|photos?|graphs?|plots?|illustrations?|schematics?|"
    r"drawings?|screenshots?|visuals?|visuali[sz]ations?|curves?|flowcharts?)\b",
    re.IGNORECASE,
)


def has_visual_intent(query: str) -> bool:
    return bool(VISUAL_INTENT.search(query))


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


class MultimodalRetriever:
    """Coordinates the text and image indexes and decides which evidence is relevant enough to use."""

    def __init__(
        self,
        text_index: Optional[TextFAISSIndex] = None,
        image_index: Optional[ImageFAISSIndex] = None,
        min_similarity_threshold: Optional[float] = None,
        image_min_score: Optional[float] = None,
        image_min_score_visual_query: Optional[float] = None,
        image_min_score_colocated: Optional[float] = None,
        text_context_min_score: Optional[float] = None,
        text_weight: Optional[float] = None,
        image_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
    ) -> None:
        self.text_index = text_index or TextFAISSIndex()
        self.image_index = image_index or ImageFAISSIndex()
        # `min_similarity_threshold` is the text threshold (kept under its original name for compatibility).
        self.min_similarity_threshold = (
            min_similarity_threshold if min_similarity_threshold is not None else settings.text_min_score
        )
        self.text_context_min_score = (
            text_context_min_score if text_context_min_score is not None else settings.text_context_min_score
        )
        self.image_min_score = image_min_score if image_min_score is not None else settings.image_min_score
        self.image_min_score_visual_query = (
            image_min_score_visual_query
            if image_min_score_visual_query is not None
            else settings.image_min_score_visual_query
        )
        self.image_min_score_colocated = (
            image_min_score_colocated if image_min_score_colocated is not None else settings.image_min_score_colocated
        )
        self.text_weight = text_weight if text_weight is not None else settings.text_weight
        self.image_weight = image_weight if image_weight is not None else settings.image_weight
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    @property
    def text_threshold(self) -> float:
        return self.min_similarity_threshold

    def index_parsed_document(self, parsed_doc: ParsedDocument) -> None:
        """Embed first, then add to both indexes, so a failure leaves both indexes untouched."""
        if parsed_doc.duplicate:
            return
        text_vectors = self.text_index.encode_chunks(parsed_doc.chunks) if parsed_doc.chunks else None
        image_vectors = self.image_index.encode_items(parsed_doc.images) if parsed_doc.images else None
        if text_vectors is not None:
            self.text_index.add_embedded(parsed_doc.chunks, text_vectors)
        if image_vectors is not None:
            self.image_index.add_embedded(parsed_doc.images, image_vectors)

    def remove_document(self, doc_id: str) -> Tuple[int, int]:
        removed_chunks = self.text_index.remove_doc(doc_id)
        removed_images = self.image_index.remove_doc(doc_id)
        return removed_chunks, len(removed_images)

    # Relevance rules -------------------------------------------------------------------------------------------

    def accept_image(
        self, result: ImageSearchResult, visual_intent: bool, text_pages: Set[Tuple[str, int]]
    ) -> Optional[str]:
        """Return the rule that accepts this image, or None. Rules are evaluated in order of strictness."""
        score = result.score
        if score >= self.image_min_score:
            return "strong_clip_match"
        if visual_intent and score >= self.image_min_score_visual_query:
            return "visual_query_match"
        page = result.image.page_number
        if page is not None and (result.image.doc_id, page) in text_pages and score >= self.image_min_score_colocated:
            return "same_page_as_text_evidence"
        return None

    def dedupe_text(self, results: List[TextSearchResult]) -> List[TextSearchResult]:
        seen: Set[str] = set()
        unique: List[TextSearchResult] = []
        for res in results:
            key = _normalize_for_dedupe(res.chunk.text)
            if key in seen:
                continue
            seen.add(key)
            unique.append(res.model_copy(update={"rank": len(unique) + 1}))
        return unique

    # Retrieval -------------------------------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_text: Optional[int] = None,
        top_k_images: Optional[int] = None,
    ) -> MultimodalEvidence:
        k_text = top_k_text if top_k_text is not None else settings.default_top_k_text
        k_img = top_k_images if top_k_images is not None else settings.default_top_k_images

        # Over-fetch slightly so de-duplication can still return k results.
        text_matches = self.dedupe_text(self.text_index.search(query, top_k=k_text + 2))[:k_text]
        # Two-level rule: the strict gate decides whether there is evidence at all; once it passes, passages above
        # the (lower) context floor may also contribute sentences to the answer.
        gate_passed = any(r.score >= self.min_similarity_threshold for r in text_matches)
        for res in text_matches:
            res.accepted = gate_passed and res.score >= min(self.text_context_min_score, self.min_similarity_threshold)

        visual_intent = has_visual_intent(query)
        image_matches: List[ImageSearchResult] = []
        if k_img > 0 and self.image_index.count() > 0:
            image_matches = self.image_index.search_by_text(query, top_k=k_img)

        accepted_text = [r for r in text_matches if r.accepted]
        # Co-location only counts passages that passed the strict gate, not the wider context pool.
        text_pages = {
            (r.chunk.doc_id, r.chunk.page_number) for r in text_matches if r.score >= self.min_similarity_threshold
        }
        accepted_image_ids: Set[str] = set()
        for res in image_matches:
            reason = self.accept_image(res, visual_intent, text_pages)
            res.match_reason = reason
            res.accepted = reason is not None
            if reason:
                accepted_image_ids.add(res.image.image_id)

        fused = self.fuse(text_matches, image_matches, accepted_image_ids)

        top_text = text_matches[0].score if text_matches else 0.0
        top_image = image_matches[0].score if image_matches else 0.0
        accepted_image_scores = [r.score for r in image_matches if r.image.image_id in accepted_image_ids]
        has_evidence = bool(accepted_text) or bool(accepted_image_scores)
        top_confidence = top_text if accepted_text or not accepted_image_scores else max(accepted_image_scores)

        return MultimodalEvidence(
            query=query,
            text_results=text_matches,
            image_results=image_matches,
            fused_results=fused,
            has_sufficient_evidence=has_evidence,
            top_confidence=round(float(top_confidence), 4),
            top_text_score=round(float(top_text), 4),
            top_image_score=round(float(top_image), 4),
            visual_intent=visual_intent,
        )

    def fuse(
        self,
        text_results: List[TextSearchResult],
        image_results: List[ImageSearchResult],
        accepted_image_ids: Set[str],
    ) -> List[FusedEvidence]:
        """Weighted Reciprocal Rank Fusion: fused = weight_modality / (rrf_k + rank_within_modality)."""
        fused: List[FusedEvidence] = []
        for res in text_results:
            fused.append(
                FusedEvidence(
                    modality="text",
                    item_id=res.chunk.chunk_id,
                    doc_id=res.chunk.doc_id,
                    filename=res.chunk.filename,
                    page_number=res.chunk.page_number,
                    modality_rank=res.rank,
                    raw_score=round(res.score, 4),
                    fused_score=round(self.text_weight / (self.rrf_k + res.rank), 6),
                    accepted=res.accepted,
                )
            )
        for res in image_results:
            fused.append(
                FusedEvidence(
                    modality="image",
                    item_id=res.image.image_id,
                    doc_id=res.image.doc_id,
                    filename=res.image.filename,
                    page_number=res.image.page_number,
                    modality_rank=res.rank,
                    raw_score=round(res.score, 4),
                    fused_score=round(self.image_weight / (self.rrf_k + res.rank), 6),
                    accepted=res.image.image_id in accepted_image_ids,
                )
            )
        # Accepted evidence first, then by fused score; ties keep text before image (stable sort).
        fused.sort(key=lambda f: (not f.accepted, -f.fused_score))
        return fused

    # Persistence -----------------------------------------------------------------------------------------------

    def save(self, directory: Optional[Path | str] = None) -> None:
        target_dir = Path(directory or settings.indexes_dir)
        self.text_index.save(target_dir)
        self.image_index.save(target_dir)

    def load(self, directory: Optional[Path | str] = None) -> bool:
        target_dir = Path(directory or settings.indexes_dir)
        loaded_text = self.text_index.load(target_dir)
        loaded_image = self.image_index.load(target_dir)
        return loaded_text or loaded_image
