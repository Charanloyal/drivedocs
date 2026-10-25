"""Multimodal hybrid retrieval orchestrator coordinating text and image vector indexes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from drivedocs.config import settings
from drivedocs.ingestion import ParsedDocument
from drivedocs.schemas import (
    ImageSearchResult,
    MultimodalEvidence,
    TextSearchResult,
)
from drivedocs.text_index import TextFAISSIndex
from drivedocs.image_index import ImageFAISSIndex


class MultimodalRetriever:
    """Orchestrates separate text and image FAISS vector stores to provide unified evidence."""

    def __init__(
        self,
        text_index: Optional[TextFAISSIndex] = None,
        image_index: Optional[ImageFAISSIndex] = None,
        min_similarity_threshold: Optional[float] = None,
    ) -> None:
        self.text_index = text_index or TextFAISSIndex()
        self.image_index = image_index or ImageFAISSIndex()
        self.min_similarity_threshold = (
            min_similarity_threshold
            if min_similarity_threshold is not None
            else settings.min_similarity_threshold
        )

    def index_parsed_document(self, parsed_doc: ParsedDocument) -> None:
        """Add all text chunks and extracted images from a parsed document to indexes."""
        if parsed_doc.chunks:
            self.text_index.add_chunks(parsed_doc.chunks)
        if parsed_doc.images:
            self.image_index.add_images(parsed_doc.images)

    def retrieve(
        self,
        query: str,
        top_k_text: Optional[int] = None,
        top_k_images: Optional[int] = None,
    ) -> MultimodalEvidence:
        """Execute multimodal retrieval: retrieve text evidence and visual evidence for query."""
        k_text = top_k_text if top_k_text is not None else settings.default_top_k_text
        k_img = top_k_images if top_k_images is not None else settings.default_top_k_images

        text_matches: List[TextSearchResult] = self.text_index.search(query, top_k=k_text)
        image_matches: List[ImageSearchResult] = []

        if k_img > 0 and self.image_index.count() > 0:
            image_matches = self.image_index.search_by_text(query, top_k=k_img)

        # Assess evidence sufficiency based on highest similarity score
        top_text_score = text_matches[0].score if text_matches else 0.0
        top_image_score = image_matches[0].score if image_matches else 0.0
        top_confidence = max(top_text_score, top_image_score)

        has_evidence = (
            top_confidence >= self.min_similarity_threshold
            and (len(text_matches) > 0 or len(image_matches) > 0)
        )

        return MultimodalEvidence(
            query=query,
            text_results=text_matches,
            image_results=image_matches,
            has_sufficient_evidence=has_evidence,
            top_confidence=round(float(top_confidence), 4),
        )

    def save(self, directory: Optional[Path | str] = None) -> None:
        """Persist both indexes to disk."""
        target_dir = Path(directory or settings.indexes_dir)
        self.text_index.save(target_dir)
        self.image_index.save(target_dir)

    def load(self, directory: Optional[Path | str] = None) -> bool:
        """Load both indexes from disk."""
        target_dir = Path(directory or settings.indexes_dir)
        loaded_text = self.text_index.load(target_dir)
        loaded_image = self.image_index.load(target_dir)
        return loaded_text or loaded_image
