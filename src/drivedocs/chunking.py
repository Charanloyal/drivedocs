"""Text chunking module with metadata preservation and page-boundary alignment."""

from __future__ import annotations

import re
from typing import List, Optional
from drivedocs.schemas import TextChunk


class TextChunker:
    """Splits text content into semantic chunks while strictly preserving source metadata."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_page_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        page_number: int,
        start_chunk_idx: int = 0,
    ) -> List[TextChunk]:
        """Split text belonging to a specific page into TextChunk instances."""
        cleaned = text.strip()
        if not cleaned:
            return []

        # Try using LangChain RecursiveCharacterTextSplitter if available
        raw_splits = self._split_text(cleaned)

        chunks: List[TextChunk] = []
        for i, split_text in enumerate(raw_splits):
            chunk_text = split_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{doc_id}_p{page_number}_c{start_chunk_idx + i}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=start_chunk_idx + i,
                    text=chunk_text,
                    char_count=len(chunk_text),
                )
            )

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursive character splitting using LangChain or native deterministic fallback."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
            )
            return splitter.split_text(text)
        except Exception:
            return self._native_recursive_split(text, self.separators)

    def _native_recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Native recursive character text splitter."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Base case: hard split by chunk_size with overlap
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += max(1, self.chunk_size - self.chunk_overlap)
            return chunks

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            parts = list(text)
        elif sep in text:
            parts = text.split(sep)
        else:
            return self._native_recursive_split(text, remaining_seps)

        final_chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for part in parts:
            part_len = len(part) + (len(sep) if current_chunk else 0)
            if current_length + part_len <= self.chunk_size:
                current_chunk.append(part)
                current_length += part_len
            else:
                if current_chunk:
                    merged = sep.join(current_chunk)
                    if len(merged) > self.chunk_size:
                        final_chunks.extend(self._native_recursive_split(merged, remaining_seps))
                    else:
                        final_chunks.append(merged)
                current_chunk = [part]
                current_length = len(part)

        if current_chunk:
            merged = sep.join(current_chunk)
            if len(merged) > self.chunk_size:
                final_chunks.extend(self._native_recursive_split(merged, remaining_seps))
            else:
                final_chunks.append(merged)

        return final_chunks
