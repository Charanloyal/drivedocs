"""Deterministic text chunking with real overlap, page metadata and Markdown heading context.

The splitter follows the well-known "recursive character" strategy: split on the coarsest separator that occurs
(paragraphs, then lines, then sentences, then words, then characters), then greedily merge pieces into chunks of at
most ``chunk_size`` characters, carrying up to ``chunk_overlap`` characters of trailing pieces into the next chunk.
It has no third-party dependency, so chunk boundaries do not change with library versions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from drivedocs.schemas import TextChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


def _list_items(body: str) -> List[str]:
    """The items of a section that consists only of a flat list (continuation lines stay with their item)."""
    lines = [line for line in body.strip().split("\n") if line.strip()]
    if not lines or not _LIST_ITEM.match(lines[0]):
        return []
    items: List[str] = []
    for line in lines:
        if _LIST_ITEM.match(line):
            items.append(line.strip())
        elif line.startswith((" ", "\t")) and items:
            items[-1] += " " + line.strip()
        else:
            return []  # prose mixed with the list: keep the section together
    return items


class TextChunker:
    """Splits text into chunks while preserving source metadata."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators else ["\n\n", "\n", ". ", " ", ""]
        if "" not in self.separators:
            self.separators.append("")

    # Public API ------------------------------------------------------------------------------------------------

    def chunk_page_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        page_number: int,
        start_chunk_idx: int = 0,
        section: Optional[str] = None,
    ) -> List[TextChunk]:
        """Split text belonging to one page (or one section) into TextChunk instances."""
        cleaned = _normalize_whitespace(text)
        if not cleaned:
            return []

        chunks: List[TextChunk] = []
        idx = start_chunk_idx
        for piece in self.split_text(cleaned):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}_p{page_number}_c{idx}",
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=idx,
                    text=piece,
                    char_count=len(piece),
                    section=section,
                )
            )
            idx += 1
        return chunks

    def chunk_markdown(self, text: str, doc_id: str, filename: str, page_number: int = 1) -> List[TextChunk]:
        """Chunk Markdown section by section, attaching the heading path (e.g. 'Guide > Setup') to each chunk.

        A section that is purely a bullet/numbered list is indexed one item per chunk: items of a list (e.g. the tasks
        due at one service interval) are usually independent facts, and embedding them together dilutes each one.
        """
        chunks: List[TextChunk] = []
        for section_path, body in split_markdown_sections(text):
            items = _list_items(body)
            pieces = items if len(items) >= 2 else [body]
            for piece in pieces:
                chunks.extend(
                    self.chunk_page_text(
                        piece,
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_number,
                        start_chunk_idx=len(chunks),
                        section=section_path,
                    )
                )
        return chunks

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    # Internals -------------------------------------------------------------------------------------------------

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[-1]
        remaining: List[str] = []
        for i, candidate in enumerate(separators):
            if candidate == "" or candidate in text:
                sep = candidate
                remaining = separators[i + 1 :]
                break

        parts = list(text) if sep == "" else text.split(sep)
        # Keep sentence punctuation attached to the sentence it ends.
        if sep == ". ":
            parts = [p + "." if i < len(parts) - 1 else p for i, p in enumerate(parts)]
            join_sep = " "
        else:
            join_sep = sep

        final: List[str] = []
        pending: List[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) <= self.chunk_size:
                pending.append(part)
                continue
            if pending:
                final.extend(self._merge(pending, join_sep))
                pending = []
            if remaining:
                final.extend(self._split(part, remaining))
            else:
                final.extend(self._merge(list(part), ""))
        if pending:
            final.extend(self._merge(pending, join_sep))
        return final

    def _merge(self, pieces: List[str], sep: str) -> List[str]:
        sep_len = len(sep)
        docs: List[str] = []
        current: List[str] = []
        total = 0
        for piece in pieces:
            plen = len(piece)
            extra = sep_len if current else 0
            if current and total + plen + extra > self.chunk_size:
                doc = sep.join(current).strip()
                if doc:
                    docs.append(doc)
                # Drop leading pieces until the carried-over tail fits the overlap budget and the next piece fits.
                while current and (
                    total > self.chunk_overlap or total + plen + (sep_len if current else 0) > self.chunk_size
                ):
                    total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                    current.pop(0)
            current.append(piece)
            total += plen + (sep_len if len(current) > 1 else 0)
        doc = sep.join(current).strip()
        if doc:
            docs.append(doc)
        return docs


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_LIST_START = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|[A-Za-z][.)])\s+")
_TERMINAL = (".", "!", "?", ":", ";")


def reflow_lines(text: str, min_wrapped_line: int = 40) -> str:
    """Join hard-wrapped lines (typical of PDF text extraction) back into paragraphs.

    A line is joined to the previous one when the previous line is long enough to be a wrapped line (at least 70% of
    the longest line, so titles and reference lines stay separate), does not end a sentence, and the next line does
    not start a list item. Short lines (headings) and blank lines are kept.
    """
    lines = _normalize_whitespace(text).split("\n")
    longest = max((len(line) for line in lines), default=0)
    wrap_threshold = max(min_wrapped_line, int(0.7 * longest))
    out: List[str] = []
    prev_physical_len = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            out.append("")
            prev_physical_len = 0
            continue
        if out and out[-1]:
            prev = out[-1]
            if (
                prev_physical_len >= wrap_threshold
                and not prev.endswith(_TERMINAL)
                and not _LIST_START.match(line)
                and not prev.isupper()
            ):
                if prev.endswith("-") and line[:1].islower():
                    out[-1] = prev + line  # re-join a hyphenated word ("micro-" + "fabricated")
                else:
                    out[-1] = f"{prev} {line}"
                prev_physical_len = len(line)
                continue
        out.append(line)
        prev_physical_len = len(line)
    return "\n".join(out).strip()


def split_markdown_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """Return (heading_path, body) pairs. Headings inside fenced code blocks are ignored."""
    sections: List[Tuple[Optional[str], str]] = []
    stack: List[Tuple[int, str]] = []
    body: List[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            path = " > ".join(title for _, title in stack) if stack else None
            sections.append((path, content))
        body.clear()

    for line in text.replace("\r\n", "\n").split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    flush()
    return sections
