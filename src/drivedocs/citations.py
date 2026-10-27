"""Citation verification.

Every ``[Source: <filename>, Page <n>]`` tag in an answer is checked before it is shown to the user:

1. the document exists in the registry (status ``ready``);
2. the page exists (1 ≤ page ≤ page_count) when page metadata is available;
3. the cited passage belongs to that document and page *and* was part of the retrieved evidence;
4. the claim preceding the tag is lexically supported by the cited passage (content-word overlap).

Tags failing checks 1–3 are removed from the answer and reported as warnings; claims failing check 4 are kept but
flagged as weakly supported. Lexical overlap is a heuristic: it can confirm that wording comes from a passage, but it
cannot prove semantic entailment, so verified citations are evidence of provenance, not a guarantee of correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from drivedocs.schemas import Citation, CitationCheck, DocumentMetadata, TextSearchResult

CITATION_PATTERN = re.compile(r"\[Source:\s*(?P<file>[^,\]\n]+?)\s*,\s*Page\s*(?P<page>\d+|N/A)\s*\]", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?%?")
REMOVED_MARKER = "[unverified citation removed]"

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves""".split()
)

SupportFn = Callable[[str], Optional[DocumentMetadata]]


def content_tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def lexical_support(claim: str, passage: str) -> float:
    """Fraction of the claim's content words present in the passage (prefix match tolerates inflection)."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0
    passage_tokens = set(content_tokens(passage))
    passage_prefixes = {t[:5] for t in passage_tokens if len(t) >= 5}
    hits = 0
    for tok in claim_tokens:
        if tok in passage_tokens or (len(tok) >= 5 and tok[:5] in passage_prefixes):
            hits += 1
    return hits / len(claim_tokens)


@dataclass
class VerificationResult:
    answer: str
    citations: List[Citation]
    check: CitationCheck
    warnings: List[str] = field(default_factory=list)


class CitationVerifier:
    """Verifies citation tags against retrieved evidence and the document registry."""

    def __init__(self, lookup_document: Optional[Callable[[str], List[DocumentMetadata]]] = None,
                 weak_support_threshold: float = 0.5) -> None:
        # lookup_document(filename) -> registered documents with that filename
        self.lookup_document = lookup_document
        self.weak_support_threshold = weak_support_threshold

    def verify(self, answer: str, evidence: List[TextSearchResult]) -> VerificationResult:
        by_source: Dict[Tuple[str, int], List[TextSearchResult]] = {}
        for res in evidence:
            by_source.setdefault((res.chunk.filename.lower(), res.chunk.page_number), []).append(res)

        warnings: List[str] = []
        citations: Dict[str, Citation] = {}
        check = CitationCheck()
        pieces: List[str] = []
        last_end = 0
        claim_start = 0

        for match in CITATION_PATTERN.finditer(answer):
            check.total += 1
            filename = match.group("file").strip().strip("'\"")
            page_raw = match.group("page")
            claim = answer[claim_start : match.start()]
            claim = re.sub(CITATION_PATTERN, "", claim).strip()
            claim = claim.split("\n")[-1].strip()
            pieces.append(answer[last_end : match.start()])
            last_end = match.end()
            claim_start = match.end()

            problem, citation = self._check_one(filename, page_raw, claim, by_source)
            if problem:
                check.removed += 1
                warnings.append(problem)
                pieces.append(REMOVED_MARKER)
                continue
            assert citation is not None
            if citation.support is not None and citation.support < self.weak_support_threshold:
                check.weakly_supported += 1
                warnings.append(
                    f"The claim cited to {citation.filename}, Page {citation.page_number} is only weakly supported "
                    f"by the cited passage ({citation.support:.0%} word overlap)."
                )
            check.verified += 1
            pieces.append(match.group(0))
            key = citation.chunk_id or f"{citation.filename}:{citation.page_number}"
            if key not in citations or (citation.support or 0) > (citations[key].support or 0):
                citations[key] = citation

        pieces.append(answer[last_end:])
        check.all_verified = check.removed == 0 and check.weakly_supported == 0
        return VerificationResult(answer="".join(pieces), citations=list(citations.values()), check=check,
                                  warnings=warnings)

    def _check_one(
        self,
        filename: str,
        page_raw: str,
        claim: str,
        by_source: Dict[Tuple[str, int], List[TextSearchResult]],
    ) -> Tuple[Optional[str], Optional[Citation]]:
        if page_raw.upper() == "N/A":
            return f"Removed a citation to '{filename}' without a page reference.", None
        page = int(page_raw)

        registered: List[DocumentMetadata] = []
        if self.lookup_document is not None:
            registered = [d for d in self.lookup_document(filename) if d.status == "ready"]
            if not registered:
                return f"Removed a citation to '{filename}': no such document is indexed.", None
            if all(d.page_count and page > d.page_count for d in registered) or page < 1:
                return f"Removed a citation to '{filename}, Page {page}': the document has no such page.", None

        candidates = by_source.get((filename.lower(), page), [])
        if registered:
            valid_ids = {d.doc_id for d in registered}
            candidates = [c for c in candidates if c.chunk.doc_id in valid_ids]
        if not candidates:
            return (
                f"Removed a citation to '{filename}, Page {page}': it does not match any retrieved passage.",
                None,
            )

        best = max(candidates, key=lambda r: lexical_support(claim, r.chunk.text))
        support = round(lexical_support(claim, best.chunk.text), 3) if claim else None
        snippet = best.chunk.text if len(best.chunk.text) <= 160 else best.chunk.text[:157] + "..."
        return None, Citation(
            filename=best.chunk.filename,
            page_number=best.chunk.page_number,
            chunk_id=best.chunk.chunk_id,
            doc_id=best.chunk.doc_id,
            snippet=snippet,
            score=round(best.score, 4),
            verified=True,
            support=support,
        )
