"""PII redaction and prompt-injection detection for untrusted document text.

* PII detection is pattern-based with validation to keep false positives low in technical manuals:
  e-mail addresses, phone numbers, US social security numbers, payment card numbers (Luhn check), vehicle
  identification numbers (ISO 3779 check digit, or the word "VIN" right before it) and names introduced by a form
  label ("Customer: …").
  Detected values are replaced with ``[REDACTED <TYPE>]`` in everything DriveDocs returns or sends to an LLM;
  stored documents keep their original text.
* Prompt-injection detection flags passages containing instruction-like text aimed at an AI system ("ignore
  previous instructions", "reveal your system prompt", ...). Flagged sentences are never quoted as answers, the
  passage is shown with a warning, and LLM prompts mark all document text as untrusted data.

Pattern matching cannot find every kind of personal data (e.g. unlabelled names or street addresses) nor every
injection phrasing; it is one layer of defence alongside output verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])")
PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])"
)
SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])")
CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
VIN = re.compile(r"(?<![A-Za-z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Za-z0-9])", re.IGNORECASE)
# A person's name is only detected when a form label introduces it ("Customer: Jane Example").
LABELLED_NAME = re.compile(
    r"(\b(?i:customer|owner|name|driver|contact|client|patient|employee)[ \t]*(?i:name)?[ \t]*:[ \t]*)"
    r"[A-Z][a-z'\-]+(?:[ \t]+[A-Z][a-z'\-]+){0,2}"
)

_VIN_VALUES = {**{str(d): d for d in range(10)}, **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMN", range(1, 6))),
               "P": 7, "R": 9, **dict(zip("STUVWXYZ", range(2, 10)))}
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def vin_check_digit_valid(vin: str) -> bool:
    vin = vin.upper()
    if len(vin) != 17 or any(c not in _VIN_VALUES for c in vin):
        return False
    remainder = sum(_VIN_VALUES[c] * w for c, w in zip(vin, _VIN_WEIGHTS)) % 11
    return vin[8] == ("X" if remainder == 10 else str(remainder))


def _is_vin(text: str, match: re.Match) -> bool:
    value = match.group(0)
    if not (any(c.isdigit() for c in value) and any(c.isalpha() for c in value)):
        return False
    context = text[max(0, match.start() - 24) : match.start()].lower()
    return vin_check_digit_valid(value) or bool(re.search(r"\bvin\b|vehicle identification", context))


@dataclass
class RedactionResult:
    text: str
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    """Replace detected PII with [REDACTED <TYPE>] markers. Order matters: specific patterns run first."""
    if not text:
        return RedactionResult(text or "")
    counts: Dict[str, int] = {}

    def sub(pattern: re.Pattern, label: str, current: str, validate=None) -> str:
        def repl(m: re.Match) -> str:
            if validate is not None and not validate(current, m):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[REDACTED {label}]"

        return pattern.sub(repl, current)

    out = sub(EMAIL, "EMAIL", text)
    out = sub(VIN, "VIN", out, _is_vin)
    out = sub(CARD, "CARD", out, lambda _t, m: luhn_valid(m.group(0)))
    out = sub(SSN, "SSN", out)
    out = sub(PHONE, "PHONE", out)

    def name_repl(m: re.Match) -> str:
        counts["NAME"] = counts.get("NAME", 0) + 1
        return f"{m.group(1)}[REDACTED NAME]"

    out = LABELLED_NAME.sub(name_repl, out)
    return RedactionResult(out, counts)


def redact_text(text: str) -> str:
    return redact(text).text


INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|the)\b[^.\n]{0,30}"
               r"\b(instructions?|rules?|prompts?|directions?|guidelines?)\b", re.IGNORECASE),
    re.compile(r"\b(reveal|print|show|output|repeat|leak|disclose)\b[^.\n]{0,30}\b(system prompt|instructions|api[ _-]?key|"
               r"secret|password|credentials?)\b", re.IGNORECASE),
    re.compile(r"\byou are now\b|\bact as (an?|the)\b|\bnew instructions?\s*:", re.IGNORECASE),
    re.compile(r"\b(system|assistant|developer)\s*(prompt|message|notice)?\s*:\s*(you|ignore|from now)", re.IGNORECASE),
    re.compile(r"\b(answer|respond|reply) (to )?every (question|query)\b", re.IGNORECASE),
    re.compile(r"-{2,}\s*(begin|end)\s+untrusted", re.IGNORECASE),
]


def injection_signals(text: str) -> List[str]:
    """Instruction-like fragments that target an AI system (empty list when none)."""
    found: List[str] = []
    for pattern in INJECTION_PATTERNS:
        for m in pattern.finditer(text or ""):
            fragment = m.group(0).strip()
            if fragment and fragment not in found:
                found.append(fragment[:80])
    return found


def looks_like_injection(text: str) -> bool:
    return bool(injection_signals(text))
