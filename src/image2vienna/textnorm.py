"""Query normalisation: whitespace, and lower-casing of text that is mostly capitals.

Descriptions from the vision model are mixed case and pass through unchanged; the
rule exists for hand-written queries and for parity with text2ipc, where lower-casing
shouting patent titles lifted recall by ten points.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"[ \t\r\f\v]+")


def collapse_whitespace(text: str) -> str:
    return "\n".join(_WS_RE.sub(" ", line).strip() for line in text.strip().splitlines())


def lowercase_if_shouting(text: str, threshold: float = 0.7) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return text
    upper = sum(1 for c in letters if c.isupper())
    return text.lower() if upper / len(letters) >= threshold else text


def normalize_query(text: str) -> str:
    return " ".join(lowercase_if_shouting(collapse_whitespace(text)).split())


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Sentences of a description, for per-element scoring; never empty."""
    parts = [p.strip() for p in _SENTENCE_END_RE.split(" ".join(text.split())) if p.strip()]
    return parts or [text.strip()]
