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


_FRAGMENT_RE = re.compile(r"[A-Za-z]{3,}")
_TRAILING_JUNK_RE = re.compile(r"[\s,;:]+[^A-Za-z]*$")


def clean_description(text: str) -> str:
    """Tidy a vision model's output: drop repeated sentences (small models loop) and
    trailing fragments without words (``,,,,0``), keeping the first occurrences in
    order. The browser worker applies the same rule (``cleanDescription``)."""
    seen: set[str] = set()
    kept: list[str] = []
    for raw in split_sentences(text):
        sentence = _TRAILING_JUNK_RE.sub("", raw).strip()
        key = " ".join(sentence.lower().split())
        if not sentence or key in seen or not _FRAGMENT_RE.search(sentence):
            continue
        seen.add(key)
        kept.append(sentence)
    return " ".join(kept).strip()
