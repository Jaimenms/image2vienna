"""Deterministic bag-of-words embedder for tests and offline smoke runs.

Each lower-cased token is hashed to one dimension, so texts sharing words are similar.
No semantics, but fully reproducible and dependency free.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hash:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324 - not security
            v[h % self._dim] += 1.0
        return v

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_passages([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_passages(texts)

    @property
    def max_tokens(self) -> int | None:
        return None

    def count_tokens(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text.lower()))
