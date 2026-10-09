"""Embedding backends behind one small interface.

A backend is selected by a spec string ``<backend>:<model>``:

- ``st:intfloat/multilingual-e5-small``  sentence-transformers, runs locally (default)
- ``ollama:nomic-embed-text``             any embedding model served by Ollama
- ``hash:64``                             deterministic bag-of-words, for tests only

Vectors are always L2-normalised so that a dot product is a cosine similarity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Full spec string, stored in the index metadata."""

    @property
    def dim(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """(len(texts), dim) float32 array of unit vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32 unit vector. Some models want a different prefix for queries."""

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """(len(texts), dim) unit vectors, one query per text."""

    @property
    def max_tokens(self) -> int | None:
        """Longest input the model embeds whole; None when unbounded or unknown."""

    def count_tokens(self, text: str) -> int:
        """Tokens ``embed_query(text)`` feeds the model, prefix and special tokens included."""


def get_embedder(spec: str | Embedder, **kwargs) -> Embedder:
    if not isinstance(spec, str):
        return spec
    backend, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"Embedder spec must look like 'backend:model', got {spec!r}")
    if backend == "st":
        from .st import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model, **kwargs)
    if backend == "ollama":
        from .ollama import OllamaEmbedder

        return OllamaEmbedder(model, **kwargs)
    if backend == "hash":
        from .hashing import HashEmbedder

        return HashEmbedder(int(model), **kwargs)
    raise ValueError(f"Unknown embedding backend {backend!r}")


def model_slug(name: str) -> str:
    """Filesystem-safe form of a spec, used in index file names."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-").lower()


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
