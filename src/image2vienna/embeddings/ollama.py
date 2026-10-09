"""Ollama backend: any model that answers ``POST /api/embed``."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np

from .base import normalize


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        batch_size: int = 32,
        timeout: float = 120.0,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_passages(["probe"]).shape[1])
        return self._dim

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.post(
            f"{self._host}/api/embed", json={"model": self._model, "input": list(texts)}
        )
        resp.raise_for_status()
        return normalize(np.asarray(resp.json()["embeddings"], dtype=np.float32))

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        chunks = [
            self._embed(texts[i : i + self._batch_size])
            for i in range(0, len(texts), self._batch_size)
        ]
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(chunks)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts))

    @property
    def max_tokens(self) -> int | None:
        return None  # Ollama truncates server side; the limit is not exposed

    def count_tokens(self, text: str) -> int:
        return len(text.split())
