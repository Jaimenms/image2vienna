"""Ollama backend: any vision model that answers ``POST /api/generate`` with images.

An instruction model such as Qwen2.5-VL 7B needs an instruction prompt (``inventory``);
``temperature`` is 0 so a description is reproducible for a given model build. The
evals of 2026-10-09 used it as the reference 7B model (``docs/evals.md``).
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from ..textnorm import clean_description
from .base import read_image
from .prompts import INVENTORY_PROMPT


class OllamaDescriber:
    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        timeout: float = 600.0,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def describe(self, image: Path | bytes, *, prompt: str = INVENTORY_PROMPT) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [base64.b64encode(read_image(image)).decode()],
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }
        try:
            resp = self._client.post(f"{self._host}/api/generate", json=payload)
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Ollama is not answering at {self._host}; start it with `ollama serve` "
                f"and pull the model with `ollama pull {self._model}`"
            ) from e
        resp.raise_for_status()
        return clean_description(resp.json()["response"])
