"""Vision-language backends behind one small interface: an image in, a description
of its figurative elements out. The description is the query text that the
embedding stage scores against the classification, exactly as a patent abstract is
the query in text2ipc.

A backend is selected by a spec string ``<backend>:<model>``:

- ``ollama:qwen2.5vl:7b``   any vision model served by Ollama (default, the "heavy" one)
- ``hf:HuggingFaceTB/SmolVLM-256M-Instruct``  a small model through transformers, run
                            locally (the "light" one, the same the browser demo runs)
- ``fixed:<dir>``           for tests and cached runs: reads ``<image stem>.txt`` next
                            to the image or under ``<dir>``, never looks at pixels
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .prompts import DEFAULT_PROMPT


@runtime_checkable
class Describer(Protocol):
    @property
    def name(self) -> str:
        """Full spec string, recorded next to every description."""

    def describe(self, image: Path | bytes, *, prompt: str = DEFAULT_PROMPT) -> str:
        """English prose naming every figurative element of the image."""


def get_describer(spec: str | Describer, **kwargs) -> Describer:
    if not isinstance(spec, str):
        return spec
    backend, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"Describer spec must look like 'backend:model', got {spec!r}")
    if backend == "ollama":
        from .ollama import OllamaDescriber

        return OllamaDescriber(model, **kwargs)
    if backend == "hf":
        from .hf import HfDescriber

        return HfDescriber(model, **kwargs)
    if backend == "fixed":
        from .fixed import FixedDescriber

        return FixedDescriber(Path(model), **kwargs)
    raise ValueError(f"Unknown describer backend {backend!r}")


def read_image(image: Path | bytes) -> bytes:
    return image if isinstance(image, bytes) else Path(image).read_bytes()
