"""Run the vision model over eval cases once and cache the descriptions in the file.

Descriptions are the expensive stage (seconds per image); the embedding stage runs in
milliseconds. Caching them in the JSONL, keyed by describer and prompt, lets every
scoring experiment re-use one vision run, as text2ipc caches cross-encoder verdicts.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

from ..describe import PROMPTS, get_describer
from .cases import load_cases, save_cases


def description_key(describer: str, prompt: str = "default") -> str:
    return describer if prompt == "default" else f"{describer}|{prompt}"


def describe_cases(
    path: Path | str,
    describer: str,
    *,
    prompt: str = "default",
    limit: int | None = None,
    force: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
    checkpoint_every: int = 25,
) -> tuple[int, int]:
    """Describe the cases lacking a cached description; returns (described, total)."""
    cases = load_cases(path)
    key = description_key(describer, prompt)
    todo = [i for i, c in enumerate(cases) if force or key not in c.descriptions][:limit]
    if not todo:
        return 0, len(cases)
    model = get_describer(describer)
    text = PROMPTS[prompt]
    started = time.time()
    for n, i in enumerate(todo, 1):
        case = cases[i]
        desc = model.describe(case.image_path(), prompt=text)
        cases[i] = dataclasses.replace(case, descriptions={**case.descriptions, key: desc})
        if progress:
            progress(n, len(todo), (time.time() - started) / n)
        if n % checkpoint_every == 0:
            save_cases(cases, path)
    save_cases(cases, path)
    return len(todo), len(cases)
