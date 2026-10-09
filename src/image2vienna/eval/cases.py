"""Eval cases: an image and the Vienna codes an office assigned to it.

A case stores the image path relative to the eval home (``<home>/images``), the gold
codes in canonical form and, once a vision model has run, its description, so that
the embedding stage can be re-evaluated without the vision model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import home


@dataclass(frozen=True)
class EvalCase:
    id: str
    image: str  # path relative to ``images_dir()``, or absolute
    vienna: tuple[str, ...]  # canonical gold codes, office order, first = main
    edition: str | None = None  # edition the office coded with, when known
    text: str | None = None  # verbal element of the mark, when any
    source: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: Cached descriptions, keyed by ``<describer spec>|<prompt name>``.
    descriptions: dict[str, str] = field(default_factory=dict)

    def image_path(self, root: Path | None = None) -> Path:
        p = Path(self.image)
        return p if p.is_absolute() else images_dir(root) / p


def images_dir(root: Path | None = None) -> Path:
    return (root or home()) / "images"


def save_cases(cases: list[EvalCase], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_cases(path: Path | str) -> list[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                d["vienna"] = tuple(d["vienna"])
                d["tags"] = tuple(d.get("tags", ()))
                d["descriptions"] = dict(d.get("descriptions", {}))
                cases.append(EvalCase(**d))
    return cases


def load_many(paths: list[Path | str] | str, *, sample: int | None = None, seed: int = 0):
    """Load several JSONL files (or a glob) and optionally draw a random sample."""
    import glob
    import random

    files = sorted(glob.glob(paths)) if isinstance(paths, str) else list(paths)
    cases = [c for f in files for c in load_cases(f)]
    if sample is None or sample >= len(cases):
        return cases
    return random.Random(seed).sample(cases, sample)
