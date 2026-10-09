"""Descriptions read from text files, for tests and for replaying a cached run.

``FixedDescriber(dir)`` answers for ``some/image.jpg`` with the contents of
``some/image.txt`` if it exists, else ``<dir>/image.txt``; bytes are described by the
hash-named file ``<dir>/<sha256[:16]>.txt``. No pixels are read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import read_image
from .prompts import DEFAULT_PROMPT


class FixedDescriber:
    def __init__(self, directory: Path):
        self._dir = Path(directory)

    @property
    def name(self) -> str:
        return f"fixed:{self._dir}"

    def describe(self, image: Path | bytes, *, prompt: str = DEFAULT_PROMPT) -> str:
        candidates = []
        if not isinstance(image, bytes):
            image = Path(image)
            candidates.append(image.with_suffix(".txt"))
            candidates.append(self._dir / f"{image.stem}.txt")
        digest = hashlib.sha256(read_image(image)).hexdigest()[:16]
        candidates.append(self._dir / f"{digest}.txt")
        for c in candidates:
            if c.is_file():
                return c.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"No fixed description among {[str(c) for c in candidates]}")
