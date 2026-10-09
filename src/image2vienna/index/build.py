"""Build an index for one edition, reusing vectors from an older one.

An entry's vector depends only on its rendered text (the full path of titles, plus the
positive notes when ``notes=True``) and the model. For every code present in the
previous index with an identical text hash the old vector is copied instead of
recomputed. Between two editions that is most of the classification: the 10th edition
changed about a hundred entries out of nearly two thousand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..embeddings.base import Embedder
from ..scheme.table import SchemeTable
from .store import IndexMeta, ViennaIndex


@dataclass
class BuildReport:
    edition: str
    lang: str
    model: str
    notes: bool = False
    total: int = 0
    reused: int = 0
    computed: int = 0
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    previous: str | None = None

    def summary(self) -> str:
        style = " with notes" if self.notes else ""
        head = f"edition {self.edition} {self.lang} {self.model}{style}: {self.total} entries"
        if self.previous is None:
            return f"{head}, all {self.computed} embedded from scratch"
        return (
            f"{head}; reused {self.reused}, embedded {self.computed} "
            f"(added {len(self.added)}, changed {len(self.changed)}, "
            f"removed {len(self.removed)}) vs edition {self.previous}"
        )


def build_index(
    scheme: SchemeTable,
    embedder: Embedder,
    *,
    edition: str,
    lang: str,
    notes: bool = False,
    previous: ViennaIndex | tuple[Path, Path] | None = None,
    batch_size: int = 128,
    progress=None,
) -> tuple[ViennaIndex, BuildReport]:
    """``previous`` is an index or a ``(index parquet, scheme parquet)`` pair."""
    lang = lang.upper()
    report = BuildReport(
        edition=edition, lang=lang, model=embedder.name, notes=notes, total=len(scheme)
    )
    prev = _load_previous(previous, embedder, lang, notes, report)
    prev_hash = prev.hashes() if prev else {}
    texts = scheme.texts(notes=notes)
    hashes = scheme.hashes(notes=notes)

    vectors = np.zeros((len(scheme), embedder.dim), dtype=np.float32)
    todo: list[int] = []
    for i, node in enumerate(scheme.nodes):
        old = prev_hash.get(node.code)
        if old is not None and old == hashes[i]:
            vectors[i] = prev.vectors[prev.position[node.code]]  # type: ignore[union-attr]
            report.reused += 1
        else:
            todo.append(i)
            (report.changed if old is not None else report.added).append(node.code)
    report.removed = sorted(set(prev_hash) - set(scheme.position))
    report.computed = len(todo)

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors[batch] = embedder.embed_passages([texts[i] for i in batch])
        if progress is not None:
            progress(min(start + batch_size, len(todo)), len(todo))

    meta = IndexMeta.now(
        edition=edition,
        lang=lang,
        model=embedder.name,
        dim=embedder.dim,
        rows=len(scheme),
        notes=notes,
    )
    return ViennaIndex(meta, scheme, vectors, hashes), report


def _load_previous(
    previous: ViennaIndex | tuple[Path, Path] | None,
    embedder: Embedder,
    lang: str,
    notes: bool,
    report: BuildReport,
) -> ViennaIndex | None:
    if previous is None:
        return None
    prev = ViennaIndex.read(*previous) if isinstance(previous, tuple) else previous
    if prev.meta.model != embedder.name or prev.meta.dim != embedder.dim:
        raise ValueError(
            f"Previous index was built with {prev.meta.model} ({prev.meta.dim}d), "
            f"cannot reuse for {embedder.name} ({embedder.dim}d)"
        )
    if prev.meta.lang != lang or prev.meta.notes != notes:
        raise ValueError("Previous index has another language or text style, cannot reuse")
    report.previous = prev.meta.edition
    return prev
