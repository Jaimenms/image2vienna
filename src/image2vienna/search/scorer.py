"""Hierarchical scoring on top of flat cosine similarity.

Flat nearest-neighbour over the ~2,000 entries has two failure modes: deep entries
with long specific texts win on word overlap alone, and an isolated hit in an
unrelated category can outrank a well-supported one. Three heuristics address that,
all ported from text2ipc (``search/scorer.py`` there):

1. **Path support** - the mean similarity of an entry's ancestors.
2. **Subtree support** - the best similarity anywhere below an entry.
3. **Beam descent** - candidates at a level are only considered under the top-``b``
   parents of the level above, parents being ranked by the best composite score in
   their subtree, so the flat best candidate always survives pruning.

The final score is a weighted sum of own similarity, path support and subtree
support. ``level="auto"`` walks down from the best divisions along the best-scoring
branch and answers with the best node on that walk.

**Branch de-duplication** runs last: a result is dropped when a better-ranked one is
its ancestor or its descendant, so the list keeps distinct branches. Vienna coding
is multi-label by design (general note (e), one code per distinct element), so the
answer is a list and may hold fewer than ``k`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AUTO_LEVEL, LEVELS
from ..index.store import ViennaIndex
from ..scheme.codes import format_code


@dataclass(frozen=True)
class Weights:
    """text2ipc measured path support as harmful on a scheme whose top titles are
    generic; the Vienna categories are concrete ("Animals", "Plants"), so the
    weights are an eval question here (docs/evals.md). Cosine alone is the default."""

    own: float = 1.0
    path: float = 0.0
    subtree: float = 0.0


@dataclass(frozen=True)
class Beam:
    """How many parents survive at each level before descending."""

    category: int = 8
    division: int = 20

    def width(self, level: str) -> int:
        return getattr(self, level)


@dataclass(frozen=True)
class SearchParams:
    level: str = "section"
    top_k: int = 10
    weights: Weights = field(default_factory=Weights)
    beam: Beam = field(default_factory=Beam)
    #: Drop results scoring more than this below the best one (None keeps top_k).
    gap: float | None = None
    #: ``auto`` level: descend while best child >= parent - margin.
    auto_margin: float = 0.02
    #: ``auto`` level: how many divisions to start descending from.
    auto_roots: int | None = None
    #: Drop results that are ancestors or descendants of a higher-ranked result.
    dedupe_branches: bool = True
    #: Leave auxiliary ("A") sections out of the candidates.
    principal_only: bool = False
    #: Codes whose whole subtree is left out of the candidates (e.g. ``("29",)`` to
    #: keep the Colours category, which every description mentions, from the list).
    exclude_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Match:
    code: str
    level: str
    title: str
    text: str
    score: float
    similarity: float
    path_support: float
    subtree_support: float
    auxiliary: bool = False

    @property
    def pretty(self) -> str:
        return format_code(self.code)

    @property
    def padded(self) -> str:
        """EUIPO style, ``01.01.02``."""
        return format_code(self.code, padded=True)


def search(
    index: ViennaIndex, query: np.ndarray, params: SearchParams | None = None
) -> list[Match]:
    """Rank entries for a unit query vector, or for a ``(chunks, dim)`` stack of them."""
    params = params or SearchParams()
    if params.level != AUTO_LEVEL and params.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS} or {AUTO_LEVEL!r}")

    q = np.asarray(query, dtype=np.float32)
    sims = index.vectors @ q if q.ndim == 1 else (index.vectors @ q.T).max(axis=1)
    masked = _mask(index, params)
    if masked is not None:
        sims = np.where(masked, -1.0, sims).astype(np.float32)
    path = _path_support(index, sims)
    subtree = _subtree_support(index, sims)
    w = params.weights
    score = w.own * sims + w.path * path + w.subtree * subtree

    best_below = _subtree_support(index, score)

    if params.level == AUTO_LEVEL:
        chosen = _auto_descend(index, sims, score, best_below, params)
    else:
        chosen = _beam_descend(index, best_below, params.level, params.beam)
        chosen = sorted(chosen, key=lambda i: -score[i])[: params.top_k]

    if masked is not None:
        chosen = [i for i in chosen if not masked[i]]
    if params.gap is not None and chosen:
        best = score[chosen[0]]
        chosen = [i for i in chosen if score[i] >= best - params.gap]
    if params.dedupe_branches:
        chosen = _distinct_branches(index, chosen)

    return [
        Match(
            code=index.nodes[i].code,
            level=index.nodes[i].level,
            title=index.nodes[i].title,
            text=index.texts[i],
            score=float(score[i]),
            similarity=float(sims[i]),
            path_support=float(path[i]),
            subtree_support=float(subtree[i]),
            auxiliary=index.nodes[i].auxiliary,
        )
        for i in chosen
    ]


def _mask(index: ViennaIndex, params: SearchParams) -> np.ndarray | None:
    """Rows to leave out: auxiliary sections and/or the subtrees of ``exclude_codes``."""
    if not params.principal_only and not params.exclude_codes:
        return None
    mask = np.zeros(len(index), dtype=bool)
    if params.principal_only:
        mask |= np.array([n.auxiliary for n in index.nodes])
    for code in params.exclude_codes:
        prefix = code + "."
        mask |= np.array([c == code or c.startswith(prefix) for c in index.codes])
    return mask


def _distinct_branches(index: ViennaIndex, ranked: list[int]) -> list[int]:
    """Keep a result only if no accepted result sits on its path or below it."""
    accepted: list[int] = []
    accepted_chains: list[set[str]] = []
    for i in ranked:
        chain = set(index.scheme.paths[i])
        code = index.codes[i]
        same_branch = any(
            index.codes[a] in chain or code in a_chain
            for a, a_chain in zip(accepted, accepted_chains, strict=True)
        )
        if not same_branch:
            accepted.append(i)
            accepted_chains.append(chain)
    return accepted


def _path_support(index: ViennaIndex, sims: np.ndarray) -> np.ndarray:
    """Mean similarity of the ancestors; equals own similarity for roots."""
    total = np.zeros_like(sims)
    count = np.zeros(len(sims), dtype=np.int32)
    for i, p in enumerate(index.parent_idx):  # parents precede children
        if p >= 0:
            total[i] = total[p] + sims[p]
            count[i] = count[p] + 1
    return np.where(count > 0, total / np.maximum(count, 1), sims)


def _subtree_support(index: ViennaIndex, sims: np.ndarray) -> np.ndarray:
    """Max similarity over the node and everything below it."""
    best = sims.copy()
    parents = index.parent_idx
    for i in range(len(sims) - 1, -1, -1):  # children precede parents in reverse
        p = parents[i]
        if p >= 0 and best[i] > best[p]:
            best[p] = best[i]
    return best


def _beam_descend(index: ViennaIndex, best_below: np.ndarray, target: str, beam: Beam) -> list[int]:
    """Walk down from the categories, keeping the parents whose subtree scores best."""
    target_rank = LEVELS.index(target)
    frontier = [i for i in range(len(index)) if index.levels[i] == 0]  # categories
    for rank, level in enumerate(LEVELS):
        if rank == target_rank:
            return frontier
        frontier = sorted(frontier, key=lambda i: -best_below[i])[: beam.width(level)]
        frontier = [c for i in frontier for c in index.children[i]]
    return frontier


def _auto_descend(
    index: ViennaIndex,
    sims: np.ndarray,
    score: np.ndarray,
    best_below: np.ndarray,
    params: SearchParams,
) -> list[int]:
    """From the best divisions, walk down while the best child's subtree promises
    something within ``auto_margin`` of the current node, then answer with the node
    on the walked path whose own similarity is highest (deepest on ties)."""
    n_roots = params.auto_roots if params.auto_roots is not None else max(5, params.top_k)
    roots = _beam_descend(index, best_below, "division", params.beam)
    roots = sorted(roots, key=lambda i: -best_below[i])[:n_roots]
    out: list[int] = []
    for i in roots:
        path = [i]
        node = i
        while index.children[node]:
            best_child = max(index.children[node], key=lambda c: best_below[c])
            if best_below[best_child] < score[node] - params.auto_margin:
                break
            node = best_child
            path.append(node)
        best = max(reversed(path), key=lambda j: sims[j])
        if best not in out:
            out.append(best)
    return sorted(out, key=lambda i: -score[i])[: params.top_k]
