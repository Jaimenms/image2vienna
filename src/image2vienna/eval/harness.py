"""Hit-rate at every hierarchy level.

A case counts as a hit at level L and cut-off k if any of the top-k predictions,
truncated to level L, equals any gold code truncated to level L. Gold sets are
multi-label (a mark carries one code per distinct figurative element) and the first
listed code is taken as the office's main one, so ``main_*`` metrics score only
against that code. ``recall@k`` is the share of gold codes found among the top-k,
the natural measure for multi-label coding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import LEVELS
from ..scheme.codes import truncate_code
from ..search import Match
from .cases import EvalCase


@dataclass
class EvalResult:
    level: str
    top_k: int
    n: int = 0
    hits: dict[str, dict[int, int]] = field(default_factory=dict)  # level -> k -> hits
    main_hits: dict[str, dict[int, int]] = field(default_factory=dict)
    recall: dict[str, dict[int, float]] = field(default_factory=dict)  # summed per case
    reciprocal_rank: dict[str, float] = field(default_factory=dict)
    scored: dict[str, int] = field(default_factory=dict)  # cases with gold at the level
    misses: list[tuple[str, str, list[str]]] = field(default_factory=list)

    def rate(self, level: str, k: int, *, main: bool = False) -> float:
        table = self.main_hits if main else self.hits
        n = self.scored.get(level, 0)
        return table.get(level, {}).get(k, 0) / n if n else 0.0

    def recall_at(self, level: str, k: int) -> float:
        n = self.scored.get(level, 0)
        return self.recall.get(level, {}).get(k, 0.0) / n if n else 0.0

    def mrr(self, level: str) -> float:
        n = self.scored.get(level, 0)
        return self.reciprocal_rank.get(level, 0.0) / n if n else 0.0

    def rows(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> list[dict]:
        ks = tuple(k for k in ks if k <= self.top_k)
        out = []
        for level in LEVELS[: LEVELS.index(self.level) + 1]:
            row = {"level": level, "n": self.scored.get(level, 0)}
            row.update({f"hit@{k}": self.rate(level, k) for k in ks})
            row.update({f"recall@{k}": self.recall_at(level, k) for k in ks if k > 1})
            row["main@1"] = self.rate(level, 1, main=True)
            row["mrr"] = self.mrr(level)
            out.append(row)
        return out

    def table(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> str:
        ks = tuple(k for k in ks if k <= self.top_k)
        rk = ks[-1]
        head = (
            "level        n" + "".join(f"  hit@{k:<3}" for k in ks) + f"  rec@{rk:<3}  main@1   mrr"
        )
        rows = [head]
        for level in LEVELS[: LEVELS.index(self.level) + 1]:
            cells = "".join(f"  {self.rate(level, k):6.1%}" for k in ks)
            rows.append(
                f"{level:<9}{self.scored.get(level, 0):6d}{cells}  {self.recall_at(level, rk):6.1%}"
                f"  {self.rate(level, 1, main=True):6.1%}  {self.mrr(level):.3f}"
            )
        return "\n".join(rows)


def evaluate(
    cases: list[EvalCase],
    classify: Callable[[EvalCase], list[Match]],
    *,
    level: str = "section",
    top_k: int = 10,
    progress: Callable[[int, int], None] | None = None,
) -> EvalResult:
    result = EvalResult(level=level, top_k=top_k)
    levels = LEVELS[: LEVELS.index(level) + 1]
    for lv in levels:
        result.hits[lv] = dict.fromkeys(range(1, top_k + 1), 0)
        result.main_hits[lv] = dict.fromkeys(range(1, top_k + 1), 0)
        result.recall[lv] = dict.fromkeys(range(1, top_k + 1), 0.0)
        result.reciprocal_rank[lv] = 0.0
        result.scored[lv] = 0

    for i, case in enumerate(cases, 1):
        preds = [m.code for m in classify(case)[:top_k]]
        result.n += 1
        for lv in levels:
            gold = {truncate_code(s, lv) for s in case.vienna if _deep_enough(s, lv)}
            if not gold:
                continue
            result.scored[lv] += 1
            main = truncate_code(case.vienna[0], lv) if _deep_enough(case.vienna[0], lv) else None
            seen: list[str] = []
            first_hit = None
            for p in preds:
                t = truncate_code(p, lv)
                if t in seen:
                    continue
                seen.append(t)
                if t in gold:
                    for k in range(len(seen), top_k + 1):
                        result.recall[lv][k] += 1.0 / len(gold)
                    if first_hit is None:
                        first_hit = len(seen)
                        result.reciprocal_rank[lv] += 1.0 / first_hit
                if t == main:
                    for k in range(len(seen), top_k + 1):
                        result.main_hits[lv][k] += 1
            if first_hit is not None:
                for k in range(first_hit, top_k + 1):
                    result.hits[lv][k] += 1
            elif lv == level:
                result.misses.append((case.id, ",".join(sorted(gold)), seen[:5]))
        if progress:
            progress(i, len(cases))
    return result


def exclude_gold(cases: list[EvalCase], codes: tuple[str, ...]) -> list[EvalCase]:
    """Drop gold codes under ``codes`` (``("29",)`` removes the colour codes) and the
    cases left without gold, so that a masked search is scored on the rest."""
    import dataclasses

    out = []
    for c in cases:
        kept = tuple(g for g in c.vienna if not any(g == x or g.startswith(x + ".") for x in codes))
        if kept:
            out.append(dataclasses.replace(c, vienna=kept))
    return out


def _deep_enough(code: str, level: str) -> bool:
    """A division-only gold code (``1.1``) cannot score at the section level."""
    return code.count(".") >= LEVELS.index(level)
