"""Score every configuration of the embedding stage from cached descriptions.

uv run python scripts/eval_sweep.py evals/l3d_300.jsonl [--describer ollama:qwen2.5vl:7b]

One vision run per case is cached in the JSONL (``i2vienna describe-cases``); this
script loads the embedder once and evaluates the text style (titles vs notes), the
hierarchy weights, auxiliary sections, the target level and the prompt, printing one
Markdown table per level for ``docs/evals.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import Counter

from image2vienna.classifier import ViennaClassifier
from image2vienna.config import LEVELS, default_describer
from image2vienna.eval import load_cases
from image2vienna.eval.describe import description_key
from image2vienna.eval.harness import evaluate, exclude_gold
from image2vienna.scheme.codes import normalize_code, truncate_code
from image2vienna.search import Weights


def frequency_baseline(cases, labels_path: str, top_k: int = 10):
    """Always answer the k most frequent codes of the whole label set."""
    with open(labels_path) as fh:
        labels = json.load(fh)
    freq: Counter[str] = Counter()
    for v in labels.values():
        for raw in v["codes"]:
            with contextlib.suppress(ValueError):
                freq[normalize_code(str(raw))] += 1
    top = [c for c, _ in freq.most_common(50)]
    rows = {}
    for level in LEVELS:
        preds = list(dict.fromkeys(truncate_code(c, level) for c in top))[:top_k]
        rows[level] = {}
        for k in (1, 3, 5, 10):
            hits = scored = 0
            rec = 0.0
            for case in cases:
                gold = {
                    truncate_code(g, level)
                    for g in case.vienna
                    if g.count(".") >= LEVELS.index(level)
                }
                if not gold:
                    continue
                scored += 1
                found = gold & set(preds[:k])
                hits += bool(found)
                rec += len(found) / len(gold)
            rows[level][k] = (hits / scored, rec / scored, scored)
    return rows


def category_breakdown(cases, clf, key, baseline_codes, chunking="whole", top_k=10):
    """Per gold category: share of gold codes (section level) found in the top-k by the
    model and by the frequency baseline, and the same at division level."""
    from image2vienna.scheme import SchemeTable

    scheme: SchemeTable = clf.index.scheme
    per_cat: dict[str, dict[str, float]] = {}
    base_div = list(dict.fromkeys(truncate_code(c, "division") for c in baseline_codes))[:top_k]
    for c in cases:
        preds = [
            m.code
            for m in clf.classify_text(
                c.descriptions[key], level="section", top_k=top_k, chunking=chunking
            )
        ]
        pred_div = list(dict.fromkeys(truncate_code(p, "division") for p in preds))
        for g in c.vienna:
            cat = g.split(".")[0]
            d = per_cat.setdefault(
                cat, {"n": 0, "sec": 0, "sec_b": 0, "div": 0, "div_b": 0, "n_div": 0}
            )
            if g.count(".") == 2:
                d["n"] += 1
                d["sec"] += g in preds
                d["sec_b"] += g in baseline_codes[:top_k]
            d["n_div"] += 1
            d["div"] += truncate_code(g, "division") in pred_div
            d["div_b"] += truncate_code(g, "division") in base_div
    print(f"## Gold codes found in the top {top_k}, by category ({chunking} description)\n")
    print(
        "| category | gold codes | section: model | section: baseline "
        "| division: model | division: baseline |"
    )
    print("|---|---|---|---|---|---|")
    for cat, d in sorted(per_cat.items(), key=lambda kv: -kv[1]["n_div"]):
        if d["n_div"] < 8:
            continue
        title = scheme.node(cat).title if cat in scheme.position else "?"
        sec = f"{d['sec'] / d['n']:.0%}" if d["n"] else "-"
        sec_b = f"{d['sec_b'] / d['n']:.0%}" if d["n"] else "-"
        print(
            f"| {cat} {title[:38]} | {d['n_div']} | {sec} | {sec_b} "
            f"| {d['div'] / d['n_div']:.0%} | {d['div_b'] / d['n_div']:.0%} |"
        )
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("--describer", default=default_describer())
    ap.add_argument("--labels", default="data/l3d/labels.json")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    all_cases = load_cases(args.cases)
    W = Weights
    configs = [
        # name, prompt, notes, weights, principal_only, chunking, exclude
        ("titles, whole description", "default", False, W(), False, "whole", ()),
        ("titles + notes, whole", "default", True, W(), False, "whole", ()),
        ("titles, sentence mean", "default", False, W(), False, "mean", ()),
        ("titles, sentence max", "default", False, W(), False, "max", ()),
        ("titles + notes, sentence max", "default", True, W(), False, "max", ()),
        ("titles, whole, path 0.3", "default", False, W(0.7, 0.3), False, "whole", ()),
        ("titles, whole, subtree 0.3", "default", False, W(0.7, 0.0, 0.3), False, "whole", ()),
        ("titles, sentence max, path 0.3", "default", False, W(0.7, 0.3), False, "max", ()),
        ("titles, whole, principal only", "default", False, W(), True, "whole", ()),
        ("titles, whole, no colours (29)", "default", False, W(), False, "whole", ("29",)),
        ("titles, sentence max, no colours (29)", "default", False, W(), False, "max", ("29",)),
        (
            "titles, whole, pictorial only (no 26-29)",
            "default",
            False,
            W(),
            False,
            "whole",
            ("26", "27", "28", "29"),
        ),
        (
            "titles, sentence mean, pictorial only (no 26-29)",
            "default",
            False,
            W(),
            False,
            "mean",
            ("26", "27", "28", "29"),
        ),
        ("titles, inventory prompt, whole", "inventory", False, W(), False, "whole", ()),
        ("titles, inventory prompt, sentence mean", "inventory", False, W(), False, "mean", ()),
        ("titles, inventory prompt, sentence max", "inventory", False, W(), False, "max", ()),
        (
            "titles, inventory prompt, pictorial only (no 26-29)",
            "inventory",
            False,
            W(),
            False,
            "whole",
            ("26", "27", "28", "29"),
        ),
        ("titles, terse prompt, whole", "terse", False, W(), False, "whole", ()),
        ("titles, terse prompt, sentence max", "terse", False, W(), False, "max", ()),
    ]
    results = []
    clfs = {}
    for name, prompt, notes, weights, principal, chunking, exclude in configs:
        key = description_key(args.describer, prompt)
        cases = [c for c in all_cases if key in c.descriptions]
        if exclude:
            cases = exclude_gold(cases, exclude)
        if not cases:
            print(f"skip {name}: no descriptions under {key}")
            continue
        if notes not in clfs:
            clfs[notes] = ViennaClassifier("10", notes=notes)
        clf = clfs[notes]
        res = evaluate(
            cases,
            lambda c, clf=clf, key=key, w=weights, p=principal, ch=chunking, ex=exclude: (
                clf.classify_text(
                    c.descriptions[key],
                    level="section",
                    top_k=args.top_k,
                    weights=w,
                    principal_only=p,
                    chunking=ch,
                    exclude_codes=ex,
                )
            ),
            level="section",
            top_k=args.top_k,
        )
        results.append((name, len(cases), res))
        print(f"{name}: n={len(cases)}")
        print(res.table())
        print()

    # Candidate recall ceiling: how often the gold is anywhere in a long first-stage
    # list. The gap between hit@1 and hit@50 is what a second stage (cross-encoder or
    # LLM judge, as in text2ipc) could recover; the rest needs a better description.
    key = description_key(args.describer, "default")
    cases = [c for c in all_cases if key in c.descriptions]
    if cases:
        clf = clfs.get(False) or ViennaClassifier("10")
        print("## Candidate recall ceiling (titles, top 50)\n")
        print("| query | category hit@50 | division hit@50 | section hit@50 | section recall@50 |")
        print("|---|---|---|---|---|")
        for chunking in ("whole", "max"):
            res = evaluate(
                cases,
                lambda c, ch=chunking: clf.classify_text(
                    c.descriptions[key], level="section", top_k=50, chunking=ch
                ),
                level="section",
                top_k=50,
            )
            print(
                f"| {chunking} | {res.rate('category', 50):.1%} | {res.rate('division', 50):.1%} "
                f"| {res.rate('section', 50):.1%} | {res.recall_at('section', 50):.1%} |"
            )
        print()

    base = frequency_baseline(all_cases, args.labels, args.top_k)
    if cases:
        with open(args.labels) as fh:
            labels = json.load(fh)
        freq: Counter[str] = Counter()
        for v in labels.values():
            for raw in v["codes"]:
                with contextlib.suppress(ValueError):
                    freq[normalize_code(str(raw))] += 1
        baseline_codes = [c for c, _ in freq.most_common(100)]
        clf = clfs.get(False) or ViennaClassifier("10")
        category_breakdown(cases, clf, key, baseline_codes, "whole", args.top_k)
    print("## Markdown\n")
    for level in LEVELS:
        print(f"**{level}** (n = {base[level][1][2]})\n")
        print("| configuration | hit@1 | hit@3 | hit@10 | recall@10 | main@1 | MRR |")
        print("|---|---|---|---|---|---|---|")
        b = base[level]
        print(
            f"| frequency baseline | {b[1][0]:.1%} | {b[3][0]:.1%} | {b[10][0]:.1%} "
            f"| {b[10][1]:.1%} | - | - |"
        )
        for name, _n, res in results:
            print(
                f"| {name} | {res.rate(level, 1):.1%} | {res.rate(level, 3):.1%} "
                f"| {res.rate(level, 10):.1%} | {res.recall_at(level, 10):.1%} "
                f"| {res.rate(level, 1, main=True):.1%} | {res.mrr(level):.3f} |"
            )
        print()


if __name__ == "__main__":
    main()
