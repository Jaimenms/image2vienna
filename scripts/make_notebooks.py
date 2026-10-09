"""Generate the study notebook from code, so it stays reproducible and thin.

uv run python scripts/make_notebooks.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_image2vienna.ipynb

Logic lives in the package; the notebook only calls it. Executing it needs the
edition 10 index built and, for the last section, ``evals/l3d_300.jsonl`` with cached
descriptions; the vision model downloads from the Hub on first use.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook_01():
    cells = [
        md("""
# 01 · image2vienna: from a trade mark image to Vienna codes

This notebook shows the public interface of `image2vienna` on two drawn logos and
evaluates it on 300 EUIPO marks with the examiners' codes.

Prerequisites, run once from the repository root:

```bash
uv sync --all-extras
uv run i2vienna build --edition 10                 # WIPO XML -> scheme table + index
uv run i2vienna l3d --n 300                        # eval cases (section 7)
uv run i2vienna describe-cases evals/l3d_300.jsonl # the vision model over the cases, cached
```

Everything below reads the Parquet tables under `data/` (or `$IMAGE2VIENNA_HOME`).
"""),
        code("""
import tempfile
from pathlib import Path

import pandas as pd
from IPython.display import Image as Show
from IPython.display import display

from image2vienna import ViennaClassifier

ROOT = Path.cwd() if (Path.cwd() / "evals").is_dir() else Path.cwd().parent
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", None)


def as_frame(matches):
    # One row per match; `path` is the full text the entry was embedded with
    # (category > division > section), which is also the explanation.
    return pd.DataFrame(
        [
            {
                "code": ("A " if m.auxiliary else "") + m.pretty,
                "level": m.level,
                "score": round(m.score, 3),
                "path": m.text,
            }
            for m in matches
        ]
    )
"""),
        md("""
## 1. One classifier, one index

A `ViennaClassifier` is bound to an edition, a scheme language, an embedding model
and a vision model (Florence-2 base by default, 230M parameters, through
transformers on CPU or GPU). Loading is lazy: the index (1,955 vectors, 7.7 MB), the
embedder and the vision model load on first use. Every entry was embedded from its
full path of titles, never from its own title alone.
"""),
        code("""
clf = ViennaClassifier("10")
print("edition", clf.edition, "|", clf.index.meta.model, "|", len(clf.index), "entries")
print(clf.index.text_of("1.1.2"))
print(clf.index.text_of("24.9.9"))
"""),
        md("""
## 2. Two drawn logos

The examples are drawn with Pillow so the notebook runs without any image file:
three stars above a crescent moon, and the letters "AB" on a shield under a crown.
"""),
        code("""
import math

from PIL import Image, ImageDraw, ImageFont

W = 512
work = Path(tempfile.mkdtemp())


def star(d, cx, cy, r, fill):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=fill)


im = Image.new("RGB", (W, W), "white")
d = ImageDraw.Draw(im)
d.ellipse((120, 200, 400, 480), fill="navy")
d.ellipse((190, 170, 450, 430), fill="white")
for cx in (160, 256, 352):
    star(d, cx, 110, 42, "goldenrod")
stars_moon = work / "stars_moon.png"
im.save(stars_moon)

im = Image.new("RGB", (W, W), "white")
d = ImageDraw.Draw(im)
d.polygon([(120, 140), (392, 140), (392, 330), (256, 460), (120, 330)], fill="firebrick")
d.polygon(
    [(150, 130), (150, 60), (205, 110), (256, 40), (307, 110), (362, 60), (362, 130)],
    fill="goldenrod",
)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 150)
except OSError:
    font = ImageFont.load_default()
d.text((256, 290), "AB", fill="white", font=font, anchor="mm")
shield = work / "shield_crown_ab.png"
im.save(shield)

display(Show(filename=str(stars_moon), width=200), Show(filename=str(shield), width=200))
"""),
        md("""
## 3. Stage one: what the vision model sees

`describe` runs Florence-2's detailed-caption task on the image: a few sentences
naming the objects, letters and colours literally (repeated sentences, which small
models produce, are dropped). The prose is the query of the next stage.
"""),
        code("""
for image in (stars_moon, shield):
    print(image.name)
    print(clf.describe(image))
    print()
"""),
        md("""
## 4. Stage two: ranked codes

`classify` runs both stages. `level` picks the answer's level (`category`,
`division`, `section`, or `auto` for as deep as the evidence supports); the `A`
before a code marks an auxiliary section, which offices code without the letter
(`m.padded` prints the EUIPO form `01.01.04`). Results on the same branch as a
better-ranked one are dropped, so each row is a distinct branch.
"""),
        code("""
as_frame(clf.classify(stars_moon, level="section", top_k=5))
"""),
        code("""
as_frame(clf.classify(shield, level="division", top_k=5))
"""),
        code("""
as_frame(clf.classify(shield, level="auto", top_k=5))
"""),
        md("""
## 5. A description as the query

`classify_text` runs the cheap stage alone, so a cached or hand-edited description
costs milliseconds and a sentence written by hand is a valid input. `chunking="max"`
embeds each sentence separately and scores an entry by its best sentence, one code
per element; `exclude_codes` masks a subtree, here the Colours category, which every
description mentions.
"""),
        code("""
text = (
    "A roaring lion's head seen from the front, with a mane. "
    "Two crossed swords below it. Red and gold."
)
as_frame(
    clf.classify_text(text, level="section", top_k=5, chunking="max", exclude_codes=("29",))
)
"""),
        md("""
## 6. Reading an entry

The scheme table keeps each entry's code path, its explanatory notes and, for an
auxiliary section, the principal sections it is associated with. The embedded text
is the chain of titles; the notes are an opt-in text style (`--notes`).
"""),
        code("""
scheme = clf.index.scheme
for code_ in ("2.1.4", "1.1.4"):
    node = scheme.node(code_)
    print(("A " if node.auxiliary else "") + code_, "|", " > ".join(scheme.path_of(code_)))
    print("  text:", scheme.text_of(code_))
    print("  with notes:", scheme.text_of(code_, notes=True))
    if node.associated:
        print("  associated with principal sections", ", ".join(node.associated))
    print()
"""),
        md("""
## 7. Evaluation on 300 EUIPO marks

`evals/l3d_300.jsonl` holds 300 figurative EU trade marks from the Large Labelled
Logo Dataset (EUIPO open data, 1996 to 2020) with the codes EUIPO examiners
assigned, and the descriptions written by every vision model tried: Florence-2 (the
default), Qwen2.5-VL 7B through Ollama (two prompts) and SmolVLM (rejected). The
eval re-scores those cached descriptions, so it runs in seconds. A gold code
shallower than a level does not count at that level; about a fifth of the gold codes
are EUIPO extension codes that no WIPO edition contains, which caps the section
level.
"""),
        code("""
from image2vienna.eval import evaluate, load_cases

cases = load_cases(ROOT / "evals" / "l3d_300.jsonl")
rows = []
for key in sorted({k for c in cases for k in c.descriptions}):
    subset = [c for c in cases if key in c.descriptions]
    for chunking in ("whole", "mean"):
        result = evaluate(
            subset,
            lambda c, k=key, ch=chunking: clf.classify_text(
                c.descriptions[k], level="section", top_k=10, chunking=ch
            ),
            level="section",
            top_k=10,
        )
        for r in result.rows(ks=(1, 3, 10)):
            rows.append({"descriptions": key, "chunking": chunking, **r})
table = pd.DataFrame(rows)
percent = {c: "{:.1%}" for c in table.columns if c.startswith(("hit", "recall", "main"))}
table.style.format(percent | {"mrr": "{:.3f}"})
"""),
        md("""
Florence-2's captions score within a few points of the 7B model's inventories at a
thirtieth of the size, which is why it is the only default. The frequency baseline
(always answer the most frequent codes of the 770k L3D labels), the per-category
breakdown and every rejected variant are in `PERFORMANCE.md` and `docs/evals.md`;
`scripts/eval_sweep.py` prints them from the same cached descriptions.
"""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    return nb


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    target = NB_DIR / "01_image2vienna.ipynb"
    nbf.write(notebook_01(), target)
    print("wrote", target)
