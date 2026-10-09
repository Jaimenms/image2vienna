"""Assemble a static Hugging Face Space that classifies in the visitor's browser.

Static Spaces are free; Spaces that run Python are not. So the demo ships the index
as small static files and does everything client side, as text2ipc's demo does
(its ADR 0007): ``transformers.js`` embeds the description with the ONNX twin of the
embedder, ``scorer.js`` (a port of ``search/scorer.py``) ranks the entries, and a
Web Worker runs a small vision-language model to describe an uploaded image. The
example images ship with the description the package's vision model wrote for them,
so the demo shows the real pipeline on those and the in-browser model on uploads.

Layout written by :func:`export_web_demo`::

    index.html, app.js, scorer.js, vision-worker.js   the page (from ``web/static/``)
    manifest.json                     embedder, vision model, prompt, the index entry
    data/scheme.json                  columns code, level, parent, title, auxiliary
    data/vectors.bin                  per-row float32 scales, then int8 vectors (or float32)
    examples.json, examples/*.png     sample images and their cached descriptions
    README.md                         Space front matter (sdk: static) and a description
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .. import __version__ as package_version
from ..config import LEVELS, home
from ..describe.prompts import INVENTORY_PROMPT
from ..embeddings.st import prefixes_for
from ..index import ViennaIndex, available_indexes, scheme_table_path
from ..index.paths import IndexRef

#: The package default; 279 MB quantised in the browser.
WEB_DEFAULT_MODEL = "st:intfloat/multilingual-e5-base"

#: sentence-transformers model -> transformers.js model (ONNX weights on the Hub).
WEB_MODELS = {
    "intfloat/multilingual-e5-small": "Xenova/multilingual-e5-small",
    "intfloat/multilingual-e5-base": "Xenova/multilingual-e5-base",
    "intfloat/multilingual-e5-large": "Xenova/multilingual-e5-large",
}

#: Vision-language model the page runs in a Web Worker on an uploaded image.
#: SmolVLM-256M is the smallest one transformers.js runs (about 400 MB of ONNX weights
#: in these dtypes, WebGPU when the browser has it); far weaker than Qwen2.5-VL 7B,
#: which is why the examples ship with the package model's descriptions.
WEB_DEFAULT_VISION = "HuggingFaceTB/SmolVLM-256M-Instruct"
WEB_VISION_DTYPE = {"embed_tokens": "fp16", "vision_encoder": "fp16", "decoder_model_merged": "q4"}
WEB_VISION_MAX_NEW_TOKENS = 220

STATIC_FILES = ("index.html", "app.js", "scorer.js", "vision-worker.js")

SPACE_README = """---
title: image2vienna
emoji: 🖼️
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Trade mark image to Vienna Classification codes, in your browser
models:
{models}
---

# image2vienna, in the browser

Pick an example image or upload your own and get a ranked list of Vienna
Classification codes (the figurative elements of marks, WIPO, edition {edition}).
Nothing is sent to a server: the page downloads the quantised embedder `{web_model}`
({web_dtype}) and the index once, then embeds the description of the image and scores
it against the hierarchy locally. The description of an uploaded image is written in
the browser by `{vision_model}` (a small model; the package uses Qwen2.5-VL 7B through
Ollama, whose descriptions the example images carry); you can also edit the
description or write one yourself.

How it works, the evaluation numbers and the Python package are at
https://github.com/Jaimenms/image2vienna. Scoring: `scorer.js` is a port of
`image2vienna/search/scorer.py`; vectors are stored as int8 with a per-row scale.

Data: Vienna Classification by [WIPO](https://www.wipo.int/web/classification-vienna/).
{examples_note}
"""


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8: ``v ~= q * scale`` with ``scale = max|v| / 127``."""
    v = np.asarray(vectors, dtype=np.float32)
    scales = np.max(np.abs(v), axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(v / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def dequantize_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


def write_vectors(path: Path, vectors: np.ndarray, encoding: str = "int8") -> dict:
    """Write ``vectors.bin``: for int8, ``rows`` little-endian float32 scales followed by
    the ``rows x dim`` int8 matrix; for float32, the matrix alone."""
    rows, dim = vectors.shape
    if encoding == "int8":
        q, scales = quantize_int8(vectors)
        payload = scales.astype("<f4").tobytes() + np.ascontiguousarray(q).tobytes()
    elif encoding == "float32":
        payload = np.ascontiguousarray(vectors, dtype="<f4").tobytes()
    else:
        raise ValueError(f"encoding must be int8 or float32, got {encoding!r}")
    path.write_bytes(payload)
    return {"rows": rows, "dim": dim, "encoding": encoding, "bytes": len(payload)}


def scheme_columns(index: ViennaIndex) -> dict[str, list]:
    """The scheme table as parallel columns, rows in index order, parent as a row number."""
    return {
        "code": index.codes,
        "level": [int(x) for x in index.levels],
        "parent": [int(p) for p in index.parent_idx],
        "title": [n.title for n in index.nodes],
        "auxiliary": [int(n.auxiliary) for n in index.nodes],
    }


def web_model_for(model: str) -> str | None:
    backend, _, name = model.partition(":")
    if backend != "st":
        return None
    return WEB_MODELS.get(name, name)


def export_web_demo(
    out: Path,
    *,
    model: str = WEB_DEFAULT_MODEL,
    edition: str = "latest",
    lang: str = "EN",
    notes: bool = False,
    root: Path | None = None,
    repo_id: str = "<user>/image2vienna",
    web_model: str | None = None,
    web_dtype: str = "q8",
    encoding: str = "int8",
    examples: Path | None = None,
    vision_model: str | None = WEB_DEFAULT_VISION,
    prompt: str = INVENTORY_PROMPT,
) -> Path:
    """Write the static Space into ``out`` for one index.

    ``examples`` is a JSONL of demo cases (``scripts/make_demo_examples.py``): each has
    an ``image`` path relative to the JSONL, a ``title``, a ``source`` and cached
    ``descriptions``; the images are copied under ``examples/`` and one description
    per image reaches the page (the ``inventory`` one of the package's vision model
    when present, else the first).
    """
    from ..classifier import resolve_built_edition
    from ..embeddings.base import model_slug

    root = root or home()
    web_model = web_model or web_model_for(model)
    if web_model is None:
        raise ValueError(f"No browser model known for {model!r}; pass web_model explicitly")
    query_prefix, _ = prefixes_for(model.partition(":")[2])
    lang = lang.upper()

    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    built = resolve_built_edition(edition, lang, model, notes, root)
    ref = _find(root, built, lang, model_slug(model), notes)
    index = ViennaIndex.read(ref.path, scheme_table_path(built, lang, root))
    (out / "data" / "scheme.json").write_text(json.dumps(scheme_columns(index), ensure_ascii=False))
    vec = write_vectors(out / "data" / "vectors.bin", index.vectors, encoding)
    entry = {
        "edition": built,
        "lang": lang,
        "notes": notes,
        "rows": vec["rows"],
        "scheme": "data/scheme.json",
        "vectors": "data/vectors.bin",
        "encoding": encoding,
        "bytes": vec["bytes"],
    }

    manifest = {
        "generator": f"image2vienna {package_version}",
        "model": model,
        "web_model": web_model,
        "web_dtype": web_dtype,
        "query_prefix": query_prefix,
        "dim": index.meta.dim,
        "levels": list(LEVELS),
        "index": entry,
        "vision": (
            {
                "web_model": vision_model,
                "dtype": WEB_VISION_DTYPE,
                "max_new_tokens": WEB_VISION_MAX_NEW_TOKENS,
                "prompt": prompt,
            }
            if vision_model
            else None
        ),
        "repo_id": repo_id,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    static = Path(__file__).with_name("static")
    for name in STATIC_FILES:
        shutil.copy2(static / name, out / name)
    examples_note = ""
    if examples is not None:
        items = load_examples(examples, out / "examples")
        (out / "examples.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
        sources = sorted({e["source"] for e in items if e.get("source")})
        examples_note = "Example images: " + "; ".join(sources) + "."
    (out / "package.json").write_text(json.dumps({"type": "module", "private": True}) + "\n")
    (out / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    models = [web_model, *([vision_model] if vision_model else [])]
    (out / "README.md").write_text(
        SPACE_README.format(
            models="\n".join(f"  - {m}" for m in models),
            edition=built,
            web_model=web_model,
            web_dtype=web_dtype,
            vision_model=vision_model or "no in-browser model",
            examples_note=examples_note,
        )
    )
    return out


def load_examples(path: Path, target_dir: Path) -> list[dict]:
    """Copy each example's image under ``target_dir`` and return what the page shows:
    image path, title, source and one description. Gold codes never reach the page."""
    path = Path(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if not c.get("image") or not c.get("title"):
            raise ValueError(f"example {c.get('id')} needs an image and a title")
        src = (path.parent / c["image"]).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"example image {src} is missing")
        shutil.copy2(src, target_dir / src.name)
        descriptions = c.get("descriptions") or {}
        preferred = [k for k in descriptions if k.endswith("|inventory")]
        key = preferred[0] if preferred else (next(iter(descriptions)) if descriptions else None)
        out.append(
            {
                "image": f"{target_dir.name}/{src.name}",
                "title": c["title"],
                "source": c.get("source") or "",
                "description": descriptions.get(key, "") if key else "",
                "described_by": key.split("|")[0] if key else "",
            }
        )
    return out


def _find(root: Path, edition: str, lang: str, slug: str, notes: bool) -> IndexRef:
    for r in available_indexes(root):
        if r.edition == edition and r.lang == lang and r.model_slug == slug and r.notes == notes:
            return r
    raise FileNotFoundError(f"No index for edition {edition} {lang} {slug} notes={notes} in {root}")
