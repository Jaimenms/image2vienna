"""The static browser demo: export layout, int8 vectors, and parity between the
JavaScript scorer and the Python one (run under Node when it is installed)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from image2vienna.search import Beam, SearchParams, search
from image2vienna.textnorm import normalize_query, split_sentences
from image2vienna.web import dequantize_int8, export_web_demo, quantize_int8

NODE = shutil.which("node")
PARITY = Path(__file__).with_name("js_parity.mjs")


def _export(mini_home, tmp_path, **kwargs):
    return export_web_demo(
        tmp_path / "space",
        model="hash:64",
        edition="10",
        root=mini_home,
        web_model="test/model",
        **kwargs,
    )


def test_int8_roundtrip_keeps_cosine(mini_index):
    q, scales = quantize_int8(mini_index.vectors)
    back = dequantize_int8(q, scales)
    cos = np.sum(back * mini_index.vectors, axis=1) / np.linalg.norm(back, axis=1)
    assert q.dtype == np.int8 and scales.shape == (len(mini_index),)
    assert cos.min() > 0.999


def test_export_layout(mini_home, tmp_path):
    out = _export(mini_home, tmp_path)
    for name in (
        "index.html",
        "app.js",
        "scorer.js",
        "vision-worker.js",
        "README.md",
        "manifest.json",
    ):
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["web_model"] == "test/model" and manifest["dim"] == 64
    assert manifest["query_prefix"] == "" and manifest["levels"] == [
        "category",
        "division",
        "section",
    ]
    vision = manifest["vision"]
    assert vision["light"]["web_model"].startswith("HuggingFaceTB/") and vision["light"]["prompt"]
    assert vision["heavy"]["spec"].startswith("ollama:")
    entry = manifest["index"]
    assert entry["edition"] == "10" and entry["lang"] == "EN" and entry["notes"] is False
    scheme = json.loads((out / entry["scheme"]).read_text())
    rows = entry["rows"]
    assert set(scheme) == {"code", "level", "parent", "title", "auxiliary"}
    assert all(len(scheme[c]) == rows for c in scheme)
    assert scheme["parent"][0] == -1 and all(p < i for i, p in enumerate(scheme["parent"]))
    assert scheme["auxiliary"][scheme["code"].index("1.1.2")] == 1
    assert (out / entry["vectors"]).stat().st_size == rows * 4 + rows * 64
    readme = (out / "README.md").read_text()
    assert "sdk: static" in readme
    assert "  - Qwen/Qwen2.5-VL-7B-Instruct\n" in readme and "  - test/model\n" in readme
    short = next(line for line in readme.splitlines() if line.startswith("short_description:"))
    assert len(short.split(":", 1)[1].strip()) <= 60  # the Hub rejects longer ones


def test_export_without_vision_model(mini_home, tmp_path):
    out = _export(mini_home, tmp_path, vision_model=None)
    assert json.loads((out / "manifest.json").read_text())["vision"] is None


def test_export_copies_examples_without_gold(mini_home, tmp_path):
    from PIL import Image

    (tmp_path / "imgs").mkdir()
    Image.new("RGB", (8, 8), "white").save(tmp_path / "imgs" / "star.png")
    cases = tmp_path / "demo.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "demo:star",
                "image": "imgs/star.png",
                "title": "One star",
                "vienna": ["1.1.2"],
                "source": "drawn",
                "descriptions": {
                    "ollama:qwen2.5vl:7b": "a star",
                    "ollama:qwen2.5vl:7b|inventory": "One star, yellow.",
                    "hf:HuggingFaceTB/SmolVLM-256M-Instruct|light": "A yellow star.",
                },
            }
        )
        + "\n"
    )
    out = _export(mini_home, tmp_path, examples=cases)
    (ex,) = json.loads((out / "examples.json").read_text())
    assert ex == {
        "image": "examples/star.png",
        "title": "One star",
        "source": "drawn",
        "descriptions": {
            "light": {
                "text": "A yellow star.",
                "model": "hf:HuggingFaceTB/SmolVLM-256M-Instruct",
                "prompt": "light",
            },
            "heavy": {
                "text": "One star, yellow.",
                "model": "ollama:qwen2.5vl:7b",
                "prompt": "inventory",
            },
        },
    }
    assert (out / "examples" / "star.png").exists()
    assert "Example images: drawn." in (out / "README.md").read_text()


def test_export_rejects_unknown_browser_model(mini_home, tmp_path):
    with pytest.raises(ValueError, match="browser model"):
        export_web_demo(tmp_path / "space", model="hash:64", edition="10", root=mini_home)


QUERIES = [
    "three stars",
    "crescent moon",
    "circles ellipses",
    "men heads busts",
    "stars comets compass",
    "acrobats athletes dancers sport",
    "inscriptions arabic characters",
]
PARAMS = [
    SearchParams(),
    SearchParams(top_k=3),
    SearchParams(level="category"),
    SearchParams(level="division"),
    SearchParams(level="section", gap=0.0),
    SearchParams(level="section", beam=Beam(category=1, division=1)),
    SearchParams(level="auto"),
    SearchParams(level="auto", auto_margin=0.0),
    SearchParams(dedupe_branches=False),
    SearchParams(principal_only=True),
    SearchParams(exclude_codes=("1",)),
    SearchParams(level="division", exclude_codes=("1.1", "26")),
    SearchParams(
        weights=__import__("image2vienna.search", fromlist=["Weights"]).Weights(0.6, 0.2, 0.2)
    ),
]


def _js_params(p: SearchParams) -> dict:
    return {
        "level": p.level,
        "topK": p.top_k,
        "gap": p.gap,
        "autoMargin": p.auto_margin,
        "autoRoots": p.auto_roots,
        "dedupeBranches": p.dedupe_branches,
        "principalOnly": p.principal_only,
        "excludeCodes": list(p.exclude_codes),
        "weights": asdict(p.weights),
        "beam": {"category": p.beam.category, "division": p.beam.division},
    }


def _run_parity(out: Path, cases: list, tmp_path: Path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    run = subprocess.run(
        [NODE, str(PARITY), str(out), str(cases_path)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_scorer_matches_python(mini_home, mini_index, embedder, tmp_path):
    out = _export(mini_home, tmp_path, encoding="float32")
    cases, expected = [], []
    for text in QUERIES:
        vec = embedder.embed_query(text)
        for p in PARAMS:
            cases.append({"query": vec.tolist(), "params": _js_params(p)})
            expected.append(search(mini_index, vec, p))
    stack = embedder.embed_queries(QUERIES[:3])  # sentence chunking "max": best sentence
    for p in (SearchParams(), SearchParams(level="division"), SearchParams(level="auto")):
        cases.append({"query": stack.tolist(), "params": _js_params(p)})
        expected.append(search(mini_index, stack, p))
    got = _run_parity(out, cases, tmp_path)
    assert len(got) == len(expected)
    for case, py, js in zip(cases, expected, got, strict=True):
        label = f"{case['params']}"
        assert [m.code for m in py] == [m["code"] for m in js], label
        for a, b in zip(py, js, strict=True):
            assert (a.pretty, a.padded, a.level, a.title, a.text, a.auxiliary) == (
                b["pretty"],
                b["padded"],
                b["level"],
                b["title"],
                b["text"],
                b["auxiliary"],
            ), label
            for field in ("score", "similarity", "path_support", "subtree_support"):
                assert abs(getattr(a, field) - b[field]) < 1e-5, (label, field)


TEXTS = [
    "Three stars. Below, a crescent moon! Red and blue.",
    "A WORD IN CAPITALS ONLY. Then more.",
    "  spaced   out\n\nwith   breaks  ",
    "One sentence without end",
    'He said "Go". (Then left.) 3 stars.',
]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_text_helpers_match_python(mini_home, tmp_path):
    out = _export(mini_home, tmp_path)
    cases = [{"split": t} for t in TEXTS] + [{"normalize": t} for t in TEXTS]
    expected = [split_sentences(t) for t in TEXTS] + [normalize_query(t) for t in TEXTS]
    assert _run_parity(out, cases, tmp_path) == expected
