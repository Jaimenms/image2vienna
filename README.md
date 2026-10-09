# image2vienna

Map a trade mark image to a ranked list of Vienna Classification codes (the
International Classification of the Figurative Elements of Marks, WIPO). A
vision-language model describes what the image shows; that description is embedded
and scored against the embedded classification entries, each embedded once from its
full path (category > division > section), with the hierarchy heuristics of
[text2ipc](https://github.com/Jaimenms/text2ipc). No training.

Study repository: the package is meant for PyPI, the docs record every decision and
every measurement along the way. English scheme, edition 10 (in force 2026-01-01).

## Quick start

Try it without installing anything:
[huggingface.co/spaces/jaimenms/image2vienna](https://huggingface.co/spaces/jaimenms/image2vienna)
runs the embedder, the scoring and a small vision model in your browser (ADR 0004).

```bash
ollama pull qwen2.5vl:7b                                 # vision model, 6 GB
uv sync --all-extras
uv run i2vienna download                                 # prebuilt index from huggingface.co/jaimenms/image2vienna-en
uv run i2vienna classify logo.png --level section --top-k 5
```

`uv run i2vienna build --edition 10` builds the index from WIPO's XML instead (4 s
plus the embedder download). The same index runs as a Hugging Face Inference
Endpoint (a description in, codes out) from the repository
[jaimenms/image2vienna-en](https://huggingface.co/jaimenms/image2vienna-en).

```python
from image2vienna import ViennaClassifier

clf = ViennaClassifier("10")
for m in clf.classify("logo.png", level="section", top_k=5):
    print(m.pretty, round(m.score, 3), m.text)
print(clf.last_description)   # what the vision model saw
```

```
1.1.4   0.813  Celestial bodies, natural phenomena, geographical maps > Stars, comets > Three stars
1.7.6   0.805  Celestial bodies, natural phenomena, geographical maps > Moon > Crescent moon, half-moon
```

The description is a readable intermediate: `i2vienna describe logo.png` prints it,
and `echo "..." | i2vienna classify -` scores any text, so a corrected description
costs milliseconds.

## Does it work?

First validation, 2026-10-09, on 300 EUIPO figurative marks from the
[L3D dataset](https://doi.org/10.5281/zenodo.5771006) with the examiners' codes
(`docs/evals.md` has every run; `docs/adr/0003-eval-sources.md` the caveats: L3D
codes follow editions 5 to 8 and a fifth of them are EUIPO-only extension codes that
no WIPO edition contains, so section-level numbers are a floor):

| level | hit@1 | hit@3 | hit@10 | recall@10 | frequency baseline hit@1 / hit@10 |
|---|---|---|---|---|---|
| category | 44.3% | 68.7% | 72.7% | 54.2% | 32.0% / 85.3% |
| division | 33.0% | 50.7% | 60.3% | 40.0% | 26.0% / 61.3% |
| section | 9.8% | 20.2% | 33.0% | 13.0% | 13.1% / 39.7% |

It beats a frequency prior at rank 1 above the section and finds the right division
for most pictorial elements (73% of the gold divisions in Animals, 54% in Heraldry,
where the prior finds none); the prior wins at depth because EUIPO's most frequent
codes are conventions (letters in a special font, quadrilaterals, colours) that need no
image. `qwen2.5vl:7b` (the "heavy" model) under the `default` prompt, whole description,
titles-only texts; [PERFORMANCE.md](PERFORMANCE.md) has every configuration, the
per-category table and the "light" model the browser demo runs (SmolVLM-256M:
category hit@1 26.0%, division 14.0%, below the frequency baseline).

## Documentation

| Document | What it answers |
|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | Where it stands: the test dataset, the protocol, every configuration's numbers against a frequency baseline, and what the misses say |
| [docs/usage.md](docs/usage.md) | How to use it: install, build, classify, evaluate, EUIPO credentials |
| [docs/methodology.md](docs/methodology.md) | How it works: scheme texts, the two stages, scoring, evaluation protocol |
| [docs/evals.md](docs/evals.md) | What it scores: every eval run with its numbers, including rejected variants |
| [docs/adr/](docs/adr/) | Why it is built this way: describe-then-embed, scheme source and text, eval sources, the browser demo |
| [CHANGELOG.md](CHANGELOG.md) | What changed in each version |
| [CLAUDE.md](CLAUDE.md) | Conventions for contributors and coding agents |

## Repository layout

```
.
├── PERFORMANCE.md                the validation report: dataset, protocol, numbers, analysis
├── pyproject.toml                package metadata; `uv sync --all-extras` installs everything
├── src/image2vienna/
│   ├── __init__.py               public API: ViennaClassifier, classify, Match
│   ├── classifier.py             loads one index, one embedder and one vision model
│   ├── cli.py                    the `i2vienna` command line
│   ├── config.py                 levels, editions, default models, home directory
│   ├── scheme/                   WIPO full.xml -> nodes -> Parquet scheme table; code formats
│   ├── embeddings/               Embedder protocol: sentence-transformers, Ollama, hash (tests)
│   ├── index/                    Parquet index per (edition, lang, model, text style); incremental build
│   ├── search/scorer.py          cosine + path/subtree support, beam descent, auto level, distinct branches
│   ├── describe/                 Describer protocol: Ollama vision backend, fixed backend (tests), prompts
│   ├── eval/                     cases, hit-rate harness, description cache, L3D and EUIPO fetchers
│   ├── hf/                       Inference Endpoints handler and model repository export
│   └── web/                      static Space export; static/scorer.js is the port of scorer.py, vision-worker.js the in-browser model
├── scripts/eval_sweep.py         every scoring configuration from cached descriptions, Markdown out
├── scripts/make_notebooks.py     generates notebooks/ (edit this, not the .ipynb)
├── scripts/make_demo_examples.py drawn logos + EUIPO marks with cached descriptions for the demo
├── scripts/publish_space.sh      export + upload the browser demo (static Space)
├── scripts/publish_hf.sh         export + upload the index as a model repo (endpoint handler); hf_tag.sh tags both
├── notebooks/01_image2vienna.ipynb  walk-through: drawn logos, both stages, eval on the L3D sample
├── evals/l3d_300.jsonl           300 EUIPO marks: codes, verbal element, cached descriptions
├── evals/demo_examples.jsonl     the demo's example images (evals/demo_images/) and their descriptions
├── tests/                        35 tests; no network, no model, no Ollama (hash embedder, fixed describer);
│                                 js_parity.mjs runs scorer.js under Node against the Python scorer
├── docs/                         usage, methodology, evals, adr/
└── data/                         (gitignored) wipo/ XML, scheme/ index/ Parquet, images/, l3d/
```

## Browser demo

`i2vienna web-export space/image2vienna` assembles a static Hugging Face Space (ADR
0004): upload a logo or pick an example, the ONNX twin of the embedder scores the
description against the hierarchy in the tab, and the results are drawn as paths
through Vienna › category › division › section. Two vision models are named on the
page: **light** (SmolVLM-256M, runs in the browser; the examples show its description
by default) and **heavy** (Qwen2.5-VL 7B through Ollama, the package's model,
precomputed for the examples). `scripts/publish_space.sh` uploads it.

## Development

```bash
uv sync --all-extras
uv run i2vienna build --edition 10 && uv run i2vienna build --edition 10 --notes
uv run i2vienna l3d --n 300 && uv run i2vienna describe-cases evals/l3d_300.jsonl
uv run python scripts/eval_sweep.py evals/l3d_300.jsonl
uv run pytest && uv run ruff check . && uv run ruff format .
```

License: [MIT](LICENSE). Data sources: Vienna Classification by
[WIPO](https://www.wipo.int/web/classification-vienna/); evaluation images and codes
from EUIPO's open data through the
[Large Labelled Logo Dataset](https://github.com/lhf-labs/tm-dataset) (CC BY 4.0)
and the [EUIPO Trademark Search API](https://dev.euipo.europa.eu).
