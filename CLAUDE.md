# CLAUDE.md

Study repository for **image-to-Vienna**: given a trade mark image, return a ranked
list of Vienna Classification codes (category, division, section). Packaged as
`image2vienna`, CLI `i2vienna`. Sibling of `../study-text-to-ipc` (`text2ipc`): same
strategy, same conventions, same embedder, applied to images.

## What this repo optimizes for

1. **A tiny public interface.** `ViennaClassifier.classify(image, level, top_k)`,
   `classify_text(description, ...)`, `describe(image)` and the `i2vienna` CLI.
2. **Describe, then embed; no training.** A small vision model (Florence-2 base via
   transformers, the same the browser demo runs) writes a caption of the figurative
   elements; that prose is embedded and scored against the embedded classification
   entries with text2ipc's hierarchy heuristics (`docs/methodology.md`, ADR 0001).
   An Ollama-served model is an option, not the default. No fine-tuned classifier.
3. **The classification is embedded from its full path.** Category > division >
   section titles, never a title alone (ADR 0002). Notes are an opt-in text style.
4. **Editions are first class.** WIPO publishes an edition every few years; indexes
   are built per edition and reuse vectors whose text is unchanged.
5. **Evals from real office coding.** EUIPO figurative marks with examiner codes:
   the L3D sample (no credentials) and the Trademark Search API (ADR 0003).

## Layout

| Path | Purpose |
|---|---|
| `src/image2vienna/scheme/` | Download and parse WIPO's `full.xml`; codes; scheme table (Parquet) |
| `src/image2vienna/embeddings/` | Embedder protocol; sentence-transformers, Ollama and hash backends (from text2ipc) |
| `src/image2vienna/index/` | Parquet index tables, incremental build, discovery |
| `src/image2vienna/search/` | Hierarchical scoring (text2ipc's scorer on three levels) |
| `src/image2vienna/describe/` | Describer protocol; transformers (Florence-2, default) and Ollama backends, fixed backend for tests, prompts |
| `src/image2vienna/eval/` | Cases, hit-rate harness, description cache, L3D and EUIPO fetchers |
| `src/image2vienna/hf/` | Inference Endpoints handler (descriptions in, codes out) and model repository export |
| `src/image2vienna/web/` | Static Hugging Face Space export; `static/scorer.js` is a port of `search/scorer.py`, `static/vision-worker.js` runs a small vision model in the page |
| `src/image2vienna/classifier.py` | The public facade |
| `evals/` | JSONL eval cases with cached descriptions (committed; images are not) |
| `scripts/eval_sweep.py` | Scores every configuration from cached descriptions, prints Markdown |
| `scripts/make_notebooks.py` | Generates `notebooks/` (edit the script, not the `.ipynb`); executed in place with nbconvert |
| `PERFORMANCE.md` | The validation report at the root: dataset, protocol, numbers, analysis, next steps |
| `docs/` | methodology, evals log, usage, `adr/` |
| `data/` | (gitignored) `wipo/` XML, `scheme/` and `index/` Parquet, `images/`, `l3d/` |

## Conventions

- Python ≥3.11, managed with `uv`. Never call `pip`; use `uv sync` / `uv run`.
- Code, identifiers, docstrings and docs are **English**.
- Tests never download anything, never load a real model: the `hash:` embedder, the
  `fixed:` describer and `tests/fixtures/mini_vienna.xml`.
- Codes are canonical in WIPO's unpadded form (`1.1.2`); `normalize_code` accepts
  EUIPO's `01.01.02`. Auxiliary sections are ordinary section-level targets.
- The vision model runs once per eval case and the description is cached in the
  JSONL under the describer spec (plus `|<prompt>` for a prompt other than the
  model's default); scoring experiments reuse it. Keys so far: Florence-2 (default
  caption), `ollama:qwen2.5vl:7b|default` and `|inventory`, SmolVLM runs (rejected).
- Any new heuristic or prompt needs: a doc section, a parameter with a default, a
  test on the mini scheme, and a sweep run before and after with the numbers
  appended to `docs/evals.md`.
- Every user-visible change gets a line under `[Unreleased]` in `CHANGELOG.md`; a
  release moves them under a version heading, bumps `pyproject.toml` and
  `__version__`, and tags git and the Hub repositories (`scripts/hf_tag.sh`).
  Repository names are fixed: model `jaimenms/image2vienna-en`, Space
  `jaimenms/image2vienna`; a new index does not get a new repository.
- Notebooks are generated from `scripts/make_notebooks.py`, stay thin (logic lives in
  the package) and must execute top-to-bottom with the index built.
- `PERFORMANCE.md` is rewritten when the standard eval changes; `docs/evals.md` keeps
  every run in chronological order.
- The browser demo scores with `src/image2vienna/web/static/scorer.js`, a port of
  `search/scorer.py` (ADR 0004). A change to the scorer is finished only when the port
  matches and `tests/test_web.py` (Node parity on the mini scheme) still passes. The
  page never shows office codes next to results.
- Keep this repo consistent with `study-text-to-ipc`: same module names where the
  code is shared, same Parquet layout, same eval table format.

## Commands

```bash
uv sync --all-extras
uv run i2vienna build --edition 10                       # scheme table + index (4 s)
uv run i2vienna build --edition 10 --notes               # text style with notes
uv run i2vienna describe logo.png                        # stage one only (Florence-2, downloads once)
uv run i2vienna describe logo.png --describer ollama:qwen2.5vl:7b --prompt inventory   # an Ollama model instead
uv run i2vienna classify logo.png --level section        # both stages
echo "three stars above a crescent moon" | uv run i2vienna classify - --level auto
uv run i2vienna show 1.1.2
uv run i2vienna l3d --n 300                              # eval cases from L3D (no credentials)
EUIPO_CLIENT_ID=... EUIPO_CLIENT_SECRET=... uv run i2vienna euipo --n 300
uv run i2vienna describe-cases evals/l3d_300.jsonl       # vision model once, cached
uv run i2vienna eval evals/l3d_300.jsonl --level section
uv run python scripts/eval_sweep.py evals/l3d_300.jsonl
uv run python scripts/make_notebooks.py && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_image2vienna.ipynb
uv run python scripts/make_demo_examples.py             # evals/demo_images + demo_examples.jsonl
uv run i2vienna web-export space/image2vienna            # static Space (browser demo)
scripts/publish_space.sh                                 # export + upload the Space
uv run i2vienna hf-export hf/image2vienna-en             # model repo (handler + tables + package)
scripts/publish_hf.sh                                    # export + upload the model repo, tagged v<version>
uv run i2vienna download                                 # what an end user runs (from the Hub)
uv run pytest && uv run ruff check . && uv run ruff format .
```
