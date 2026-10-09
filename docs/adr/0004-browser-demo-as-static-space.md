# 0004 Browser demo as a static Hugging Face Space, with a small vision model in the page

Status: accepted, 2026-10-09

## Context

text2ipc's demo (its ADR 0007) is a static Space: free to keep up, nothing leaves the
visitor's browser, `transformers.js` embeds the query and a port of the scorer ranks
the entries. The same is wanted here, with two additions the user asked for: an
image upload and a gallery of example images, and the results drawn as a graph of
paths through the hierarchy as in text2ipc.

The difference is the first stage. The package describes images with Qwen2.5-VL 7B
through Ollama, which no browser runs. Options for the page:

1. Call a hosted vision model. Needs a token in the page or a server, and the image
   leaves the browser; against the point of a static Space.
2. Run a small vision-language model in the browser with `transformers.js`:
   SmolVLM-256M-Instruct (about 400 MB of ONNX weights, WebGPU when available, WASM
   otherwise) is the smallest one the library runs; its descriptions are far
   shallower than Qwen's.
3. Let the visitor write or paste the description: the second stage alone.

## Decision

All of 2 and 3, and the example images carry the description Qwen2.5-VL 7B wrote for
them (`scripts/make_demo_examples.py` caches it in `evals/demo_examples.jsonl`), so
the gallery shows the real pipeline while an upload shows the in-browser model.

`i2vienna web-export <dir>` assembles the Space:

- **Vectors** go out as int8 with one float32 scale per row (`data/vectors.bin`,
  1.5 MB for 1,955 entries at 768 dimensions; cosine after dequantisation stays above
  0.999). **Scheme** goes out as columns `code, level, parent, title, auxiliary`
  (`data/scheme.json`, 115 KB); the page renders the path text from the titles as
  `SchemeTable.text_at` does.
- **Embedder**: `transformers.js` loads `Xenova/multilingual-e5-base` (q8, 279 MB)
  with the same `query: ` prefix as the Python backend.
- **Vision**: `vision-worker.js` loads `HuggingFaceTB/SmolVLM-256M-Instruct` in a
  Web Worker on first use (`AutoProcessor`, `AutoModelForVision2Seq`, the
  `inventory` prompt, greedy decoding, at most 220 tokens) and streams a token
  count while it generates. The visitor can edit the result or type a description.
- **Scoring**: `scorer.js` is a line-by-line port of `search/scorer.py` on three
  levels, with `principalOnly` and `excludeCodes` (the page offers "No colours" and
  "Principal sections only"). `tests/test_web.py` exports the mini scheme with
  float32 vectors, runs the port under Node and requires the same codes in the
  same order, scores within 1e-5, plus the sentence splitter and the query
  normalisation.
- **Graph**: the result paths merged into one tree, Vienna › category › division ›
  section, similarities on the edges, result edges labelled with the score, titles on
  hover; the same drawing as text2ipc's with codes as node labels.
- `scripts/publish_space.sh` creates the Space and uploads with `hf upload`.

## Consequences

- The scorer exists twice; a change to `scorer.py` is not done until `scorer.js`
  matches and the parity test covers the new behaviour (the text2ipc rule).
- A first visit downloads about 280 MB (embedder, index, page); an upload adds about
  400 MB for the vision model, cached by the browser afterwards. On a machine
  without WebGPU the vision model runs on WASM and takes a minute per image.
- What the Space publishes is derived data and public images only: WIPO's titles,
  quantised vectors, four drawn logos and six EUIPO marks from L3D (CC BY 4.0) with
  their cached descriptions. Gold codes never reach the page, as in text2ipc's demo.
- The demo's in-browser descriptions are not the package's; the page says so next to
  the text box, and the evals (`PERFORMANCE.md`) measure the package, not the page.
