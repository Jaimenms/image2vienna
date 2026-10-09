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

All of 2 and 3, with one vision model: Florence-2 base (230M, a captioner driven by
task tokens; its detailed-caption task). The page runs its ONNX twin in a Web Worker
on uploads, and the package runs the same model through transformers (`hf:` backend,
the default) to write the example descriptions, so what the gallery shows is what the
browser produces (quantised weights can word it slightly differently). Two earlier
states the same day are worth recording: the page first showed Qwen2.5-VL 7B's
descriptions on the examples and SmolVLM-256M's after a click on Describe, which read
as one model changing its mind; then two named models, "light" and "heavy", with a
selector, until Florence-2 proved close to the 7B model on the evaluation and the
7B model was dropped for simplicity (ADR 0001, amendment). SmolVLM-256M and -500M
were rejected because they answered with one word, looped, echoed "trade" from
"trade mark image" or described the Earth's oceans on a crown above a word;
Florence-2 named the word, the symbol, the shield, the crown, the letters, the
bottle, the moon and the stars literally and never looped (`docs/evals.md`).
Repeated sentences are dropped from any model's output (`textnorm.clean_description`,
mirrored in the worker).

`i2vienna web-export <dir>` assembles the Space:

- **Vectors** go out as int8 with one float32 scale per row (`data/vectors.bin`,
  1.5 MB for 1,955 entries at 768 dimensions; cosine after dequantisation stays above
  0.999). **Scheme** goes out as columns `code, level, parent, title, auxiliary`
  (`data/scheme.json`, 115 KB); the page renders the path text from the titles as
  `SchemeTable.text_at` does.
- **Embedder**: `transformers.js` loads `Xenova/multilingual-e5-base` (q8, 279 MB)
  with the same `query: ` prefix as the Python backend.
- **Vision**: `vision-worker.js` loads Florence-2 in a Web Worker on first use
  (`Florence2ForConditionalGeneration` and its processor; the
  `<MORE_DETAILED_CAPTION>` task, greedy decoding, at most 120 tokens; fp16 on
  WebGPU with `shader-f16`, fp32 on WebGPU without it, q8/fp32 on WASM) and
  streams a token count while it generates; a chat-style model (SmolVLM) stays
  supported behind `kind: "chat"`. The visitor can edit the result or type a
  description.
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
  250 MB for the vision model, cached by the browser afterwards. On a machine
  without WebGPU the vision model runs on WASM and takes some seconds per image.
- What the Space publishes is derived data and public images only: WIPO's titles,
  quantised vectors, four drawn logos and six EUIPO marks from L3D (CC BY 4.0) with
  their cached descriptions. Gold codes never reach the page, as in text2ipc's demo.
- The page's descriptions are the package's: the same model, through transformers
  offline and through ONNX in the browser; the note next to the text box says which
  run wrote the text.
