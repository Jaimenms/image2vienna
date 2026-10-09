# Usage

## Requirements

- Python 3.11 to 3.13 and `uv`.
- Nothing else to install: the vision model (Florence-2 base, about 460 MB) and the
  embedder (`intfloat/multilingual-e5-base`, 1.1 GB) download from the Hub on first
  use and run through transformers on CPU or GPU.
- Optional: any vision model served by [Ollama](https://ollama.com), with
  `--describer ollama:<model> --prompt inventory` (an instruction model needs the
  instruction prompt); the evaluation used `qwen2.5vl:7b` as a 7B reference.

## Install and build

```bash
uv sync --all-extras
uv run i2vienna download                     # prebuilt index from huggingface.co/jaimenms/image2vienna-en
uv run i2vienna build --edition 10           # or build it: downloads WIPO's XML, embeds 1,955 entries
```

The model repository is also a custom Inference Endpoints handler: deploy it from
its Hub page and send `{"inputs": "<description of the figurative elements>",
"parameters": {"level": "section", "top_k": 5}}`; its card lists every parameter.

The scheme table and the index go to `data/` inside the repository (gitignored), or
to `$IMAGE2VIENNA_HOME`, or to `~/.cache/image2vienna` for installed users.

## Classify an image

```bash
uv run i2vienna classify logo.png                      # sections, top 10
uv run i2vienna classify logo.png --level division --top-k 5
uv run i2vienna classify logo.png --level auto         # as deep as the evidence goes
uv run i2vienna classify logo.png --padded             # codes as 01.01.02 (EUIPO style)
uv run i2vienna classify logo.png --json
```

The command prints the caption the vision model wrote, then the ranked entries with
the full path each one was embedded with. `--describer ollama:qwen2.5vl:7b --prompt
inventory` switches to a model served by Ollama, `--notes` picks the index built with
explanatory notes, `--principal-only` hides the auxiliary (A) sections, `--exclude
29` masks a category, `--chunking max` scores every entry by its best sentence of
the description.

Stage one and stage two separately:

```bash
uv run i2vienna describe logo.png                       # the prose only
echo "a lion's head above two crossed swords" | uv run i2vienna classify - --level auto
```

```python
from image2vienna import ViennaClassifier

clf = ViennaClassifier("10")                             # edition 10, English
for m in clf.classify("logo.png", level="section", top_k=5):
    print(m.pretty, m.auxiliary, round(m.score, 3), m.text)
print(clf.last_description)
clf.classify_text("three stars above a crescent moon", level="division")
```

## Evaluate

```bash
uv run i2vienna l3d --n 300                             # 300 EUIPO marks from L3D
uv run i2vienna describe-cases evals/l3d_300.jsonl      # vision model, 0.3 s per image on an Apple GPU
uv run i2vienna eval evals/l3d_300.jsonl                # hit@k per level
uv run python scripts/eval_sweep.py evals/l3d_300.jsonl # every configuration
```

With an EUIPO developer account (free, `https://dev.euipo.europa.eu`: create an
application and subscribe to the *Trademark search* product):

```bash
export EUIPO_CLIENT_ID=... EUIPO_CLIENT_SECRET=...
uv run i2vienna euipo --n 300 --query "markFeature==FIGURATIVE"
```

Images are stored under `data/images/` and never committed; the case files carry
codes, file names, the verbal element and the cached descriptions.

## Browser demo (static Hugging Face Space)

```bash
uv run python scripts/make_demo_examples.py      # drawn logos + six EUIPO marks with cached descriptions
uv run i2vienna web-export space/image2vienna --repo-id <user>/image2vienna
cd space/image2vienna && python -m http.server 8765   # open http://localhost:8765
scripts/publish_space.sh                          # needs `uv run hf auth login` once
```

The page lets a visitor pick an example or upload an image, writes the caption with
the ONNX twin of Florence-2 in a Web Worker (the examples carry the caption the same
model wrote offline), embeds it with the ONNX twin of the embedder, ranks with
`scorer.js` and draws the results as paths through the hierarchy (ADR 0004).
`--vision-model ""` ships the page without the in-browser model.
