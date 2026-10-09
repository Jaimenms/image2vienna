# 0001 Describe the image with a vision-language model, then classify the description

Status: accepted, 2026-10-09; amended the same evening (the model, see the end)

## Context

text2ipc maps a patent text to IPC symbols by embedding every classification entry
from its full ancestor path and scoring a query embedding against them, with
hierarchy heuristics on top and no training. The Vienna Classification of the
figurative elements of marks has the same shape (categories > divisions > sections,
titles written to be read in context, multi-label coding by offices), but the input
is an image. Three ways to bridge the gap were considered:

1. A multimodal embedder (CLIP-like) that puts images and the entry texts in one
   space. Cheap at query time, but the entry texts are abstract ("Other
   representations of stars", "Men wearing a dinner jacket or a suit") and CLIP-style
   models align images with captions, not with taxonomies.
2. A vision-language model asked to pick codes directly from the classification.
   Needs the 1,955 entries in the prompt or a fine-tuned model; opaque; costly.
3. A vision-language model that *describes* the figurative elements in prose, and
   text2ipc's embedding stage on that prose.

## Decision

Option 3. `Qwen2.5-VL` (7B, served by Ollama) describes the image under a prompt that
asks for an inventory of the visible elements in the vocabulary of the classification
(objects, beings, plants, celestial bodies, heraldry, geometric figures, letters,
colours) and forbids brand guessing. The description is the query; the embedding and
scoring code is text2ipc's, ported to three levels. Both stages are swappable behind
small protocols (`Describer`, `Embedder`), and descriptions are cached in the eval
files so that scoring experiments never rerun the vision model.

## Consequences

- The vision stage is the slow part: about 3.5 s per image on an Apple M-series GPU,
  against milliseconds for embedding and scoring. The index itself is tiny (1,955
  rows, 7.7 MB) and builds in four seconds.
- The description is a readable intermediate: a wrong code can be traced to what the
  model saw or failed to see, and a user can edit the description and re-run the
  cheap stage (`i2vienna classify -` reads a text from stdin).
- Mistakes compound: elements the model does not mention cannot be coded. The
  prompt is therefore part of the measured configuration (`docs/evals.md`).
- A multimodal embedder stays possible as an `Embedder` backend if a later
  comparison warrants it.

## Amendment, 2026-10-09 evening: a 230M captioner instead of a 7B instruction model

The validation ran with Qwen2.5-VL 7B through Ollama (6 GB, a GPU or an M-series Mac
with 16 GB, about 5 s per image). For the browser demo a small model was needed, and
Florence-2 base (230M, a captioner driven by a task token, through transformers on
CPU or GPU, 0.3 s per image) turned out to score within a few points of the 7B model
on the same 300 marks (category hit@1 37.0% against 44.3%, hit@3 69.7% against 68.7%;
division 31.0% against 33.0% and 53.3% against 50.7%; `PERFORMANCE.md`). Its captions
are literal and never loop, where the small instruction models tried (SmolVLM-256M
and -500M) hallucinated or looped. Florence-2 is therefore the only default: no
Ollama, no 6 GB download, one model in the package and in the page. The `ollama:`
backend stays as an option for any served vision model, with the instruction prompts
it needs.
