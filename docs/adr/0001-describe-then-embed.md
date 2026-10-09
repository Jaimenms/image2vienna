# 0001 Describe the image with a vision-language model, then classify the description

Status: accepted, 2026-10-09

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
