# Performance report: image2vienna, first validation

As of 2026-10-09. Every number here comes from `scripts/eval_sweep.py` over
`evals/l3d_300.jsonl`; `docs/evals.md` keeps the full log, including rejected variants.

## Summary

**A 230M captioner does the job a 7B model did, and the approach works where a
description can work.** The pipeline's vision model is Florence-2 base (230M
parameters, through transformers on CPU or GPU, 0.3 s per image on an Apple GPU);
its literal captions, embedded as one query against titles-only path texts, beat the
frequency baseline at rank 1 above the section level (category hit@1 37.0% against
32.0%, division 31.0% against 26.0%) and find the right division for most marks
whose elements are pictorial. The baseline wins at depth (section hit@10 30.6%
against 39.7%), because EUIPO's most frequent codes are boilerplate the prior answers
without looking (letters in a special form of writing, quadrilaterals, colours), and
because a fifth of the gold sections are EUIPO extension codes absent from the
edition indexed.

| level | n | hit@1 | hit@3 | hit@10 | recall@10 | baseline hit@1 | baseline hit@10 |
|---|---|---|---|---|---|---|---|
| category | 300 | 37.0% | 69.7% | 74.3% | 58.3% | 32.0% | 85.3% |
| division | 300 | 31.0% | 53.3% | 62.7% | 42.5% | 26.0% | 61.3% |
| section | 297 | 11.1% | 20.5% | 30.6% | 11.2% | 13.1% | 39.7% |

The study was run with Qwen2.5-VL 7B through Ollama as the vision model (6 GB, a
GPU or an M-series Mac), under two prompts; the configuration sweep below uses its
descriptions. Florence-2, tried for the browser demo, scored within a few points of
it (7 behind at category hit@1, level or ahead from hit@3 on), so the small model
became the only default and the 7B model an option (`--describer ollama:...`). The
candidate-recall ceiling says how much a second stage could add: with 50 candidates
the gold section is present about half the time, the gold division three times in
four.

## The pipeline under test

```
WIPO full.xml (Vienna 10, EN) ─> path texts "category > division > section" ─> e5-base vectors (index, 1,955 rows)
                                                                                        │
trade mark image ─> Florence-2 base: caption of the figurative elements ─> query vector ─> scoring ─> ranked codes
```

The only new part, compared with text2ipc, is the vision stage: Florence-2 writes a
few sentences naming the objects, letters and colours, and that prose is the query.
The index is the same construction as in text2ipc (one vector per entry, embedded
from its full path of titles, `intfloat/multilingual-e5-base`), so a description is
re-scored in milliseconds and a hand-written description is a valid input
(`docs/methodology.md`, ADR 0001 and 0002).

## Test dataset: 300 EUIPO figurative marks from L3D

The test set is `evals/l3d_300.jsonl`: 300 figurative EU trade marks with the Vienna
codes EUIPO examiners assigned, drawn from the Large Labelled Logo Dataset (L3D;
Gutiérrez-Fandiño, Pérez-Fernández and Armengol-Estapé, 2021, CC BY 4.0,
[doi:10.5281/zenodo.5771006](https://doi.org/10.5281/zenodo.5771006),
[github.com/lhf-labs/tm-dataset](https://github.com/lhf-labs/tm-dataset)). L3D holds
about 770k marks taken from EUIPO's open data between 1996 and 2020, resized to
256x256 pixels with white padding.

The sample was drawn without downloading the 12 GB archive: the label file
`results.json` (769,674 records) sits in its last 120 MB and the images sit at its
front in random UUID order, so the first 48 MB (3,336 images) and the last 120 MB were
fetched with HTTP range requests and 300 labelled images were picked with seed 0
(`i2vienna l3d --n 300`). The repository keeps the codes, file names, verbal elements
and cached descriptions; the images stay under `data/images/` and out of git.

| Fact | Value |
|---|---|
| Marks | 300 |
| Gold codes | 907 (1 to 18 per mark, median 2) |
| Section-level / division-level gold codes | 901 / 6 |
| Largest categories in the gold (codes) | 26 geometric figures (280), 27 forms of writing (142), 29 colours (85), 1 celestial bodies (63), 24 heraldry (61), 25 ornamental motifs (59), 5 plants (56), 3 animals (49) |
| Gold codes absent from WIPO edition 10 | 179 (19.7%): EUIPO extension codes such as 1.1.99, 25.1.95, 29.1.98 |
| Filing years | 1996 to 2020 |
| Image size | 256x256, white padding |

Three limitations bound what the numbers can show:

- The codes follow editions 5 to 8 while the index is edition 10 (over a hundred
  section changes between editions 8 and 9 alone), so section-level agreement is
  approximate; categories and divisions are stable across editions.
- The EUIPO-only extension codes can never be hit at section level, which caps
  section hit rates well below 100%.
- 256-pixel images hide small lettering and detail from the vision model.

## Protocol and metrics

Each image is described once per prompt by the vision model and the description is
cached in the case file (`i2vienna describe-cases`); every scoring configuration is
then evaluated from that cache in seconds, so the tables compare scoring choices on
identical descriptions.

| Component | Setting under test |
|---|---|
| Classification | Vienna edition 10, English, from WIPO's `full.xml`: 29 categories, 145 divisions, 845 principal and 936 auxiliary sections (1,955 entries) |
| Entry text | Titles chained top-down; a second index appends the "Including ..." notes |
| Embedder | `intfloat/multilingual-e5-base`, 768 dimensions; the 1,955 entries embed in 3.7 s |
| Vision model | Florence-2 base (`hf:florence-community/Florence-2-base-ft`, transformers, detailed-caption task, greedy, at most 220 tokens; 0.3 s per image on an Apple M5 Pro). Reference for the sweep: Qwen2.5-VL 7B through Ollama (`qwen2.5vl:7b`, temperature 0, at most 400 tokens, 5 to 6 s per image) |
| Prompts (Qwen) | `default` (an enumeration of element kinds to look for); `inventory` (one sentence per element present, how letters are written, colours last, no mention of absent kinds) |
| Scoring | Cosine over the entry vectors; text2ipc's hierarchy heuristics as options: path and subtree support, beam descent, auxiliary-section masking, category masking, sentence-level chunking (`whole`, `mean`, `max`) |

Metrics are computed per level (category, division, section) over the top-10
predictions truncated to that level. A gold code shallower than the level does not
count there, so the section level scores 297 of the 300 marks.

| Metric | Meaning |
|---|---|
| hit@k | any gold code is among the top-k |
| recall@k | share of the mark's gold codes among the top-k (one code per figurative element) |
| main@1 | the office's first-listed code is at rank 1 |
| MRR | mean reciprocal rank of the first gold code |

Two reference points frame the model. The **frequency baseline** always answers the
most frequent codes of the whole 770k-label set (27.5.1 "Letters presenting a special
form of writing" alone is on 12.6% of all EUIPO figurative marks); a system has to beat
it to be worth running. The **candidate-recall ceiling**, hit@50, says how often the
gold is anywhere in a long first-stage list, which is what a second-stage judge could
recover.

## Results

The configuration sweep uses edition 10, `multilingual-e5-base`, top 10, and the
descriptions of the reference model `qwen2.5vl:7b` under its `default` prompt (one
vision run, cached); Florence-2's own numbers are in "The vision model" below. "titles" means the
path text of titles only; "+ notes" the index whose texts append the "Including ..."
notes; "whole" embeds the description as one query, "sentence mean" the mean of its
sentence vectors, "sentence max" scores every entry by its best sentence; "path 0.3"
and "subtree 0.3" are the hierarchy supports of text2ipc; "principal only" leaves the
auxiliary (A) sections out; "no colours (29)" and "pictorial only (no 26-29)" mask
those categories from both candidates and gold, so their `n` is smaller.

### Per level, every configuration

**category** (n = 300)

| configuration | hit@1 | hit@3 | hit@10 | recall@10 | main@1 | MRR |
|---|---|---|---|---|---|---|
| frequency baseline | 32.0% | 65.7% | 85.3% | 75.0% | - | - |
| titles, whole description | 44.3% | 68.7% | 72.7% | 54.2% | 26.7% | 0.563 |
| titles + notes, whole | 43.7% | 69.7% | 73.3% | 54.3% | 26.0% | 0.562 |
| titles, sentence mean | 41.7% | 69.3% | 73.0% | 54.3% | 25.3% | 0.546 |
| titles, sentence max | 36.7% | 67.3% | 72.7% | 54.5% | 20.3% | 0.518 |
| titles + notes, sentence max | 35.3% | 67.0% | 74.0% | 55.6% | 19.7% | 0.515 |
| titles, whole, path 0.3 | 46.3% | 65.0% | 67.3% | 47.3% | 27.0% | 0.555 |
| titles, whole, subtree 0.3 | 44.3% | 68.7% | 72.7% | 54.2% | 26.7% | 0.563 |
| titles, sentence max, path 0.3 | 38.0% | 64.3% | 68.7% | 50.0% | 20.0% | 0.513 |
| titles, whole, principal only | 36.0% | 61.7% | 65.0% | 44.9% | 17.3% | 0.484 |
| titles, whole, no colours (29) | 48.8% | 69.2% | 72.2% | 55.5% | 35.1% | 0.588 |
| titles, sentence max, no colours (29) | 40.1% | 66.9% | 71.2% | 56.4% | 27.8% | 0.531 |
| titles, whole, pictorial only (no 26-29) | 42.0% | 63.0% | 70.7% | 58.6% | 31.5% | 0.528 |
| titles, sentence mean, pictorial only (no 26-29) | 38.1% | 55.8% | 62.4% | 49.5% | 24.9% | 0.471 |

**division** (n = 300)

| configuration | hit@1 | hit@3 | hit@10 | recall@10 | main@1 | MRR |
|---|---|---|---|---|---|---|
| frequency baseline | 26.0% | 44.3% | 61.3% | 50.0% | - | - |
| titles, whole description | 33.0% | 50.7% | 60.3% | 40.0% | 18.3% | 0.429 |
| titles + notes, whole | 32.7% | 51.3% | 60.3% | 40.4% | 18.0% | 0.429 |
| titles, sentence mean | 28.3% | 47.7% | 58.3% | 39.5% | 17.0% | 0.388 |
| titles, sentence max | 26.0% | 48.7% | 59.3% | 40.3% | 14.0% | 0.385 |
| titles + notes, sentence max | 25.0% | 49.3% | 59.7% | 40.8% | 13.3% | 0.383 |
| titles, whole, path 0.3 | 32.3% | 46.7% | 53.3% | 33.9% | 17.3% | 0.403 |
| titles, whole, subtree 0.3 | 33.0% | 50.7% | 60.3% | 40.0% | 18.3% | 0.429 |
| titles, sentence max, path 0.3 | 27.3% | 47.3% | 53.7% | 34.8% | 14.7% | 0.377 |
| titles, whole, principal only | 27.0% | 43.3% | 50.7% | 29.4% | 10.7% | 0.357 |
| titles, whole, no colours (29) | 30.8% | 46.8% | 57.2% | 40.7% | 22.4% | 0.401 |
| titles, sentence max, no colours (29) | 25.4% | 43.5% | 53.5% | 38.9% | 18.4% | 0.359 |
| titles, whole, pictorial only (no 26-29) | 24.9% | 41.4% | 51.9% | 39.8% | 17.7% | 0.342 |
| titles, sentence mean, pictorial only (no 26-29) | 21.0% | 33.7% | 43.6% | 31.1% | 14.4% | 0.286 |

**section** (n = 297)

| configuration | hit@1 | hit@3 | hit@10 | recall@10 | main@1 | MRR |
|---|---|---|---|---|---|---|
| frequency baseline | 13.1% | 21.5% | 39.7% | 22.8% | - | - |
| titles, whole description | 9.8% | 20.2% | 33.0% | 13.0% | 3.7% | 0.167 |
| titles + notes, whole | 9.8% | 20.2% | 33.0% | 12.9% | 3.7% | 0.166 |
| titles, sentence mean | 7.7% | 12.8% | 23.6% | 9.5% | 2.4% | 0.121 |
| titles, sentence max | 8.1% | 16.8% | 28.3% | 11.2% | 2.4% | 0.136 |
| titles + notes, sentence max | 8.1% | 17.2% | 28.6% | 11.8% | 2.4% | 0.138 |
| titles, whole, path 0.3 | 10.1% | 18.5% | 31.0% | 12.5% | 4.0% | 0.160 |
| titles, whole, subtree 0.3 | 9.8% | 20.2% | 33.0% | 13.0% | 3.7% | 0.167 |
| titles, sentence max, path 0.3 | 7.7% | 15.5% | 25.6% | 10.2% | 2.0% | 0.130 |
| titles, whole, principal only | 8.1% | 18.2% | 28.6% | 11.3% | 3.4% | 0.146 |
| titles, whole, no colours (29) | 7.8% | 15.9% | 27.4% | 12.6% | 4.4% | 0.131 |
| titles, sentence max, no colours (29) | 6.8% | 11.8% | 22.0% | 9.8% | 3.7% | 0.106 |
| titles, whole, pictorial only (no 26-29) | 8.3% | 19.3% | 30.4% | 19.2% | 4.4% | 0.149 |
| titles, sentence mean, pictorial only (no 26-29) | 6.1% | 13.3% | 24.3% | 13.5% | 2.2% | 0.113 |

### Candidate-recall ceiling (titles, top 50)

| query | category hit@50 | division hit@50 | section hit@50 | section recall@50 |
|---|---|---|---|---|
| whole | 81.7% | 75.3% | 53.9% | 27.5% |
| max | 84.0% | 73.7% | 45.1% | 23.3% |

### Gold codes found in the top 10, by category (titles, whole description)

Share of the gold codes of each category found among the top 10, at section level
and after truncation to the division, for the model and for the frequency baseline.
Categories with fewer than 8 gold codes are left out.

| category | gold codes | section: model | section: baseline | division: model | division: baseline |
|---|---|---|---|---|---|
| 26 Geometrical figures and solids | 280 | 13% | 27% | 46% | 81% |
| 27 Forms of writing, numerals | 142 | 2% | 36% | 28% | 94% |
| 29 Colours | 85 | 35% | 49% | 79% | 100% |
| 1 Celestial bodies, natural phenomena, g | 63 | 11% | 0% | 32% | 0% |
| 24 Heraldry, coins, emblems, symbols | 61 | 11% | 0% | 54% | 0% |
| 25 Ornamental motifs, surfaces or backgro | 59 | 0% | 0% | 0% | 39% |
| 5 Plants | 56 | 29% | 0% | 32% | 41% |
| 3 Animals | 49 | 29% | 0% | 73% | 0% |
| 2 Human beings | 32 | 12% | 0% | 34% | 0% |
| 4 Supernatural, fabulous, fantastic or u | 13 | 0% | 0% | 8% | 0% |
| 7 Constructions, structures for advertis | 9 | 11% | 0% | 22% | 0% |
| 19 Containers and packing, representation | 9 | 11% | 0% | 33% | 0% |

### Inventory prompt (Qwen2.5-VL 7B)

The `inventory` prompt (one sentence per element present, how letters are written,
colours last, no mention of absent kinds of elements) was run over the same 300
images (27 min) and scored with the same index. It lifts the lists (hit@3 and hit@10)
and costs a little at rank 1; on the pictorial categories alone it is the best
section-level configuration of the study.

| level | configuration | hit@1 | hit@3 | hit@10 | recall@10 | main@1 | MRR |
|---|---|---|---|---|---|---|---|
| category | frequency baseline | 32.0% | 65.7% | 85.3% | 75.0% | - | - |
| category | titles, whole description | 44.3% | 68.7% | 72.7% | 54.2% | 26.7% | 0.563 |
| category | titles, inventory prompt, whole | 41.7% | 71.7% | 74.3% | 56.0% | 24.7% | 0.559 |
| category | titles, inventory prompt, sentence mean | 39.3% | 59.0% | 69.0% | 51.5% | 23.7% | 0.502 |
| category | titles, inventory prompt, sentence max | 35.0% | 68.0% | 72.3% | 52.8% | 18.3% | 0.506 |
| category | titles, inventory prompt, pictorial only (no 26-29) | 44.2% | 63.0% | 67.4% | 54.6% | 32.0% | 0.531 |
| division | frequency baseline | 26.0% | 44.3% | 61.3% | 50.0% | - | - |
| division | titles, whole description | 33.0% | 50.7% | 60.3% | 40.0% | 18.3% | 0.429 |
| division | titles, inventory prompt, whole | 32.3% | 55.7% | 63.7% | 42.7% | 18.7% | 0.444 |
| division | titles, inventory prompt, sentence mean | 31.3% | 44.3% | 54.7% | 37.6% | 19.0% | 0.389 |
| division | titles, inventory prompt, sentence max | 30.7% | 52.3% | 59.7% | 40.6% | 16.3% | 0.416 |
| division | titles, inventory prompt, pictorial only (no 26-29) | 29.3% | 45.9% | 56.9% | 43.5% | 21.0% | 0.383 |
| section | frequency baseline | 13.1% | 21.5% | 39.7% | 22.8% | - | - |
| section | titles, whole description | 9.8% | 20.2% | 33.0% | 13.0% | 3.7% | 0.167 |
| section | titles, inventory prompt, whole | 9.8% | 19.9% | 32.3% | 13.0% | 3.0% | 0.162 |
| section | titles, inventory prompt, sentence mean | 6.4% | 14.1% | 24.9% | 10.5% | 1.0% | 0.116 |
| section | titles, inventory prompt, sentence max | 10.1% | 17.5% | 30.3% | 12.1% | 2.0% | 0.153 |
| section | titles, inventory prompt, pictorial only (no 26-29) | 10.5% | 22.7% | 34.3% | 20.4% | 6.1% | 0.176 |

Reading: with the whole description as the query, the inventory prompt gains 3.0
points at category hit@3, 5.0 at division hit@3 and 3.4 at division hit@10 over the
default prompt, and loses 2.6 at category hit@1; the section level is unchanged
(9.8% at rank 1). Sentence chunking hurts with this prompt as with the other.
Restricted to the pictorial categories, the inventory prompt reaches section hit@10
34.3% and recall@10 20.4% against 30.4% and 19.2% with the default prompt.

### The vision model: Florence-2 against the 7B reference

The browser demo needed a model a browser can run, which led to the comparison
below on the same 300 images and index, the whole description as the query. Three
small candidates were tried on the nine demo images first (`docs/evals.md`): SmolVLM-256M
answered with one word, looped or described "the Earth's oceans" on a crown above a
word, SmolVLM-500M looped on the same image, and Florence-2 base (230M, a captioner
driven by a task token) named every element literally without looping.

| level | model | hit@1 | hit@3 | hit@10 | recall@10 | main@1 | MRR |
|---|---|---|---|---|---|---|---|
| category | frequency baseline | 32.0% | 65.7% | 85.3% | 75.0% | - | - |
| category | Florence-2 base, detailed caption (the default) | 37.0% | 69.7% | 74.3% | 58.3% | 22.0% | 0.521 |
| category | Qwen2.5-VL 7B, default prompt (reference) | 44.3% | 68.7% | 72.7% | 54.2% | 26.7% | 0.563 |
| category | SmolVLM-256M, plain prompt (rejected) | 29.7% | 49.3% | 50.3% | 32.6% | 16.7% | 0.394 |
| division | frequency baseline | 26.0% | 44.3% | 61.3% | 50.0% | - | - |
| division | Florence-2 base, detailed caption (the default) | 31.0% | 53.3% | 62.7% | 42.5% | 18.0% | 0.424 |
| division | Qwen2.5-VL 7B, default prompt (reference) | 33.0% | 50.7% | 60.3% | 40.0% | 18.3% | 0.429 |
| division | SmolVLM-256M, plain prompt (rejected) | 12.7% | 24.3% | 30.0% | 16.9% | 6.0% | 0.188 |
| section | frequency baseline | 13.1% | 21.5% | 39.7% | 22.8% | - | - |
| section | Florence-2 base, detailed caption (the default) | 11.1% | 20.5% | 30.6% | 11.2% | 4.7% | 0.169 |
| section | Qwen2.5-VL 7B, default prompt (reference) | 9.8% | 20.2% | 33.0% | 13.0% | 3.7% | 0.167 |
| section | SmolVLM-256M, plain prompt (rejected) | 3.7% | 6.4% | 14.1% | 5.9% | 2.0% | 0.063 |

Reading: Florence-2's literal captions ("a red shield with white letters A and B, a
yellow crown at the top") score close to the 7B model's inventories: 7 points
behind at category hit@1 and 2 at division hit@1, level or ahead from hit@3 on and
at section hit@1, at a thirtieth of the size and 0.3 s per image. The vision stage
needs a model that names what is there without interpreting; size matters less than
that discipline. SmolVLM-256M, which interprets and loops, is far behind at every
level. Florence-2 is therefore the package's only default and the model the demo
page runs; the 7B model stays an option through the `ollama:` backend.

## Analysis

- **Pictorial elements are found; boilerplate codes are not.** The per-category
  table splits the problem in two. Where a mark shows something (animals, heraldry,
  human beings, celestial bodies, plants), the description places it in the right
  division one to three times out of four and the frequency prior never does. Where
  the gold is a convention of EUIPO coding (27.5.1 for any word in a special font,
  the 26.4 quadrilaterals for label-shaped backgrounds, the 29.1 colour codes), the
  prior scores 81% to 100% at the division and the description 28% to 79%. Those
  three categories hold 507 of the 907 gold codes, which is why the overall section
  numbers sit below the baseline while the pictorial ones sit above it.
- **The read misses explain the section level.** The reference model's `default`
  prompt's enumeration came back as lists of absences ("there are no human beings, animals,
  plants, ..."), and those sentences attract the entries they deny: heads in
  silhouette, empty shields, treble clefs rank in the top 5 of plain word marks.
  Colours are named in every description, and the twelve "Colours > Colours > ..."
  entries are short texts close to any such sentence, so one or two sit in every
  top 5; masking them lifts category hit@1 to 48.8% and main@1 to 35.1%, at the
  cost of the 85 colour codes in the gold. The `inventory` prompt removes the
  absence lists and brings the gold into the top 3 and top 10 more often, but not to
  rank 1: what a description can say about a word mark in a plain font does not
  resemble "Letters presenting a special form of writing" under either prompt, nor
  in Florence-2's captions. A rule-based colour stage and a prior would address the
  rest.
- **The scoring levers of text2ipc change little here.** Appending the notes to the
  texts moves every number by at most 0.6 points. Path support adds 2 points at
  category hit@1 and removes 2 to 7 at hit@10; subtree support changes nothing.
  Sentence-level chunking hurts at every level (section hit@1 9.8% whole, 8.1%
  max, 7.7% mean): a sentence such as "The background is white" is a worse query
  than the whole paragraph. Leaving the auxiliary sections out hurts too (category
  hit@1 36.0% against 44.3%), because EUIPO codes them as often as the principal
  ones. Cosine over the whole description with titles-only texts stays the default.
- **A second stage has room, a prior has more.** With 50 candidates the gold
  section is present 53.9% of the time against 9.8% at rank 1, the division 75.3%
  against 33.0%: a judge over the candidates, as text2ipc's cross-encoder, could
  recover part of that gap. The boilerplate codes need something else: a prior
  learned from the 770k L3D labels, fused with the similarity, is the cheapest
  candidate.
- **The set caps the section level.** 179 of 907 gold codes are EUIPO extension
  codes (`x.y.91` to `x.y.99`) that no WIPO edition contains, the remaining codes
  follow editions 5 to 8 against an edition 10 index, and the images are 256
  pixels wide. Marks coded by EUIPO with edition 10 through the API are the next
  eval set.

## Next steps

1. Evals on current EUIPO coding: `i2vienna euipo --n 300` fetches marks coded with
   edition 10 at full image size through the Trademark Search API once the free
   developer credentials exist (`EUIPO_CLIENT_ID`, `EUIPO_CLIENT_SECRET` from
   [dev.euipo.europa.eu](https://dev.euipo.europa.eu)); this removes the edition
   drift and the extension-code ceiling of the L3D numbers.
2. A frequency prior fused with the similarity: the boilerplate codes (27.5.1, the
   26.4 quadrilaterals, the 29.1 colours) are predictable without looking at the
   image, and the baseline shows how much they weigh.
3. A rule-based stage for colours and letter forms, which the caption states
   explicitly; the sweep's configuration study (notes, chunking, supports, masks)
   should be repeated on Florence-2's captions (`scripts/eval_sweep.py` does it
   from the cache).
4. A second stage over the top 50 (a cross-encoder, or the vision model judging
   entry texts against the image), as text2ipc's reranker; the hit@50 ceiling
   bounds the gain.
5. The browser demo (`i2vienna web-export`, ADR 0004) is published at
   https://huggingface.co/spaces/jaimenms/image2vienna and runs the same vision
   model as the package. Florence-2's larger variant (`Florence-2-large`, 770M) is
   worth measuring next: if captions are what the embedding stage wants, more of
   them may pay.
6. `notebooks/01_image2vienna.ipynb` (generated by `scripts/make_notebooks.py`,
   executed) walks through the interface on the drawn logos and re-runs the eval
   from the cached descriptions.

## Reproduce

```bash
uv sync --all-extras
uv run i2vienna build --edition 10 && uv run i2vienna build --edition 10 --notes
uv run i2vienna l3d --n 300
uv run i2vienna describe-cases evals/l3d_300.jsonl                      # Florence-2, the default
uv run i2vienna eval evals/l3d_300.jsonl
uv run python scripts/eval_sweep.py evals/l3d_300.jsonl                 # every scoring configuration
# the 7B reference (needs Ollama and `ollama pull qwen2.5vl:7b`):
uv run i2vienna describe-cases evals/l3d_300.jsonl --describer ollama:qwen2.5vl:7b --prompt default
uv run python scripts/eval_sweep.py evals/l3d_300.jsonl --describer ollama:qwen2.5vl:7b --prompt default
```
