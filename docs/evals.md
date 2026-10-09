# Eval log

A log in chronological order: every run is kept, including the ones that were
rejected, so a section describes the setup of its day.

## Setup, 2026-10-09

- **Scheme**: Vienna Classification edition 10, English, from
  `nivilo.wipo.int/vienna10/xml/en/full.xml`: 29 categories, 145 divisions, 845
  principal and 936 auxiliary sections, 1,955 entries. Path texts: mean 103
  characters, max 326.
- **Embedder**: `st:intfloat/multilingual-e5-base` (768d). Embedding the 1,955
  texts takes 3.7 s on an Apple M5 Pro.
- **Vision model**: `qwen2.5vl:7b` through Ollama (Q4, 6 GB), temperature 0, at most
  400 tokens. About 3.5 s per image on the same machine (1,115 prompt tokens with
  the default prompt, 90 to 135 output tokens).
- **Cases**: `evals/l3d_300.jsonl`, 300 figurative EU trade marks drawn with seed 0
  from the front of the L3D archive (EUIPO open data, 1996 to 2020, 256x256 images),
  with the examiners' Vienna codes: 907 gold codes, 1 to 18 per mark (median 2),
  901 at section level and 6 at division level. Most frequent categories in the
  gold: 26 geometric figures (280 codes), 27 forms of writing (142), 29 colours (85),
  1 celestial bodies (63), 24 heraldry (61), 25 ornamental motifs (59), 5 plants (56),
  3 animals (49).
- **Gold not in edition 10**: 179 of the 907 codes (19.7%) do not exist in WIPO's
  edition 10. They are EUIPO's own extension codes (`1.1.99`, `2.1.91`, `24.17.97`,
  `25.1.95`, `29.1.98` ...), used by the office for finer or office-specific
  subdivisions. They can never be hit at section level, only at division level
  after truncation; section-level hit rates on this set are therefore a floor.
- **Metric**: hit@k if any gold code truncated to the level appears in the top-k
  predictions truncated to that level; recall@k the share of gold codes found;
  main@1 the office's first-listed code at rank 1; MRR of the first gold hit. A
  gold code shallower than the level does not count at that level (`n` per level).

## Smoke test on four synthetic logos (before the index existed)

Pillow drawings: three stars over a crescent moon, a sun rising over the sea, a
bottle inside a circle, letters "AB" on a shield under a crown. Qwen2.5-VL named
every element correctly in 3.3 to 4.1 s per image. Flat cosine over the 1,955 path
texts put the right division first in 4 of 4 (1.7 moon, 1.3 sun, 26.1 circles,
24.1 shields) and the right section among the top 5 in 4 of 4 (1.1.4 three stars,
1.3.15 sun with straight rays, 19.7.1 bottle, 24.1.25 / 24.9.9 shield and crown).
"Colours > Five colours and over" (29.1.15) ranked first for the stars-and-moon
drawing because the description lists colours; EUIPO does code colours under 29.1,
so the prompt keeps asking for them.

## Frequency baseline (always answer the most frequent codes of the 770k L3D labels)

The ten most frequent codes: 27.5.1, 26.4.5, 26.4.2, 29.1.4, 26.4.22, 26.1.3,
27.5.21, 29.1.1, 26.4.1, 29.1.8 (12.6% of all L3D marks carry 27.5.1, "letters
presenting a special form of writing").

| level | hit@1 | hit@3 | hit@10 |
|---|---|---|---|
| category | 32.0% | 65.7% | 65.7% |
| division | 26.0% | 44.3% | 49.0% |
| section | 13.1% | 21.5% | 39.7% |

The baseline plateaus because ten codes cover only three categories; any model
must beat these numbers to be worth running.

## Qualitative findings on the first 100 described cases (default prompt)

Seven random cases read side by side with their descriptions and top-5 sections:

- **The prompt's enumeration came back as a list of absences.** Asked to look for
  "human beings, animals, plants, celestial bodies, landscapes, ...", the model wrote
  for every word mark "There are no human beings, animals, plants, celestial bodies,
  landscapes, buildings, vehicles, tools, containers, ...", and those sentences
  attract exactly the entries they deny: "Heads in silhouette", "Shields containing
  neither a figurative element nor an inscription", "Treble clefs alone" rank in the
  top 5 for a plain word mark. The `inventory` prompt forbids naming absent kinds of
  elements and is measured below.
- **Colours are a hub.** Every description names colours, and the twelve "Colours >
  Colours > ..." entries are short texts close to any such sentence, so one or two
  of them sit in every top 5. EUIPO does code colours (29.1.4 "Blue" and 29.1.6
  "White, grey, silver" are in the gold of a blue-and-white logo), so they are not
  noise, but they crowd out the pictorial sections at rank 1. `--exclude 29` and the
  "pictorial only" rows show the figurative quality without them.
- **Boilerplate codes need a prior, not a description.** 27.5.1 "Letters presenting
  a special form of writing" is on 12.6% of all EUIPO figurative marks and on most
  word-in-a-font logos of the sample; a description like "the word Källscare in a
  bold sans-serif font" does not resemble that title, and the model lands on 27.5.24
  or on heraldic "Dots". The same holds for the quadrilateral codes of 26.4 that
  EUIPO assigns to every label-shaped background. A frequency prior gets these for
  free, which is why the baseline is strong; the pictorial categories are where the
  description earns its cost (per-category table below).
- **Pictorial elements are found, at the division.** A crown above a word gives
  24.9.5 "One crown" and 24.9.3 "Stylized or fanciful crowns" at ranks 1 and 2
  while EUIPO coded 24.9.2 "Crowns open at the top" and 24.9.9 "Crowns having three
  triangular points": right division, a more specific auxiliary section in the gold.
- **Edition and office drift.** Gold such as 25.5.99, 27.99.7, 27.99.9 are EUIPO
  extension codes with no WIPO entry; 28.3 (inscriptions in Chinese characters) is
  a division-only code that the section-level run cannot hit although the
  description says "Chinese characters".


## Sweep on the 300 L3D cases, default prompt (2026-10-09, 16:20)

One vision run (`qwen2.5vl:7b`, `default` prompt, 300 images in 27 min), every
scoring configuration from the cache (`scripts/eval_sweep.py`). Reading: the
pipeline beats the frequency baseline at rank 1 at category and division, loses at
hit@10 and at the section level; notes change nothing, sentence chunking and
principal-only hurt, path support trades hit@10 for hit@1, masking colours lifts
category main@1 to 35%. `PERFORMANCE.md` carries the analysis.

## Candidate recall ceiling (titles, top 50)

| query | category hit@50 | division hit@50 | section hit@50 | section recall@50 |
|---|---|---|---|---|
| whole | 81.7% | 75.3% | 53.9% | 27.5% |
| max | 84.0% | 73.7% | 45.1% | 23.3% |


## Gold codes found in the top 10, by category (whole description)

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


## Sweep on the 300 L3D cases, both prompts (2026-10-09, 16:55)

Second vision run with the `inventory` prompt (300 images, 27 min); every
configuration scored again from the cache. The inventory prompt lifts hit@3 and
hit@10 at category and division (+3.0 and +1.6, +5.0 and +3.4 points over the
default prompt, whole description) and loses 2.6 points at category hit@1; the
section level is unchanged. Sentence chunking hurts under both prompts. The best
section-level numbers of the study are the inventory prompt on the pictorial
categories alone (hit@10 34.3%, recall@10 20.4%).

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
| titles, inventory prompt, whole | 41.7% | 71.7% | 74.3% | 56.0% | 24.7% | 0.559 |
| titles, inventory prompt, sentence mean | 39.3% | 59.0% | 69.0% | 51.5% | 23.7% | 0.502 |
| titles, inventory prompt, sentence max | 35.0% | 68.0% | 72.3% | 52.8% | 18.3% | 0.506 |
| titles, inventory prompt, pictorial only (no 26-29) | 44.2% | 63.0% | 67.4% | 54.6% | 32.0% | 0.531 |

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
| titles, inventory prompt, whole | 32.3% | 55.7% | 63.7% | 42.7% | 18.7% | 0.444 |
| titles, inventory prompt, sentence mean | 31.3% | 44.3% | 54.7% | 37.6% | 19.0% | 0.389 |
| titles, inventory prompt, sentence max | 30.7% | 52.3% | 59.7% | 40.6% | 16.3% | 0.416 |
| titles, inventory prompt, pictorial only (no 26-29) | 29.3% | 45.9% | 56.9% | 43.5% | 21.0% | 0.383 |

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
| titles, inventory prompt, whole | 9.8% | 19.9% | 32.3% | 13.0% | 3.0% | 0.162 |
| titles, inventory prompt, sentence mean | 6.4% | 14.1% | 24.9% | 10.5% | 1.0% | 0.116 |
| titles, inventory prompt, sentence max | 10.1% | 17.5% | 30.3% | 12.1% | 2.0% | 0.153 |
| titles, inventory prompt, pictorial only (no 26-29) | 10.5% | 22.7% | 34.3% | 20.4% | 6.1% | 0.176 |


## The light model: SmolVLM-256M (2026-10-09, evening)

The browser demo runs `HuggingFaceTB/SmolVLM-256M-Instruct` (the "light" model;
the package can run it too, `--describer light`, through transformers, 1 to 1.6 s
per image on the M5 Pro). Prompts tried on five demo images:

| prompt | behaviour |
|---|---|
| `inventory` (the heavy model's) | one word ("Yellow.", "Black and white.") or a repetition loop; sometimes a usable paragraph |
| `terse` | short plausible sentences: "The child is holding a teddy bear", "a tree ... leaves are green"; sometimes interpretive ("The sun is a symbol of warmth") |
| a light-specific inventory prompt | echoes the prompt or names a word |
| any prompt + repetition penalty 1.3 | invents elements (dogs, cats, letters) |

A second round on the nine demo images, after `terse` looped on one of them and
echoed "trade" from "trade mark image": replacing the phrase by "logo", "image" or
"picture" removes the loop but shrinks the answers to a word or two ("C", "The
sun."); `no_repeat_ngram_size` removes the loop and keeps "trade"; a plain "Describe
this image in detail. Name every object, ... Do not name brands and do not
interpret." gives the longest and most concrete answers (81 words on average
against 30, no loop, no "trade": the red shield with a gold crown and the letters,
the dark green bottle in a circle, the boy with a stuffed animal). That prompt is
`light` and is the light model's from here on (`config.DESCRIBER_PROMPTS`);
repeated sentences are dropped from every model's output
(`textnorm.clean_description`). The demo examples carry the light model's `light`
description and the heavy model's `inventory` one.

Light model on the 300 L3D cases (whole description, titles index):

| level | light, terse: hit@1 / @3 / @10 | light, inventory: hit@1 / @3 / @10 | heavy, default: hit@1 / @3 / @10 |
|---|---|---|---|
| category | 26.0 / 44.3 / 48.0 | 20.3 / 42.3 / 47.3 | 44.3 / 68.7 / 72.7 |
| division | 14.0 / 26.3 / 31.7 | 14.3 / 25.7 / 31.0 | 33.0 / 50.7 / 60.3 |
| section | 6.1 / 11.1 / 17.5 | 5.7 / 10.8 / 15.5 | 9.8 / 20.2 / 33.0 |

The light model sits below the frequency baseline at every level; `terse` beats
`inventory` by 5.7 points at category hit@1 and ties elsewhere.
