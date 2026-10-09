# Methodology

## Problem

Given a trade mark image, return Vienna Classification codes ranked by how well they
describe its figurative elements, at a requested level (category, division, section)
or at the deepest level the evidence supports. Offices assign several codes to one
mark, one per distinct element (general note (e) of the classification), so the
answer is a list.

## 1. The classification as a set of texts

WIPO's reference XML nests 29 categories, 145 divisions, 845 principal sections and
936 auxiliary sections (edition 10). A section title alone is not enough to embed:
"One star" or "Heads, busts" only mean something under their division. The embedded
text of an entry is therefore the chain of titles from the category down, joined
with `>`:

```
Celestial bodies, natural phenomena, geographical maps > Stars, comets > One star
Human beings > Men > Heads, busts
```

Titles printed in capitals are sentence-cased, cross references are stripped, and
auxiliary sections are embedded like principal ones (ADR 0002). The explanatory
notes stay out of the text by default; an index built with `--notes` appends the
"Including ..." notes, and `docs/evals.md` compares the two.

## 2. Two stages at query time

**Describe.** A vision-language model (default `qwen2.5vl:7b` through Ollama)
receives the image and a prompt asking for an inventory of the visible figurative
elements in the vocabulary of the classification: beings and their attributes,
animals, plants, celestial bodies, objects, heraldry, geometric figures, letters and
how they are written, colours; neutral terms, no brand guessing, no interpretation
(`describe/prompts.py`). The answer is three to six sentences of English prose.
Three prompts are measured: `default` (the enumeration above), `terse` (one line
per element), and `inventory` (one sentence per element that is present, how letters
are written, colours last, and an explicit ban on naming absent kinds of elements,
after the default was seen to produce lists of absences).

**Embed and score.** The description is embedded as a query (`multilingual-e5-base`,
the text2ipc default, so the two studies compare) and scored against the entry
vectors by cosine. The hierarchy heuristics of text2ipc apply unchanged: path
support, subtree support, beam descent over the three levels, auto level, gap, and
branch de-duplication (`search/scorer.py`). `principal_only` leaves the auxiliary
sections out when an office does not use them.

**Sentences as elements.** A description names one element per sentence more often
than not, and an office codes one section per element (general note (e)). With
`chunking="max"` every sentence is embedded separately and an entry scores by its
best sentence, so the lion and the crossed swords each find their own section
instead of competing inside one averaged vector; `"mean"` searches with the mean of
the sentence vectors; `"whole"` embeds the text as one query. Which one wins is an
eval question (`docs/evals.md`).

**Masking.** `exclude_codes` leaves whole subtrees out of the candidates. Every
description mentions colours, so the Colours category (29) is a hub that any
description resembles; masking it shows how the figurative elements themselves are
coded, and the eval drops the same codes from the gold (`exclude_gold`).

Either stage runs alone: `classify_text` scores any text, so a cached or hand-edited
description costs milliseconds, and `describe` returns the prose for inspection.

## 3. Evaluation

Cases are EUIPO figurative marks with the examiners' codes (ADR 0003): the L3D
sample now, the Trademark Search API when credentials exist. The vision model runs
once per case and its description is cached in the case file; every scoring
experiment (text style, weights, prompt, level) reuses it.

Metrics, per level, over the top-k predictions truncated to that level:

| metric | meaning |
|---|---|
| hit@k | any gold code among the top-k |
| recall@k | share of the gold codes among the top-k |
| main@1 | the office's first-listed code is first |
| MRR | mean reciprocal rank of the first gold code |

A gold code shallower than the level (a division-only code at section level) does
not count at that level; `n` is reported per level. A frequency baseline (always
answer the most frequent codes of the whole label set) is reported next to the
model, since Vienna coding is dominated by a few typographic and geometric codes.
