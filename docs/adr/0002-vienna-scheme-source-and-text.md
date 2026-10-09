# 0002 WIPO's nivilo XML as the scheme source; embed the full path, titles only

Status: accepted, 2026-10-09

## Context

WIPO publishes the Vienna Classification online (`nivilo.wipo.int/vienna<edition>`)
and lists "files and data" per edition in its download area, but the latter pages
are rendered by scripts and expose no stable file URLs. The online publication is
generated from one reference file per language, `xml/<lang>/full.xml`, which nests
categories, divisions, principal sections, auxiliary sections, explanatory notes,
cross references and example images.

Auxiliary ("A") sections group the elements of the principal sections of their
division by a criterion (one star, three stars, stars with rays). The "Guidance for
the User" says the letter never appears in a coded mark, and offices assign auxiliary
codes freely (EUIPO codes "three stars" as 01.01.04), so they are classification
targets like any other section.

## Decision

- `fetch_scheme(edition, lang)` downloads `full.xml` from nivilo; editions are keyed
  by their number (7 to 10) with the year of entry into force in `config.EDITIONS`.
- Every category, division, principal section and auxiliary section becomes a node,
  in depth-first order; auxiliary sections are children of their division and record
  the principal sections they are associated with.
- Codes are canonical in WIPO's unpadded form (`1.1.2`); EUIPO's `01.01.02` and the
  CFE notation of the Committee's recommendations are parsed and formatted.
- The embedded text of an entry is the chain of titles from the category down,
  joined with `>`, as in text2ipc (ADR 0001 there). Titles printed in capitals are
  sentence-cased; cross references ("(except 2.1.2)", "not classified in division
  1.11") are stripped because they name what belongs elsewhere.
- Explanatory notes are kept in the scheme table and left out of the text by
  default. `--notes` builds a second index whose texts append the "Including ..."
  notes (never the "Not including ..." ones); the two are compared in the evals.
- Two Parquet tables, scheme and index, with the layout of text2ipc's ADR 0006
  (`code, level, parent, title, auxiliary, notes, associated, path` and
  `code, path, text_hash, e000..eNNN`).

## Consequences

- No dependence on WIPO's download area; the XML is 190 KB and cached under
  `data/wipo/`. Should WIPO move the file, only `scheme_url` changes.
- Texts are short (mean 103 characters, max 326), far below the embedder's limit;
  no chunking is needed on the index side.
- Edition changes invalidate only the subtree below a renamed title; incremental
  builds reuse the rest, as in text2ipc.
- English only for now. The French file exists at the same path and the code takes
  `--lang FR`; queries are English prose from the vision model, so the scheme
  language is not a lever here as it was for Portuguese patent texts.
