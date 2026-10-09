# 0003 Evals from EUIPO coding: the L3D sample now, the Trademark Search API next

Status: accepted, 2026-10-09

## Context

Gold labels must be codes assigned by an office to real marks. EUIPO codes every
figurative EU trade mark with Vienna sections and publishes its register. Three
routes were examined on 2026-10-09:

1. EUIPO's historical bulk FTP (`ftp.euipo.europa.eu`, user `opendata`), used by
   the authors of the Large Labelled Logo Dataset in 2021: the login no longer
   authenticates.
2. EUIPO's Trademark Search API (`api.euipo.europa.eu/trademark-search`, free
   developer account, OAuth2 client credentials): search by RSQL
   (`markFeature==FIGURATIVE`), detail with `markImage.viennaClasses`, and an
   `/image` endpoint. The OpenAPI document is embedded in the developer portal page.
   It needs credentials only the account owner can create.
3. The Large Labelled Logo Dataset (L3D, Gutiérrez-Fandiño et al. 2021, CC BY 4.0):
   about 770k EUIPO figurative marks from 1996 to 2020, 256x256, with the
   examiners' Vienna codes, one 12 GB tar on Zenodo. The images sit at the front of
   the archive in UUID (random) order and the label file `results.json` at its end;
   Zenodo honours range requests.

## Decision

- `i2vienna l3d --n N` builds eval cases from L3D without downloading the archive:
  the last 120 MB give the labels, the first tens of megabytes give images in random
  order, and `N` of the labelled ones are sampled with a seed. `evals/l3d_300.jsonl`
  is the first standard file. Descriptions from the vision model are cached in the
  same file (`i2vienna describe-cases`), so every scoring experiment reuses one
  vision run.
- `i2vienna euipo --n N` builds cases from the API when `EUIPO_CLIENT_ID` and
  `EUIPO_CLIENT_SECRET` are set; these marks carry current codes (edition 10 from
  2026) and current images.
- Metrics follow text2ipc (hit@k per level, main@1, MRR) plus recall@k, since a
  mark carries one code per figurative element.

## Consequences

- L3D codes follow editions 5 to 8; the index is edition 10. Categories and
  divisions are stable across editions, sections drift (over a hundred changes
  between 8 and 9 alone), and EUIPO-only codes such as `29.1.98` never match. The
  section-level numbers on L3D are therefore a floor.
- L3D images are 256x256 with white padding, smaller than what the API serves; the
  vision model's recall on small images is part of what the L3D numbers measure.
- Nothing from EUIPO or L3D is committed except the case files (codes, file names,
  verbal elements, cached descriptions); images stay under `data/images/`.
