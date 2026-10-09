"""Fetch and cache the reference XML of a Vienna Classification edition.

WIPO publishes the classification online at ``nivilo.wipo.int/vienna<edition>``; the
page is generated from ``xml/<lang>/full.xml``, which is what gets downloaded.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..config import EDITIONS, NIVILO_BASE_URL, WIPO_LANGS, home


def scheme_url(edition: str, lang: str = "EN") -> str:
    return f"{NIVILO_BASE_URL}/vienna{edition}/xml/{lang.lower()}/full.xml"


def scheme_path(edition: str, lang: str = "EN", root: Path | None = None) -> Path:
    """Raw WIPO file, kept under ``wipo/``; derived tables go to ``scheme/``."""
    return (root or home()) / "wipo" / f"vienna{edition}_{lang.upper()}_full.xml"


def fetch_scheme(
    edition: str,
    lang: str = "EN",
    *,
    root: Path | None = None,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Return the local path of the edition's ``full.xml``, downloading if needed."""
    lang = lang.upper()
    if lang not in WIPO_LANGS:
        raise ValueError(f"WIPO publishes the classification in {WIPO_LANGS}, not {lang}")
    if edition not in EDITIONS:
        raise ValueError(f"Unknown edition {edition!r}; known: {', '.join(EDITIONS)}")
    target = scheme_path(edition, lang, root)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(scheme_url(edition, lang), timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    if b"<category" not in resp.content:
        raise ValueError(f"{scheme_url(edition, lang)} does not look like the classification")
    target.write_bytes(resp.content)
    return target
