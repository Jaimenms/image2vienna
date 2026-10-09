"""Where scheme tables and indexes live, and how to find the previous edition."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import home
from ..embeddings.base import model_slug

_NAME_RE = re.compile(
    r"^vienna(?P<edition>\d+)_(?P<lang>[a-z]{2})_(?P<model>.+?)(?P<notes>_notes)?\.parquet$"
)


@dataclass(frozen=True)
class IndexRef:
    edition: str
    lang: str
    model_slug: str
    notes: bool
    path: Path


def index_dir(root: Path | None = None) -> Path:
    return (root or home()) / "index"


def scheme_dir(root: Path | None = None) -> Path:
    return (root or home()) / "scheme"


def scheme_table_path(edition: str, lang: str, root: Path | None = None) -> Path:
    return scheme_dir(root) / f"vienna{edition}_{lang.lower()}.parquet"


def index_path(
    edition: str, lang: str, model: str, *, notes: bool = False, root: Path | None = None
) -> Path:
    suffix = "_notes" if notes else ""
    return index_dir(root) / f"vienna{edition}_{lang.lower()}_{model_slug(model)}{suffix}.parquet"


def available_indexes(root: Path | None = None) -> list[IndexRef]:
    d = index_dir(root)
    if not d.exists():
        return []
    return [
        IndexRef(m["edition"], m["lang"].upper(), m["model"], bool(m["notes"]), p)
        for p in sorted(d.glob("vienna*.parquet"))
        if (m := _NAME_RE.match(p.name))
    ]


def find_previous_index(
    edition: str, lang: str, model: str, *, notes: bool = False, root: Path | None = None
) -> Path | None:
    """Newest built index for the same language, model and text style of an older edition."""
    slug = model_slug(model)
    older = sorted(
        (
            r
            for r in available_indexes(root)
            if r.lang == lang.upper()
            and r.model_slug == slug
            and r.notes == notes
            and int(r.edition) < int(edition)
        ),
        key=lambda r: int(r.edition),
    )
    return older[-1].path if older else None
