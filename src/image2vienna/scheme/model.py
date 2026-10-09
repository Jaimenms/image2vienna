"""The unit of the Vienna Classification as this package sees it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

PATH_SEPARATOR = " > "
CODE_SEPARATOR = "|"


@dataclass(frozen=True)
class ViennaNode:
    """One classifiable entry: a category, a division or a section.

    Auxiliary sections (printed with an "A" before the code) group the figurative
    elements of the principal sections of their division by a specific criterion;
    offices code them without the letter, so they are ordinary section-level targets.
    """

    code: str  # canonical dotted form without zero padding, e.g. 1, 1.1, 1.1.2
    level: str  # one of config.LEVELS
    parent: str | None  # code of the parent entry, None for categories
    title: str  # own title, explanatory notes and example images stripped
    auxiliary: bool = False  # an "A" section
    #: Explanatory notes attached to the entry ("Including ...", "Not including ...").
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Auxiliary sections only: the principal sections they are associated with.
    associated: tuple[str, ...] = field(default_factory=tuple)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
