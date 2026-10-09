"""Vienna code formats.

WIPO writes codes without zero padding (``1.1.2``, recommendation (a) of the Committee
of Experts); offices such as EUIPO pad every part to two digits (``01.01.02``). The
canonical form here is WIPO's. A code has one part (category), two (division) or three
(section); the "A" of auxiliary sections never appears in a coded mark.
"""

from __future__ import annotations

import re

from ..config import LEVELS

_CODE_RE = re.compile(r"^\s*A?\s*(\d{1,2})(?:\.(\d{1,2}))?(?:\.(\d{1,2}))?\s*$")


def normalize_code(code: str) -> str:
    """``01.01.02``, ``1.1.2``, ``A 1.1.2`` or ``1.01`` -> ``1.1.2``."""
    m = _CODE_RE.match(code)
    if m is None:
        raise ValueError(f"Not a Vienna code: {code!r}")
    parts = [str(int(p)) for p in m.groups() if p is not None]
    if any(p == "0" for p in parts):
        raise ValueError(f"Not a Vienna code: {code!r}")
    return ".".join(parts)


def level_of_code(code: str) -> str:
    n = code.count(".") + 1
    if n > len(LEVELS):
        raise ValueError(f"Not a Vienna code: {code!r}")
    return LEVELS[n - 1]


def truncate_code(code: str, level: str) -> str:
    """``1.1.2`` at level ``division`` -> ``1.1``; a shallower code is returned as is."""
    parts = code.split(".")
    return ".".join(parts[: LEVELS.index(level) + 1])


def format_code(code: str, *, padded: bool = False) -> str:
    """Canonical -> ``01.01.02`` when ``padded`` (EUIPO style), else unchanged."""
    if not padded:
        return code
    return ".".join(p.zfill(2) for p in code.split("."))


def parse_cfe(text: str) -> list[str]:
    """Codes from a CFE string following WIPO's recommendation, e.g.
    ``CFE 1.1.2,10,25; 1.15.17; 2.9.1`` or ``1.1.2-4``, expanded and canonical."""
    out: list[str] = []
    body = re.sub(r"^\s*CFE\s*(\(\d+\))?\s*", "", text.strip(), flags=re.I)
    for group in filter(None, (g.strip() for g in body.split(";"))):
        items = [i.strip() for i in group.split(",") if i.strip()]
        if not items:
            continue
        head = normalize_code(items[0].split("-")[0])
        prefix = head.rsplit(".", 1)[0] if "." in head else None
        for item in items:
            lo, _, hi = item.partition("-")
            lo = lo.strip()
            first = normalize_code(lo if "." in lo or prefix is None else f"{prefix}.{lo}")
            if not hi:
                out.append(first)
                continue
            base, last = first.rsplit(".", 1)
            for n in range(int(last), int(hi) + 1):
                out.append(f"{base}.{n}")
    return list(dict.fromkeys(out))
