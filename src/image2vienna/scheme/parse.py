"""Turn WIPO's ``full.xml`` of the Vienna Classification into an ordered list of nodes.

The file behind nivilo.wipo.int nests the classification physically::

    <category id="1" text="CELESTIAL BODIES, ...">
      <note>...</note>                      (or <notes><note>...</note></notes>)
      <division id="1" text="STARS, COMETS">
        <section id="1" aux="">Stars<note>...</note><img src="..."/></section>
        <auxiliaries>
          <text>Auxiliary Sections of Division <ref>1.1</ref></text>
          <head>(associated with Principal Sections <ref>1.1.1</ref>, <ref>1.1.15</ref>)</head>
          <auxiliary id="2">One star</auxiliary>

Categories, divisions, principal sections and auxiliary sections are classification
targets. Explanatory notes are kept on the node (they are not part of the embedded
text by default, see ``SchemeTable``); example images are dropped. The ``aux=""``
attribute marks a principal section that has auxiliary sections associated with it
(the asterisk of the printed edition) and carries no information of its own.

Cross references inside a title ("(except 2.1.2, 2.1.12)", "not classified in
division 1.11") point *away* from the entry and are stripped, as in text2ipc.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from .model import ViennaNode

_REF_CODE = r"\d{1,2}(?:\.\d{1,2}){0,2}"
# "(except 2.1.2, 2.1.12 and 2.1.14)", "(see 1.3.1)"
_PAREN_REF_RE = re.compile(rf"\s*\((?:[^()]*?\b{_REF_CODE}\b)[^()]*\)")
# "Other representations of stars, not classified in division 1.11"
_NOT_CLASSIFIED_RE = re.compile(
    rf",?\s*not (?:classified|included) (?:in|under)\b[^,;]*?\b{_REF_CODE}\b"
    rf"(?:\s*(?:,|and|or|to)\s*(?:\w+\s+)?{_REF_CODE})*",
    flags=re.I,
)


def parse_scheme(path: Path | str) -> list[ViennaNode]:
    """Entries in depth-first order, so every parent precedes its children."""
    root = ET.parse(path).getroot()
    return list(_walk(root))


def _walk(root: ET.Element) -> Iterator[ViennaNode]:
    for cat in root.findall("category"):
        cid = cat.get("id", "")
        yield ViennaNode(
            code=cid,
            level="category",
            parent=None,
            title=clean_title(cat.get("text", "")),
            notes=_notes_of(cat),
        )
        for div in cat.findall("division"):
            did = f"{cid}.{div.get('id', '')}"
            yield ViennaNode(
                code=did,
                level="division",
                parent=cid,
                title=clean_title(div.get("text", "")),
                notes=_notes_of(div),
            )
            for sec in div.findall("section"):
                yield ViennaNode(
                    code=f"{did}.{sec.get('id', '')}",
                    level="section",
                    parent=did,
                    title=clean_title(_own_text(sec)),
                    notes=_notes_of(sec),
                )
            for block in div.findall("auxiliaries"):
                associated = tuple(
                    r.text.strip()
                    for head in block.findall("head")
                    for r in head.findall("ref")
                    if r.text and r.text.strip()
                )
                for aux in block.findall("auxiliary"):
                    yield ViennaNode(
                        code=f"{did}.{aux.get('id', '')}",
                        level="section",
                        parent=did,
                        title=clean_title(_own_text(aux)),
                        auxiliary=True,
                        notes=_notes_of(aux),
                        associated=associated,
                    )


def _own_text(el: ET.Element) -> str:
    """The entry's title: its text and inline references, without nested notes or images."""
    parts = [el.text or ""]
    for child in el:
        if child.tag == "ref":
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def _notes_of(el: ET.Element) -> tuple[str, ...]:
    out = []
    for note in [*el.findall("note"), *el.findall("notes/note")]:
        text = " ".join("".join(note.itertext()).split())
        if text:
            out.append(text)
    return tuple(out)


def clean_title(title: str) -> str:
    """Strip cross references and sentence-case titles printed in capitals.

    Category and division titles are printed in capitals ("STARS, COMETS"); tokenizers
    of multilingual models handle shouting badly (text2ipc: +10 points of recall from
    lower-casing), so they are rendered as "Stars, comets". Mixed-case titles are left
    alone.
    """
    t = _PAREN_REF_RE.sub("", title)
    t = _NOT_CLASSIFIED_RE.sub("", t)
    t = " ".join(t.split()).strip(" ,;")
    if t and t == t.upper() and any(c.isalpha() for c in t):
        t = t[0] + t[1:].lower()
    return t
