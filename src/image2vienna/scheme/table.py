"""The classification as a table: one row per entry, with the code path to the root.

The text of an entry is the concatenation, top-down, of the titles along its path
(category > division > section), joined with ``>``. That whole text is what gets
embedded, never the entry's own title alone: a section reads "One star" and only the
path makes it "Celestial bodies, natural phenomena, geographical maps > Stars,
comets > One star".

``notes=True`` appends the entry's "Including ..." explanatory notes to the text
(the "Not including ..." ones name what belongs elsewhere and are never embedded);
whether that helps is an eval question (``docs/evals.md``).

Stored as ``scheme/vienna<edition>_<lang>.parquet`` with columns ``code, level,
parent, title, auxiliary, notes, associated, path``; ``path`` is the ``|``-joined
chain of codes from the category to the entry.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .model import CODE_SEPARATOR, PATH_SEPARATOR, ViennaNode, text_hash

COLUMNS = ("code", "level", "parent", "title", "auxiliary", "notes", "associated", "path")
_NEGATIVE_NOTE_RE = re.compile(r"^\s*(not including|excluding|does not include)\b", re.I)


@dataclass
class SchemeTable:
    nodes: list[ViennaNode]
    paths: list[tuple[str, ...]]  # codes from category to self, aligned with nodes

    def __post_init__(self):
        self.position = {n.code: i for i, n in enumerate(self.nodes)}

    @classmethod
    def from_nodes(cls, nodes: list[ViennaNode]) -> SchemeTable:
        paths: dict[str, tuple[str, ...]] = {}
        out = []
        for n in nodes:  # parents precede children
            path = (n.code,) if n.parent is None else (*paths[n.parent], n.code)
            paths[n.code] = path
            out.append(path)
        return cls(nodes, out)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[ViennaNode]:
        return iter(self.nodes)

    def node(self, code: str) -> ViennaNode:
        return self.nodes[self.position[code]]

    def path_of(self, code: str) -> tuple[str, ...]:
        return self.paths[self.position[code]]

    def text_of(self, code: str, *, notes: bool = False) -> str:
        return self.text_at(self.position[code], notes=notes)

    def text_at(self, i: int, *, notes: bool = False) -> str:
        parts = [self.nodes[self.position[c]].title for c in self.paths[i]]
        text = PATH_SEPARATOR.join(p for p in parts if p)
        if notes:
            positive = [n for n in self.nodes[i].notes if not _NEGATIVE_NOTE_RE.match(n)]
            if positive:
                text = f"{text}. {' '.join(positive)}"
        return text

    def texts(self, *, notes: bool = False) -> list[str]:
        return [self.text_at(i, notes=notes) for i in range(len(self.nodes))]

    def hashes(self, *, notes: bool = False) -> list[str]:
        return [text_hash(t) for t in self.texts(notes=notes)]

    # -- persistence ---------------------------------------------------------------

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "code": [n.code for n in self.nodes],
                "level": [n.level for n in self.nodes],
                "parent": [n.parent for n in self.nodes],
                "title": [n.title for n in self.nodes],
                "auxiliary": pa.array([n.auxiliary for n in self.nodes], type=pa.bool_()),
                "notes": pa.array([list(n.notes) for n in self.nodes], type=pa.list_(pa.string())),
                "associated": pa.array(
                    [list(n.associated) for n in self.nodes], type=pa.list_(pa.string())
                ),
                "path": [CODE_SEPARATOR.join(p) for p in self.paths],
            }
        )
        pq.write_table(table, path, compression="zstd")

    @classmethod
    def read(cls, path: Path) -> SchemeTable:
        t = pq.read_table(path, columns=list(COLUMNS)).to_pydict()
        nodes = [
            ViennaNode(
                code=t["code"][i],
                level=t["level"][i],
                parent=t["parent"][i] or None,
                title=t["title"][i],
                auxiliary=bool(t["auxiliary"][i]),
                notes=tuple(t["notes"][i] or ()),
                associated=tuple(t["associated"][i] or ()),
            )
            for i in range(len(t["code"]))
        ]
        paths = [tuple(p.split(CODE_SEPARATOR)) for p in t["path"]]
        return cls(nodes, paths)
