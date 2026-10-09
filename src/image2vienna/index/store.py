"""One Parquet index per (edition, language, embedding model, text style).

The index carries only what depends on the model: ``code``, ``path`` (the ``|``-joined
code chain the text was built from), ``text_hash`` and the vector as float32 columns
``e000 .. eNNN``. Titles live in the scheme table (``SchemeTable``), so the same text
is rendered for display and for hashing. Metadata (edition, language, model,
dimension, whether notes were embedded, build time) is stored in the Parquet schema
metadata under ``image2vienna``.

Rows keep the scheme's depth-first order, so every parent precedes its children; the
search code relies on that.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import LEVELS
from ..scheme.model import CODE_SEPARATOR, ViennaNode, text_hash
from ..scheme.table import SchemeTable

META_KEY = b"image2vienna"


@dataclass(frozen=True)
class IndexMeta:
    edition: str
    lang: str
    model: str
    dim: int
    rows: int
    built_at: str
    notes: bool = False

    @classmethod
    def now(cls, **kwargs) -> IndexMeta:
        return cls(built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **kwargs)


def vector_columns(dim: int) -> list[str]:
    width = max(3, len(str(dim - 1)))
    return [f"e{i:0{width}d}" for i in range(dim)]


class ViennaIndex:
    """Scheme table + aligned vectors, plus the derived hierarchy arrays."""

    def __init__(
        self,
        meta: IndexMeta,
        scheme: SchemeTable,
        vectors: np.ndarray,
        hashes: list[str] | None = None,
    ):
        if vectors.shape != (len(scheme), meta.dim):
            raise ValueError(f"vectors {vectors.shape} do not match {len(scheme)} x {meta.dim}")
        self.meta = meta
        self.scheme = scheme
        self.nodes: list[ViennaNode] = scheme.nodes
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.texts = scheme.texts(notes=meta.notes)
        self.hashes_list = hashes or [text_hash(t) for t in self.texts]
        self.codes = [n.code for n in self.nodes]
        self.position = scheme.position
        self.levels = np.array([LEVELS.index(n.level) for n in self.nodes], dtype=np.int8)
        self.parent_idx = np.array(
            [self.position[n.parent] if n.parent else -1 for n in self.nodes], dtype=np.int64
        )
        if np.any(self.parent_idx >= np.arange(len(self.nodes))):
            raise ValueError("Index rows must be in depth-first order (parents before children)")
        self.children: list[list[int]] = [[] for _ in self.nodes]
        for i, p in enumerate(self.parent_idx):
            if p >= 0:
                self.children[p].append(i)

    def __len__(self) -> int:
        return len(self.nodes)

    def hashes(self) -> dict[str, str]:
        return dict(zip(self.codes, self.hashes_list, strict=True))

    def node(self, code: str) -> ViennaNode:
        return self.scheme.node(code)

    def text_of(self, code: str) -> str:
        return self.texts[self.position[code]]

    # -- persistence ---------------------------------------------------------------

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, object] = {
            "code": pa.array(self.codes),
            "path": pa.array([CODE_SEPARATOR.join(p) for p in self.scheme.paths]),
            "text_hash": pa.array(self.hashes_list),
        }
        for j, name in enumerate(vector_columns(self.meta.dim)):
            columns[name] = pa.array(self.vectors[:, j], type=pa.float32())
        table = pa.table(columns).replace_schema_metadata(
            {META_KEY: json.dumps(asdict(self.meta)).encode()}
        )
        pq.write_table(table, path, compression="zstd")

    @staticmethod
    def read_meta(path: Path) -> IndexMeta:
        raw = pq.read_schema(path).metadata or {}
        if META_KEY not in raw:
            raise ValueError(f"{path} carries no image2vienna metadata")
        return IndexMeta(**json.loads(raw[META_KEY]))

    @classmethod
    def read(cls, path: Path, scheme: SchemeTable | Path) -> ViennaIndex:
        meta = cls.read_meta(path)
        if isinstance(scheme, Path):
            scheme = SchemeTable.read(scheme)
        table = pq.read_table(path)
        codes = table.column("code").to_pylist()
        if codes != [n.code for n in scheme.nodes]:
            raise ValueError(f"{path.name} rows do not match the scheme table")
        cols = vector_columns(meta.dim)
        vectors = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in cols]
        ).astype(np.float32, copy=False)
        return cls(meta, scheme, vectors, table.column("text_hash").to_pylist())
