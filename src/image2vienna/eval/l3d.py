"""Eval cases from the Large Labelled Logo Dataset (L3D).

L3D (Gutiérrez-Fandiño, Pérez-Fernández, Armengol-Estapé, 2021, CC BY 4.0) holds about
770k figurative EU trade marks taken from EUIPO's open data up to 2020, resized to
256x256, each with the Vienna codes assigned by EUIPO examiners. It is published as
one 12 GB tar on Zenodo. The images sit at the front of the archive in random (UUID)
order and the label file at its end, and Zenodo honours HTTP range requests, so a
sample of a few hundred labelled images costs a few tens of megabytes: the labels
are read from the tail of the tar and the images from its head.

Codes follow the edition in force when each mark was examined (editions 5 to 8), so
section-level agreement with a later edition is approximate; categories and
divisions are stable.
"""

from __future__ import annotations

import json
import random
import tarfile
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

from ..config import home
from ..scheme.codes import normalize_code
from .cases import EvalCase, images_dir

L3D_URL = "https://zenodo.org/api/records/5771006/files/L3D%20dataset.tar/content"
L3D_SIZE = 12_047_102_976
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_BLOCK = 512


def l3d_dir(root: Path | None = None) -> Path:
    return (root or home()) / "l3d"


def fetch_range(start: int, end: int | None, target: Path, *, timeout: float = 600.0) -> Path:
    """Download bytes ``start..end`` (inclusive; ``None`` = to the end) of the tar."""
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Range": f"bytes={start}-{'' if end is None else end}"}
    with httpx.stream("GET", L3D_URL, headers=headers, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    return target


def scan_tar(data: bytes) -> Iterator[tuple[int, str, int]]:
    """``(offset, name, size)`` of every ustar header in a 512-aligned byte buffer."""
    pos = 0
    while pos + _BLOCK <= len(data):
        hdr = data[pos : pos + _BLOCK]
        if hdr[257:262] != b"ustar":
            pos += _BLOCK
            continue
        name = hdr[0:100].split(b"\0")[0].decode(errors="replace")
        prefix = hdr[345:500].split(b"\0")[0].decode(errors="replace")
        size_field = hdr[124:136].split(b"\0")[0].strip()
        size = int(size_field, 8) if size_field else 0
        yield pos, f"{prefix}/{name}" if prefix else name, size
        pos += _BLOCK + ((size + _BLOCK - 1) // _BLOCK) * _BLOCK


def l3d_labels(*, root: Path | None = None, force: bool = False, tail_mb: int = 120) -> dict:
    """``{image stem: record}`` from ``results.json`` at the end of the tar, cached.

    The archive ends with one ``output_<year>.json`` per year and ``results.json``,
    the filtered union of them (116 MB, about 770k records); the last 120 MB of the
    tar hold the latter whole.
    """
    cache = l3d_dir(root) / "labels.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    tail = l3d_dir(root) / "tail.tar"
    size = tail_mb * 1024 * 1024
    if not tail.exists() or tail.stat().st_size < size or force:
        fetch_range(L3D_SIZE - size, None, tail)
    data = tail.read_bytes()
    found = [(o, n, s) for o, n, s in scan_tar(data) if n.lower().endswith(".json")]
    if not found:
        raise RuntimeError(f"no JSON entry in the last {tail_mb} MB of the L3D tar")
    results = [f for f in found if Path(f[1]).name == "results.json"]
    records: list[dict] = []
    for offset, _name, size_ in results or found:
        body = data[offset + _BLOCK : offset + _BLOCK + size_]
        loaded = json.loads(body)
        records.extend(loaded if isinstance(loaded, list) else loaded.get("data", []))
    labels = {}
    for r in records:
        codes = r.get("vienna_codes") or []
        if isinstance(codes, str):
            codes = [codes]
        stem = Path(r["file"]).stem.lower()
        labels[stem] = {"codes": codes, "text": r.get("text"), "year": r.get("year")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(labels))
    return labels


def l3d_head_images(n: int, *, root: Path | None = None, chunk_mb: int = 32) -> list[Path]:
    """The first ``n`` images of the tar, extracted under ``<home>/images/l3d``."""
    out_dir = images_dir(root) / "l3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(p for p in out_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if len(have) >= n:
        return have[:n]
    head = l3d_dir(root) / "head.tar"
    size = chunk_mb * 1024 * 1024
    while True:
        if not head.exists() or head.stat().st_size < size:
            fetch_range(0, size - 1, head)
        extracted = _extract_images(head, out_dir)
        if len(extracted) >= n or size >= L3D_SIZE:
            return extracted[:n]
        size *= 2


def _extract_images(tar_path: Path, out_dir: Path) -> list[Path]:
    out = []
    with tarfile.open(tar_path, "r", ignore_zeros=True) as tf:
        try:
            for member in tf:
                if not member.isfile() or not member.name.lower().endswith(_IMAGE_SUFFIXES):
                    continue
                target = out_dir / Path(member.name).name.lower()
                if not target.exists():
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    data = fh.read()
                    if len(data) != member.size:
                        break  # truncated at the end of the range
                    target.write_bytes(data)
                out.append(target)
        except (tarfile.ReadError, EOFError):
            pass
    return sorted(out)


def build_l3d_cases(
    n: int = 300,
    *,
    seed: int = 0,
    root: Path | None = None,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[EvalCase]:
    """``n`` labelled images drawn with ``seed`` from the front of the archive."""
    labels = l3d_labels(root=root, force=force)
    pool = l3d_head_images(int(n * 1.5) + 50, root=root)
    labelled = [p for p in pool if p.stem in labels and labels[p.stem]["codes"]]
    rng = random.Random(seed)
    picked = sorted(rng.sample(labelled, min(n, len(labelled))), key=lambda p: p.name)
    cases = []
    for i, p in enumerate(picked, 1):
        rec = labels[p.stem]
        codes = []
        for raw in rec["codes"]:
            try:
                codes.append(normalize_code(str(raw)))
            except ValueError:
                continue
        if not codes:
            continue
        cases.append(
            EvalCase(
                id=f"l3d:{p.stem[:8]}",
                image=str(p.relative_to(images_dir(root))),
                vienna=tuple(dict.fromkeys(codes)),
                edition=None,
                text=rec.get("text") or None,
                source=f"L3D (EUIPO open data, {rec.get('year')})",
                tags=("l3d", f"year-{rec.get('year')}"),
            )
        )
        if progress:
            progress(i, len(picked))
    return cases
