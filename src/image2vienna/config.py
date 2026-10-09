"""Package-wide constants and locations."""

from __future__ import annotations

import os
from pathlib import Path

#: Hierarchy levels of the Vienna Classification, top to bottom. Codes read
#: ``category.division.section`` (``1.1.2``); auxiliary sections ("A" sections) share
#: the section level and numbering space of their division.
LEVELS: tuple[str, ...] = ("category", "division", "section")

#: Special level meaning "descend while the evidence supports it".
AUTO_LEVEL = "auto"

#: Editions published by WIPO on nivilo.wipo.int, with the year they entered into force.
EDITIONS: dict[str, str] = {"7": "2013", "8": "2018", "9": "2023", "10": "2026"}
DEFAULT_EDITION = "10"

#: Languages WIPO publishes the classification in.
WIPO_LANGS: tuple[str, ...] = ("EN", "FR")
DEFAULT_LANG = "EN"

#: Same embedder as text2ipc so the two studies compare: multilingual, 768 dimensions,
#: 512 tokens. Override with ``IMAGE2VIENNA_MODEL`` or the ``model=`` argument.
DEFAULT_MODEL = "st:intfloat/multilingual-e5-base"

#: Vision-language model that turns the image into a description of its figurative
#: elements. Served by Ollama; ``ollama pull qwen2.5vl:7b`` (6 GB) first.
DEFAULT_DESCRIBER = "ollama:qwen2.5vl:7b"

#: Hugging Face model repository holding the published English index.
DEFAULT_HF_REPO = "jaimenms/image2vienna-en"

#: The online publication of the classification; ``xml/<lang>/full.xml`` is the
#: reference file the pages are generated from.
NIVILO_BASE_URL = "https://nivilo.wipo.int"


def home() -> Path:
    """Directory holding downloaded schemes, eval images and built indexes.

    Resolution order:

    1. ``IMAGE2VIENNA_HOME``.
    2. ``<repo>/data`` when the current directory (or one of its parents) is the
       image2vienna study repository and that directory exists.
    3. ``$XDG_CACHE_HOME/image2vienna`` or ``~/.cache/image2vienna``.
    """
    if env := os.environ.get("IMAGE2VIENNA_HOME"):
        return Path(env).expanduser()
    if (repo_data := _repo_data_dir()) is not None:
        return repo_data
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "image2vienna"


def _repo_data_dir(cwd: Path | None = None) -> Path | None:
    for candidate in [cwd or Path.cwd(), *(cwd or Path.cwd()).parents]:
        pyproject = candidate / "pyproject.toml"
        try:
            is_repo = pyproject.is_file() and 'name = "image2vienna"' in pyproject.read_text()
        except OSError:
            return None
        if is_repo:
            data = candidate / "data"
            return data if data.is_dir() else None
    return None


def default_model() -> str:
    return os.environ.get("IMAGE2VIENNA_MODEL", DEFAULT_MODEL)


def default_describer() -> str:
    return os.environ.get("IMAGE2VIENNA_DESCRIBER", DEFAULT_DESCRIBER)
