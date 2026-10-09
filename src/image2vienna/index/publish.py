"""Fetch a prebuilt index from a Hugging Face model repository made by ``i2vienna hf-export``.

Building an index needs the embedder; end users fetch the Parquet tables (``index/``
and ``scheme/``) from the Hub instead. The repository's ``image2vienna.json`` says
which edition, language and model they belong to.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import home


def download_from_hf(
    repo_id: str, *, root: Path | None = None, revision: str | None = None, token=None
) -> tuple[Path, dict]:
    """Copy ``index/*.parquet`` and ``scheme/*.parquet`` from a Hub model repo into home.

    Returns the path of the default index and the repo's ``image2vienna.json``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    root = root or home()
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="model", revision=revision)
    wanted = [f for f in files if f.endswith(".parquet") and f.split("/")[0] in {"index", "scheme"}]
    if "image2vienna.json" not in files or not wanted:
        raise FileNotFoundError(f"{repo_id}: no image2vienna.json or parquet tables found")
    cfg_path = hf_hub_download(repo_id, "image2vienna.json", revision=revision, token=token)
    cfg = json.loads(Path(cfg_path).read_text())
    index_target: Path | None = None
    for f in sorted(wanted):
        cached = Path(hf_hub_download(repo_id, f, revision=revision, token=token))
        target = root / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        if f.startswith("index/") and not f.endswith("_notes.parquet"):
            index_target = target
    if index_target is None:
        index_target = root / sorted(wanted)[0]
    return index_target, cfg
