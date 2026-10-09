import json
from pathlib import Path

from image2vienna.index import publish


def test_download_from_hf_copies_index_and_scheme(mini_home, tmp_path, monkeypatch):
    # Fake Hub: files come from mini_home, laid out like an hf-export repository.
    repo = tmp_path / "repo"
    (repo / "index").mkdir(parents=True)
    (repo / "scheme").mkdir()
    for sub in ("index", "scheme"):
        for f in (mini_home / sub).glob("*.parquet"):
            (repo / sub / f.name).write_bytes(f.read_bytes())
    (repo / "image2vienna.json").write_text(
        json.dumps({"edition": "10", "lang": "EN", "model": "hash:64", "notes": False})
    )
    files = ["README.md", "handler.py", "image2vienna.json"] + [
        f"{p.parent.name}/{p.name}" for p in repo.glob("*/*.parquet")
    ]

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            return files

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, revision=None, token=None: str(repo / filename),
    )
    target = tmp_path / "home"
    path, cfg = publish.download_from_hf("someone/image2vienna-en", root=target)
    assert path == target / "index" / "vienna10_en_hash-64.parquet"
    assert (target / "scheme" / "vienna10_en.parquet").exists()
    assert cfg["model"] == "hash:64"
    from image2vienna.classifier import ViennaClassifier

    clf = ViennaClassifier("latest", model="hash:64", root=Path(target))
    assert clf.classify_text("crescent moon", level="division", top_k=1)[0].code == "1.7"
