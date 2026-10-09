import json
import sys

from image2vienna.hf import export_hf_repo


def test_export_and_handler_roundtrip(mini_home, tmp_path, monkeypatch):
    repo = export_hf_repo(
        tmp_path / "repo", edition="10", lang="EN", model="hash:64", root=mini_home
    )

    assert (repo / "handler.py").exists()
    assert (repo / "image2vienna" / "classifier.py").exists()
    assert not (repo / "image2vienna" / "web").exists()
    assert (repo / "index" / "vienna10_en_hash-64.parquet").exists()
    assert (repo / "scheme" / "vienna10_en.parquet").exists()
    cfg = json.loads((repo / "image2vienna.json").read_text())
    assert cfg == {"edition": "10", "lang": "EN", "model": "hash:64", "notes": False}
    card = (repo / "README.md").read_text()
    assert "pipeline_tag: text-classification" in card and "| `exclude_codes` |" in card
    # hash:64 -> "64"; a real embedder gives its Hub id. The vision model follows.
    assert (
        "base_model:\n  - 64\n  - microsoft/Florence-2-base-ft\nbase_model_relation: merge\n"
        in card
    )

    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.delenv("IMAGE2VIENNA_DESCRIBER", raising=False)
    sys.modules.pop("handler", None)
    import handler

    h = handler.EndpointHandler(str(repo))
    out = h({"inputs": "three stars", "parameters": {"top_k": 3}})
    assert out[0]["code"] == "1.1.4" and out[0]["padded"] == "01.01.04" and out[0]["auxiliary"]
    assert set(out[0]) >= {"code", "level", "score", "similarity", "title", "path"}
    batch = h({"inputs": ["crescent moon", "circles"], "parameters": {"level": "division"}})
    assert [b[0]["code"] for b in batch] == ["1.7", "26.1"]
    assert h({"inputs": "stars", "parameters": {"bogus": 1, "top_k": 1, "exclude_codes": ["26"]}})
    import pytest

    with pytest.raises(ValueError, match="no vision model"):
        h({"inputs": {"image": "aGVsbG8="}})
