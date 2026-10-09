from pathlib import Path

from image2vienna.classifier import ViennaClassifier
from image2vienna.eval import EvalCase, evaluate, load_cases, save_cases
from image2vienna.eval.describe import describe_cases, description_key


def _png(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (8, 8), "white").save(path)
    return path


def test_classifier_with_fixed_describer(mini_home, tmp_path):
    img = _png(tmp_path / "logo.png")
    (tmp_path / "logo.txt").write_text("three stars above a crescent moon")
    clf = ViennaClassifier("latest", model="hash:64", describer=f"fixed:{tmp_path}", root=mini_home)
    assert clf.edition == "10"
    out = clf.classify(img, level="section", top_k=4)
    assert clf.last_description == "three stars above a crescent moon"
    assert {m.code for m in out} >= {"1.1.4", "1.7.6"}
    assert clf.classify_text("circles", level="division")[0].code == "26.1"


def test_describe_cases_caches_and_eval_scores(mini_home, tmp_path):
    images = mini_home / "images"
    images.mkdir()
    _png(images / "a.png")
    (images / "a.txt").write_text("three stars")
    _png(images / "b.png")
    (images / "b.txt").write_text("heads, busts of men")
    cases_path = tmp_path / "cases.jsonl"
    save_cases(
        [
            EvalCase(id="a", image="a.png", vienna=("1.1.4", "1.1.1")),
            EvalCase(id="b", image="b.png", vienna=("2.1.1",)),
            EvalCase(id="c", image="b.png", vienna=("2.1",)),  # division-only gold
        ],
        cases_path,
    )
    import os

    os.environ["IMAGE2VIENNA_HOME"] = str(mini_home)
    try:
        spec = f"fixed:{images}"
        done, total = describe_cases(cases_path, spec)
        assert (done, total) == (3, 3)
        cases = load_cases(cases_path)
        key = description_key(spec)
        assert cases[0].descriptions[key] == "three stars"
        assert describe_cases(cases_path, spec) == (0, 3)  # cached

        clf = ViennaClassifier("10", model="hash:64", root=mini_home)
        result = evaluate(
            cases,
            lambda c: clf.classify_text(c.descriptions[key], level="section", top_k=5),
            level="section",
            top_k=5,
        )
        assert result.n == 3
        assert result.scored["section"] == 2 and result.scored["division"] == 3
        assert result.rate("section", 1) == 1.0
        assert result.rate("category", 1) == 1.0
        assert 0 < result.recall_at("section", 5) <= 1.0
        assert "hit@1" in result.table()
    finally:
        del os.environ["IMAGE2VIENNA_HOME"]


def test_sentence_chunking_and_exclusion(mini_home):
    from image2vienna.eval.harness import exclude_gold
    from image2vienna.textnorm import split_sentences

    assert split_sentences("Three stars. Below, a crescent moon! Red and blue.") == [
        "Three stars.",
        "Below, a crescent moon!",
        "Red and blue.",
    ]
    clf = ViennaClassifier("10", model="hash:64", root=mini_home)
    text = "Three stars at the top. A crescent moon below."
    top_max = {m.code for m in clf.classify_text(text, top_k=4, chunking="max")}
    assert top_max >= {"1.1.4", "1.7.6"}
    assert clf.embed_text(text, chunking="max").shape[0] == 2
    assert clf.embed_text(text, chunking="mean").shape == (64,)
    cases = [
        EvalCase(id="x", image="a.png", vienna=("29.1.1", "1.1.4")),
        EvalCase(id="y", image="a.png", vienna=("29.1.4",)),
    ]
    kept = exclude_gold(cases, ("29",))
    assert [c.vienna for c in kept] == [("1.1.4",)]


def test_describer_names_resolve_to_specs(mini_home):
    from image2vienna.config import HEAVY_DESCRIBER, LIGHT_DESCRIBER

    light = ViennaClassifier("10", model="hash:64", describer="light", root=mini_home)
    heavy = ViennaClassifier("10", model="hash:64", describer="heavy", root=mini_home)
    assert light._describer_spec == LIGHT_DESCRIBER and heavy._describer_spec == HEAVY_DESCRIBER
    assert (
        ViennaClassifier("10", model="hash:64", root=mini_home)._describer_spec == HEAVY_DESCRIBER
    )
