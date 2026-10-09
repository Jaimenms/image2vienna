import numpy as np

from image2vienna.search import Beam, SearchParams, Weights, search


def q(embedder, text):
    return embedder.embed_query(text)


def test_flat_cosine_finds_the_matching_section(mini_index, embedder):
    out = search(mini_index, q(embedder, "three stars"), SearchParams(level="section", top_k=3))
    assert out[0].code == "1.1.4" and out[0].auxiliary and out[0].pretty == "1.1.4"
    assert out[0].padded == "01.01.04"


def test_levels_are_honoured(mini_index, embedder):
    for level in ("category", "division", "section"):
        out = search(mini_index, q(embedder, "crescent moon"), SearchParams(level=level, top_k=5))
        assert out and all(m.level == level for m in out)
    assert (
        search(mini_index, q(embedder, "crescent moon"), SearchParams(level="division"))[0].code
        == "1.7"
    )


def test_principal_only_hides_auxiliary_sections(mini_index, embedder):
    params = SearchParams(level="section", top_k=10, principal_only=True)
    out = search(mini_index, q(embedder, "one star"), params)
    assert out and not any(m.auxiliary for m in out)


def test_beam_width_prunes_branches(mini_index, embedder):
    narrow = SearchParams(level="section", top_k=10, beam=Beam(category=1, division=1))
    out = search(mini_index, q(embedder, "circles stars"), narrow)
    divisions = {m.code.rsplit(".", 1)[0] for m in out}
    assert len(divisions) == 1


def test_path_support_and_subtree_support_shapes(mini_index, embedder):
    params = SearchParams(level="section", top_k=3, weights=Weights(own=0.6, path=0.2, subtree=0.2))
    out = search(mini_index, q(embedder, "stars"), params)
    assert out and all(
        0 <= m.path_support <= 1.0001 and m.subtree_support >= m.similarity - 1e-6 for m in out
    )


def test_auto_level_descends_while_evidence_holds(mini_index, embedder):
    out = search(mini_index, q(embedder, "stars comets"), SearchParams(level="auto", top_k=3))
    assert out and out[0].code.startswith("1.1")


def test_dedupe_branches_removes_ancestor_and_descendant(mini_index, embedder):
    out = search(
        mini_index, q(embedder, "stars"), SearchParams(level="auto", top_k=5, auto_margin=1.0)
    )
    codes = [m.code for m in out]
    for a in codes:
        for b in codes:
            assert a == b or not (b.startswith(a + ".") or a.startswith(b + "."))


def test_multi_vector_query_scores_by_best_chunk(mini_index, embedder):
    stack = np.stack([q(embedder, "crescent moon"), q(embedder, "three stars")])
    out = search(mini_index, stack, SearchParams(level="section", top_k=2))
    assert {m.code for m in out} == {"1.7.6", "1.1.4"}


def test_exclude_codes_masks_a_subtree(mini_index, embedder):
    params = SearchParams(level="section", top_k=10, exclude_codes=("1",))
    out = search(mini_index, q(embedder, "three stars"), params)
    assert out and not any(m.code.startswith("1.") for m in out)
