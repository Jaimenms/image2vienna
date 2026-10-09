from image2vienna.index import ViennaIndex, build_index, index_path, scheme_table_path
from image2vienna.scheme import SchemeTable


def test_path_text_is_the_full_chain(mini_scheme):
    assert (
        mini_scheme.text_of("1.1.2")
        == "Celestial bodies, natural phenomena, geographical maps > Stars, comets > One star"
    )
    assert mini_scheme.path_of("1.1.2") == ("1", "1.1", "1.1.2")


def test_notes_rendering_keeps_only_positive_notes(mini_scheme):
    with_notes = mini_scheme.text_of("2.1.4", notes=True)
    assert with_notes.endswith("costume. Including, for example, cowboys.")
    assert "Not including" not in with_notes
    assert mini_scheme.text_of("1.1.1", notes=True) == mini_scheme.text_of("1.1.1")


def test_scheme_roundtrip(mini_scheme, tmp_path):
    p = tmp_path / "s.parquet"
    mini_scheme.write(p)
    back = SchemeTable.read(p)
    assert [n.code for n in back] == [n.code for n in mini_scheme]
    assert back.node("1.1.2").associated == ("1.1.1", "1.1.15")
    assert back.node("1.1").notes == mini_scheme.node("1.1").notes
    assert back.texts() == mini_scheme.texts()


def test_index_roundtrip_and_hierarchy(mini_index, tmp_path):
    p = tmp_path / "i.parquet"
    mini_index.write(p)
    back = ViennaIndex.read(p, mini_index.scheme)
    assert back.meta.edition == "10" and back.meta.dim == 64 and back.meta.notes is False
    assert (back.vectors == mini_index.vectors).all()
    i = back.position["1.1.2"]
    assert back.codes[back.parent_idx[i]] == "1.1"
    assert back.children[back.position["1"]] == [back.position["1.1"], back.position["1.7"]]


def test_incremental_build_reuses_unchanged_vectors(mini_scheme, embedder, mini_nodes):
    first, _ = build_index(mini_scheme, embedder, edition="9", lang="EN")
    import dataclasses

    nodes = [
        dataclasses.replace(n, title="Two stars") if n.code == "1.1.4" else n for n in mini_nodes
    ]
    second, report = build_index(
        SchemeTable.from_nodes(nodes), embedder, edition="10", lang="EN", previous=first
    )
    assert report.previous == "9" and report.changed == ["1.1.4"] and report.added == []
    assert report.reused == len(mini_scheme) - 1
    i = second.position["1.1.1"]
    assert (second.vectors[i] == first.vectors[first.position["1.1.1"]]).all()


def test_notes_index_has_its_own_file_name(tmp_path):
    assert index_path("10", "EN", "hash:64", root=tmp_path).name == "vienna10_en_hash-64.parquet"
    assert (
        index_path("10", "EN", "hash:64", notes=True, root=tmp_path).name
        == "vienna10_en_hash-64_notes.parquet"
    )
    assert scheme_table_path("10", "EN", tmp_path).name == "vienna10_en.parquet"
