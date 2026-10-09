from image2vienna.scheme import clean_title


def test_depth_first_order_and_counts(mini_nodes):
    codes = [n.code for n in mini_nodes]
    assert codes[:4] == ["1", "1.1", "1.1.1", "1.1.15"]
    assert codes.index("1.1") < codes.index("1.1.2") and codes.index("1.1.17") < codes.index(
        "1.1.2"
    )
    assert sum(n.level == "category" for n in mini_nodes) == 4
    assert sum(n.level == "division" for n in mini_nodes) == 5
    assert sum(n.level == "section" for n in mini_nodes) == 13


def test_auxiliary_sections(mini_nodes):
    by = {n.code: n for n in mini_nodes}
    assert by["1.1.2"].auxiliary and by["1.1.2"].parent == "1.1"
    assert by["1.1.2"].associated == ("1.1.1", "1.1.15")
    assert not by["1.1.1"].auxiliary and by["1.1.1"].associated == ()


def test_titles_sentence_cased_and_refs_stripped(mini_nodes):
    by = {n.code: n for n in mini_nodes}
    assert by["1"].title == "Celestial bodies, natural phenomena, geographical maps"
    assert by["1.1"].title == "Stars, comets"
    assert by["2.1.8"].title == "Acrobats, athletes, dancers, men engaging in sport"
    assert by["1.1.25"].title == "Other representations of stars"
    assert by["1.7.6"].title == "Crescent moon, half-moon"  # <img> dropped


def test_notes_kept_on_the_node(mini_nodes):
    by = {n.code: n for n in mini_nodes}
    assert by["1.1"].notes[0] == "Including stars which indicate military rank."
    assert "Not including sparks (1.15.7)" in by["1.1"].notes[1]
    assert by["2"].notes == ("Inscriptions representing a human being will be placed in 27.3.1.",)
    assert by["2.1.4"].notes[0].startswith("Including")


def test_leaf_division(mini_nodes):
    by = {n.code: n for n in mini_nodes}
    assert by["28.1"].level == "division" and by["28.1"].parent == "28"


def test_clean_title_cases():
    assert clean_title("SUN") == "Sun"
    assert clean_title("Sun with animals") == "Sun with animals"
    assert clean_title("Shields (except 24.1.5)") == "Shields"
    assert clean_title("Heads of animals of division 3.9") == "Heads of animals of division 3.9"
