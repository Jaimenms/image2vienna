import pytest

from image2vienna.scheme import format_code, level_of_code, normalize_code, parse_cfe, truncate_code


def test_normalize():
    assert normalize_code("01.01.02") == "1.1.2"
    assert normalize_code("1.1.2") == "1.1.2"
    assert normalize_code("A 1.1.2") == "1.1.2"
    assert normalize_code("26.01") == "26.1"
    assert normalize_code("29") == "29"
    for bad in ("", "x", "1.2.3.4", "0.1", "1.0"):
        with pytest.raises(ValueError):
            normalize_code(bad)


def test_levels_and_truncation():
    assert level_of_code("1") == "category"
    assert level_of_code("1.1") == "division"
    assert level_of_code("1.1.2") == "section"
    assert truncate_code("1.1.2", "division") == "1.1"
    assert truncate_code("1.1", "section") == "1.1"


def test_format():
    assert format_code("1.1.2") == "1.1.2"
    assert format_code("1.1.2", padded=True) == "01.01.02"
    assert format_code("26.1", padded=True) == "26.01"


def test_parse_cfe():
    assert parse_cfe("CFE 1.1.2,10,25; 1.15.17; 2.9.1") == [
        "1.1.2",
        "1.1.10",
        "1.1.25",
        "1.15.17",
        "2.9.1",
    ]
    assert parse_cfe("3.9.16-18") == ["3.9.16", "3.9.17", "3.9.18"]
    assert parse_cfe("CFE (10) 1.1; 1.15") == ["1.1", "1.15"]
