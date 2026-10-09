from pathlib import Path

import pytest

from image2vienna.embeddings.hashing import HashEmbedder
from image2vienna.index import build_index
from image2vienna.scheme import SchemeTable, parse_scheme

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_nodes():
    return parse_scheme(FIXTURES / "mini_vienna.xml")


@pytest.fixture
def mini_scheme(mini_nodes):
    return SchemeTable.from_nodes(mini_nodes)


@pytest.fixture
def embedder():
    return HashEmbedder(64)


@pytest.fixture
def mini_index(mini_scheme, embedder):
    index, _ = build_index(mini_scheme, embedder, edition="10", lang="EN")
    return index


@pytest.fixture
def mini_home(mini_index, tmp_path):
    """An image2vienna home with the mini index and scheme table written to disk."""
    from image2vienna.index import index_path, scheme_table_path

    mini_index.scheme.write(scheme_table_path("10", "EN", tmp_path))
    mini_index.write(index_path("10", "EN", "hash:64", root=tmp_path))
    return tmp_path
