import pytest
from src.retriever.query_parser import route_sources


@pytest.mark.parametrize("query,expected", [
    ("meclis tutanaklarında neler konuşuldu", ["minutes"]),
    ("tbmm genel kurulu kararları", ["minutes"]),
    ("gazete kupürleri 1997", ["gazete"]),
    ("köşe yazısı neler anlatıyor", ["gazete"]),
    ("1997 yılındaki yazılar", ["gazete", "minutes"]),
    ("istanbul depremi", ["gazete", "minutes"]),
])
def test_route_sources(query, expected):
    result = route_sources(query.lower())
    assert set(result) == set(expected)
