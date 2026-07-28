"""Unit tests for the opt-in auto_date_where_filter() helper (src/retriever/filters.py)."""
from src.retriever.filters import auto_date_where_filter


def test_single_year_builds_eq_filter():
    where, parsed = auto_date_where_filter("1996 yılında ne oldu?")
    assert where == {"year": {"$eq": 1996}}
    assert "1996" in parsed["years"]


def test_multiple_years_builds_or_filter():
    where, _ = auto_date_where_filter("1996 ve 1997 arasında neler oldu?")
    assert "$or" in where


def test_no_year_returns_none():
    where, parsed = auto_date_where_filter("Kardak kayalıkları")
    assert where is None
    assert parsed["years"] == []
