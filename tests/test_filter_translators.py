"""Unit tests for the filter translation layer.

Ensures proper decoupling and correct conversion of FilterCriteria into database-specific filter definitions.
"""
import pytest
from src.common.schemas import FilterCriteria
from src.common.filter_translators import (
    BaseFilterTranslator,
    ChromaFilterTranslator,
    build_chroma_where,
)
from src.common.author_resolver import AuthorResolver, _tokens


def test_base_translator_abstract():
    """Ensure BaseFilterTranslator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseFilterTranslator()  # type: ignore


def test_chroma_translator_single():
    """ChromaFilterTranslator should translate a single filter to an $eq comparison."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria(year=1996)
    result = translator.translate(filters)
    assert result == {"year": {"$eq": 1996}}


def test_chroma_translator_multiple():
    """ChromaFilterTranslator should translate multiple filters to an $and sequence."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria(year=1996, author="Deniz Baykal", document_type="tutanak")
    result = translator.translate(filters)
    assert result == {
        "$and": [
            {"year": {"$eq": 1996}},
            {
                "$or": [
                    {"author": {"$eq": "Deniz Baykal"}},
                    {"author": {"$eq": "DENİZ BAYKAL"}}
                ]
            },
            {"document_type": {"$eq": "tutanak"}}
        ]
    }


def test_chroma_translator_empty():
    """ChromaFilterTranslator should return None when all criteria are empty."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria()
    result = translator.translate(filters)
    assert result is None


def test_chroma_translator_year_lte():
    """year_lte should produce a $lte condition on the year field."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria(year_lte=2000)
    result = translator.translate(filters)
    assert result == {"year": {"$lte": 2000}}


def test_chroma_translator_year_gte():
    """year_gte should produce a $gte condition on the year field."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria(year_gte=1990)
    result = translator.translate(filters)
    assert result == {"year": {"$gte": 1990}}


def test_chroma_translator_year_range():
    """Combining year_lte and year_gte should produce an $and with both $lte/$gte."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria(year_lte=2000, year_gte=1990)
    result = translator.translate(filters)
    assert result == {
        "$and": [
            {"year": {"$lte": 2000}},
            {"year": {"$gte": 1990}},
        ]
    }


def test_chroma_translator_year_range_with_author():
    """Range filter combined with author should produce a full $and chain."""
    translator = ChromaFilterTranslator()
    filters = FilterCriteria(year_lte=2000, year_gte=1990, author="Önder Sav")
    result = translator.translate(filters)
    assert result == {
        "$and": [
            {"year": {"$lte": 2000}},
            {"year": {"$gte": 1990}},
            {
                "$or": [
                    {"author": {"$eq": "Önder Sav"}},
                    {"author": {"$eq": "ÖNDER SAV"}},
                ]
            },
        ]
    }


# --- AuthorResolver: case-insensitive, Turkish-aware token-subset matching ---

def _seeded_resolver(labels: list[str] | None) -> AuthorResolver:
    """Resolver with a pre-seeded vocab (no ChromaDB read). None = unavailable."""
    r = AuthorResolver()
    r._cache["col"] = labels
    return r


def test_tokens_turkish_case_fold():
    # İ→i, dotless handling, Ş/Ğ etc. — query tokens are a subset of the label's.
    assert _tokens("Recep Tayyip Erdoğan") <= _tokens("BAŞBAKAN RECEP TAYYİP ERDOĞAN")
    assert _tokens("erdoğan") <= _tokens("BAŞBAKAN RECEP TAYYİP ERDOĞAN")


def test_resolver_matches_titled_label():
    r = _seeded_resolver(["BAŞBAKAN RECEP TAYYİP ERDOĞAN", "AHMET KABİL"])
    assert r.resolve("col", "Recep Tayyip Erdoğan") == ["BAŞBAKAN RECEP TAYYİP ERDOĞAN"]
    assert r.resolve("col", "erdoğan") == ["BAŞBAKAN RECEP TAYYİP ERDOĞAN"]


def test_resolver_no_match_returns_empty_list():
    r = _seeded_resolver(["AHMET KABİL"])
    assert r.resolve("col", "Deniz Baykal") == []


def test_resolver_unavailable_vocab_returns_none():
    r = _seeded_resolver(None)
    assert r.resolve("col", "Deniz Baykal") is None


# --- build_chroma_where: author resolved to $in / dropped / default ---

def test_build_where_author_in_when_resolved():
    where = build_chroma_where(
        FilterCriteria(author="Recep Tayyip Erdoğan", document_type="tutanak"),
        "col",
        resolver=_seeded_resolver(["BAŞBAKAN RECEP TAYYİP ERDOĞAN"]),
    )
    assert where == {
        "$and": [
            {"author": {"$in": ["BAŞBAKAN RECEP TAYYİP ERDOĞAN"]}},
            {"document_type": {"$eq": "tutanak"}},
        ]
    }


def test_build_where_drops_author_when_no_match():
    # Vocab available but name absent → author dropped, only year remains.
    where = build_chroma_where(
        FilterCriteria(year=1996, author="Deniz Baykal"),
        "col",
        resolver=_seeded_resolver(["AHMET KABİL"]),
    )
    assert where == {"year": {"$eq": 1996}}


def test_build_where_falls_back_to_eq_when_vocab_unavailable():
    # Resolver returns None → legacy $eq/$or behavior preserved.
    where = build_chroma_where(
        FilterCriteria(author="Deniz Baykal"),
        "col",
        resolver=_seeded_resolver(None),
    )
    assert where == {
        "$or": [
            {"author": {"$eq": "Deniz Baykal"}},
            {"author": {"$eq": "DENİZ BAYKAL"}},
        ]
    }
