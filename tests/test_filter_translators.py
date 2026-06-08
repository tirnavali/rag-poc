"""Unit tests for the filter translation layer.

Ensures proper decoupling and correct conversion of FilterCriteria into database-specific filter definitions.
"""
import pytest
from src.common.schemas import FilterCriteria
from src.common.filter_translators import BaseFilterTranslator, ChromaFilterTranslator


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
