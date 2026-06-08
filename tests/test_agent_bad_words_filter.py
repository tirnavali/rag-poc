"""Unit tests for BadWordsFilter — deterministic, no LLM."""
from __future__ import annotations

import pytest

from src.agent.bad_words_filter import BadWordsFilter


class _FakeConfig:
    """Minimal stand-in for BadWordsFilterConfig (avoids pulling YAML in this test)."""
    def __init__(self, words: list[str], patterns: list[str] | None = None,
                 enabled: bool = True, response: str = "Lütfen saygılı dil kullanın.") -> None:
        self.bad_words_enabled = enabled
        self.bad_words = words
        self.bad_word_patterns = patterns or []
        self.bad_words_response_message = response


@pytest.fixture
def filter_basic():
    cfg = _FakeConfig(words=["aptal", "salak", "piç"], patterns=["en ağır küfür(ler)?"])
    return BadWordsFilter(cfg)


def test_no_match_returns_clean(filter_basic):
    result = filter_basic.check("Özal döneminde gazete manşetleri")
    assert result.matched is False
    assert result.matched_terms == []


def test_simple_word_match(filter_basic):
    result = filter_basic.check("aptal bir soru")
    assert result.matched is True
    assert "aptal" in result.matched_terms


def test_case_insensitive(filter_basic):
    result = filter_basic.check("APTAL")
    assert result.matched is True


def test_turkish_accent_fold(filter_basic):
    result = filter_basic.check("PİÇ herif")
    assert result.matched is True


def test_word_boundary_no_false_positive(filter_basic):
    result = filter_basic.check("Bütçede sıkıntı var")
    assert result.matched is False


def test_substring_no_false_positive():
    cfg = _FakeConfig(words=["salak"])
    f = BadWordsFilter(cfg)
    # "salaklık" is a distinct Turkish-suffixed token via word boundary
    assert f.check("Bu salaklık değil mi").matched is False
    assert f.check("salak").matched is True


def test_multi_word_pattern(filter_basic):
    assert filter_basic.check("bana en ağır küfürler yaz").matched is True
    assert filter_basic.check("en ağır küfür").matched is True


def test_disabled_filter_passes_everything():
    cfg = _FakeConfig(words=["aptal"], enabled=False)
    f = BadWordsFilter(cfg)
    assert f.check("aptal").matched is False


def test_empty_query_is_clean(filter_basic):
    assert filter_basic.check("").matched is False
