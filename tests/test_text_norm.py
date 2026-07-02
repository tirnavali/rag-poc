import pytest

from src.common import text as text_module
from src.common.text import expand_parliamentary_synonyms, normalize_tr, extract_relevant_windows


@pytest.fixture(autouse=True)
def _isolated_approved_synonyms_cache(monkeypatch):
    """Hermetic by default: no test in this file should touch the real DB or
    leak cached state across tests. Tests that specifically exercise the
    DB-merge behavior override `src.api.db.list_approved_term_synonyms` and/or
    the cache timestamp themselves."""
    monkeypatch.setattr(text_module, "_approved_cache", {})
    monkeypatch.setattr(text_module, "_approved_cache_at", 0.0)
    monkeypatch.setattr("src.api.db.list_approved_term_synonyms", lambda: {})


def test_normalize_lowercase():
    assert normalize_tr("İSTANBUL") == "istanbul"


def test_normalize_turkish_i():
    assert normalize_tr("IŞIK") == "ışık"


def test_normalize_strips_punctuation():
    result = normalize_tr("merhaba, dünya!")
    assert "," not in result
    assert "!" not in result


def test_window_returns_text_around_match():
    text = "a " * 100 + "hedef kelime" + " b" * 100
    result = extract_relevant_windows(text, "hedef kelime")
    assert "hedef kelime" in result


def test_window_fallback_when_no_match():
    text = "x" * 500
    result = extract_relevant_windows(text, "nomatch")
    assert len(result) <= 3000
    assert result == text[:3000]


def test_window_respects_max_total():
    text = "hedef " * 5000
    result = extract_relevant_windows(text, "hedef", max_total=500)
    assert len(result) <= 600


def test_expand_parliamentary_synonyms_prepends_official_term():
    """'kadük' rarely matches the corpus's own phrasing ("hükümsüz sayılan kanun
    teklifleri") in embedding space (verified empirically) — prepending the
    official phrase closes the gap. Prepending (not appending) and using the
    FULL phrase (not just "hükümsüz sayılan") both matter empirically: the
    leading terms carry more weight in the pooled query vector for this
    embedding model, and the bare word alone under-performs the full phrase."""
    result = expand_parliamentary_synonyms("kadük tüm listeyi ver")
    assert "kadük tüm listeyi ver" in result
    assert "hükümsüz sayılan kanun teklifleri" in result
    # Prepended, not appended: the synonym comes first.
    assert result.index("hükümsüz sayılan kanun teklifleri") < result.index("kadük tüm listeyi ver")


def test_expand_parliamentary_synonyms_case_insensitive():
    result = expand_parliamentary_synonyms("KADÜK olan teklifler")
    assert "hükümsüz sayılan kanun teklifleri" in result


def test_expand_parliamentary_synonyms_noop_without_known_term():
    query = "1997 bütçe görüşmeleri"
    assert expand_parliamentary_synonyms(query) == query


def test_expand_parliamentary_synonyms_no_duplicate_when_already_present():
    """If a synonym phrase is already (exactly) in the query, don't add it again."""
    query = "hükümsüz sayılan kanun teklifleri hakkında kadük teklifler"
    result = expand_parliamentary_synonyms(query)
    assert result.count("hükümsüz sayılan kanun teklifleri") == 1


def test_expand_parliamentary_synonyms_merges_db_approved_terms(monkeypatch):
    """Human-approved discoveries (Öğrenilen Terimler panel) apply on top of
    the static dict — the dynamic, self-populating counterpart to hardcoding
    every term manually."""
    monkeypatch.setattr("src.api.db.list_approved_term_synonyms",
                         lambda: {"içtihat": ["yargı kararı emsali"]})

    result = expand_parliamentary_synonyms("içtihat nedir")

    assert "yargı kararı emsali" in result
    assert result.index("yargı kararı emsali") < result.index("içtihat nedir")


def test_expand_parliamentary_synonyms_static_and_approved_both_apply(monkeypatch):
    """A query containing both a statically-known term and a DB-approved term
    gets synonyms from both sources — they merge, not override each other."""
    monkeypatch.setattr("src.api.db.list_approved_term_synonyms",
                         lambda: {"içtihat": ["yargı kararı emsali"]})

    result = expand_parliamentary_synonyms("kadük ve içtihat kavramları")

    assert "hükümsüz sayılan kanun teklifleri" in result
    assert "yargı kararı emsali" in result


def test_expand_parliamentary_synonyms_caches_within_ttl(monkeypatch):
    """A second call within the TTL window must NOT hit the DB again."""
    calls = {"n": 0}

    def _counting_fetch():
        calls["n"] += 1
        return {}

    monkeypatch.setattr("src.api.db.list_approved_term_synonyms", _counting_fetch)
    monkeypatch.setattr(text_module, "_approved_cache_at", 0.0)  # force one refresh

    expand_parliamentary_synonyms("kadük tüm listeyi ver")
    expand_parliamentary_synonyms("kadük tüm listeyi ver")
    expand_parliamentary_synonyms("kadük tüm listeyi ver")

    assert calls["n"] == 1  # cached after the first refresh


def test_expand_parliamentary_synonyms_refetches_after_ttl_expires(monkeypatch):
    calls = {"n": 0}

    def _counting_fetch():
        calls["n"] += 1
        return {}

    monkeypatch.setattr("src.api.db.list_approved_term_synonyms", _counting_fetch)
    monkeypatch.setattr(text_module, "_approved_cache_at", 0.0)

    expand_parliamentary_synonyms("kadük tüm listeyi ver")
    assert calls["n"] == 1

    # Simulate TTL having elapsed by rewinding the cache timestamp again.
    monkeypatch.setattr(text_module, "_approved_cache_at", 0.0)
    expand_parliamentary_synonyms("kadük tüm listeyi ver")
    assert calls["n"] == 2


def test_expand_parliamentary_synonyms_falls_back_gracefully_when_db_unavailable(monkeypatch):
    """CLI-only callers (no web API/db.py context) must not crash — the static
    dict alone still works."""
    def _raise():
        raise RuntimeError("db unavailable")
    monkeypatch.setattr("src.api.db.list_approved_term_synonyms", _raise)

    result = expand_parliamentary_synonyms("kadük tüm listeyi ver")

    assert "hükümsüz sayılan kanun teklifleri" in result
