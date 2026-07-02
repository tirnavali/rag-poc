"""Unit test for SearchTool's parliamentary-jargon query expansion wiring.

Mocks VectorSearch entirely (no embeddings/reranker) — only verifies that
SearchTool.search() passes the EXPANDED query text into the underlying vector
search call, closing the "kadük" vs "hükümsüz sayılan" vocabulary gap
deterministically (see src.common.text.expand_parliamentary_synonyms).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agent.tools import SearchTool


def _tool() -> SearchTool:
    config = SimpleNamespace(retrieval=SimpleNamespace(reranker_enabled=False))
    return SearchTool(config, client_pool=MagicMock())


def test_search_expands_known_jargon_term_before_vector_search(monkeypatch):
    tool = _tool()
    fake_search = MagicMock()
    fake_search.search.return_value = []
    fake_spec = SimpleNamespace(doc_type="tutanak")
    monkeypatch.setattr(tool, "_get_search", lambda key: (fake_search, fake_spec))

    tool.search("tutanaklar_nomic_chunk256_768d", "kadük tüm listeyi ver", filters={}, top_k=10)

    sent_query = fake_search.search.call_args.args[0]
    assert "hükümsüz sayılan kanun teklifleri" in sent_query
    assert "kadük" in sent_query
    # Prepended (empirically outperforms appending for this embedding model).
    assert sent_query.index("hükümsüz sayılan kanun teklifleri") < sent_query.index("kadük")


def test_search_does_not_alter_query_without_known_term(monkeypatch):
    tool = _tool()
    fake_search = MagicMock()
    fake_search.search.return_value = []
    fake_spec = SimpleNamespace(doc_type="tutanak")
    monkeypatch.setattr(tool, "_get_search", lambda key: (fake_search, fake_spec))

    tool.search("tutanaklar_nomic_chunk256_768d", "1997 bütçe görüşmeleri", filters={}, top_k=10)

    sent_query = fake_search.search.call_args.args[0]
    assert sent_query == "1997 bütçe görüşmeleri"
