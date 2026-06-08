"""Unit tests for CrossEncoderReranker model selection (no model load)."""
from src.config import settings
from src.retriever.reranker import CrossEncoderReranker


def test_default_model_name_falls_back_to_settings():
    r = CrossEncoderReranker()
    assert r._model_name == settings.RERANK_MODEL


def test_explicit_model_name_overrides():
    r = CrossEncoderReranker("cross-encoder/some-other-model")
    assert r._model_name == "cross-encoder/some-other-model"


def test_empty_candidates_returns_empty_without_loading_model():
    # rerank() short-circuits on empty candidates before touching the model,
    # so this stays offline.
    r = CrossEncoderReranker("cross-encoder/never-loaded")
    assert r.rerank("q", [], top_n=5) == []
