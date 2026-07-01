"""Unit tests for the LLM client pool (offline — no Ollama connection)."""
import pytest

from src.common.llm_client_pool import BlockClient, LLMClientPool
from src.config.pipeline_loader import load_pipeline_config


def _pool() -> LLMClientPool:
    return LLMClientPool.from_config(load_pipeline_config())


def test_get_client_is_lazy_and_cached():
    pool = _pool()
    c = pool.get_client("fast-01")
    assert isinstance(c, BlockClient)
    assert pool.get_client("fast-01") is c  # same cached instance


def test_get_model_for_block():
    pool = _pool()
    model = pool.get_model_for_block("fast-01", "planner")
    assert isinstance(model, str) and model


def test_get_model_for_block_bad_key_raises():
    pool = _pool()
    with pytest.raises(ValueError):
        pool.get_model_for_block("fast-01", "no_such_role")


def test_get_host():
    pool = _pool()
    assert pool.get_host("fast-01").startswith("http")


def test_health_check_all_only_initialized():
    pool = _pool()
    # No clients touched yet → empty mapping, no network calls.
    assert pool.health_check_all() == {}


def test_config_keep_alive_default():
    cfg = load_pipeline_config()
    assert cfg.keep_alive == "2h"


def test_pool_threads_keep_alive_into_client():
    pool = _pool()
    client = pool.get_client("fast-01")
    assert client.keep_alive == "2h"


def test_chat_passes_keep_alive_to_ollama(monkeypatch):
    """keep_alive is forwarded to the underlying ollama client.chat call."""
    client = BlockClient(host="http://x", block_name="b", keep_alive="2h")
    seen = {}

    def _fake_chat(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(client._client, "chat", _fake_chat)
    client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert seen["keep_alive"] == "2h"


def test_chat_omits_keep_alive_when_none(monkeypatch):
    client = BlockClient(host="http://x", block_name="b", keep_alive=None)
    seen = {}
    monkeypatch.setattr(client._client, "chat", lambda **kw: seen.update(kw) or object())
    client.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    assert "keep_alive" not in seen
