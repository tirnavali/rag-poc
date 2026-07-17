"""Unit tests for the LLM client pool (offline — no Ollama connection).

BlockClient'ın ChatOllama cephesi raw-ollama sözleşmesini birebir korumalı:
num_ctx pinlemesi, think→reasoning eşlemesi, format geçişi, yanıt şimleri
(message.content/.thinking), stream ayrımı ve retry/health defteri.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import src.common.llm_client_pool as pool_mod
from src.common.llm_client_pool import BlockClient, LLMClientPool, _ChatResponseShim
from src.config.pipeline_loader import load_pipeline_config


# ------------------------------------------------------------- pool seviyesi

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


# ----------------------------------------------------------- ChatOllama sahtesi

class _FakeChatOllama:
    """ChatOllama'nın çağrı-kaydeden sahtesi."""

    instances: list["_FakeChatOllama"] = []

    def __init__(self, **kwargs):
        self.ctor_kwargs = kwargs
        self.invoke_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.invoke_results: list = []  # AIMessage veya Exception kuyruğu
        self.stream_chunks: list = [
            AIMessageChunk(content="", additional_kwargs={"reasoning_content": "düşünce "}),
            AIMessageChunk(content="merhaba"),
            AIMessageChunk(content=" dünya"),
        ]
        _FakeChatOllama.instances.append(self)

    def invoke(self, messages, config=None, **kwargs):
        self.invoke_calls.append({"messages": messages, "config": config, "kwargs": kwargs})
        if self.invoke_results:
            result = self.invoke_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return AIMessage(content="ok")

    def stream(self, messages, config=None, **kwargs):
        self.stream_calls.append({"messages": messages, "config": config, "kwargs": kwargs})
        yield from self.stream_chunks


@pytest.fixture
def fake_chat(monkeypatch):
    _FakeChatOllama.instances = []
    monkeypatch.setattr(pool_mod, "ChatOllama", _FakeChatOllama)
    # Cephe ağa yalnız health_check'te çıkar; raw istemciyi de nötrle.
    monkeypatch.setattr(pool_mod.ollama, "Client", MagicMock())
    yield _FakeChatOllama


def _client(**overrides) -> BlockClient:
    defaults = dict(
        host="http://localhost:11434",
        block_name="fast-01",
        timeout_seconds=60,
        retries=1,
        keep_alive="2h",
        default_num_ctx=32768,
    )
    defaults.update(overrides)
    return BlockClient(**defaults)


_MSGS = [{"role": "user", "content": "ping"}]


# --------------------------------------------------------------- num_ctx pin

def test_num_ctx_pinned_when_options_omit_it(fake_chat):
    _client().chat("m1", _MSGS, options={"num_predict": 1})
    kwargs = fake_chat.instances[0].invoke_calls[0]["kwargs"]
    assert kwargs["options"] == {"num_predict": 1, "num_ctx": 32768}


def test_explicit_num_ctx_wins_over_block_default(fake_chat):
    _client().chat("m1", _MSGS, options={"num_ctx": 4096})
    kwargs = fake_chat.instances[0].invoke_calls[0]["kwargs"]
    assert kwargs["options"]["num_ctx"] == 4096


def test_num_ctx_pinned_even_without_options(fake_chat):
    _client().chat("m1", _MSGS)
    kwargs = fake_chat.instances[0].invoke_calls[0]["kwargs"]
    assert kwargs["options"] == {"num_ctx": 32768}


def test_stream_calls_also_pin_num_ctx(fake_chat):
    list(_client().chat("m1", _MSGS, options={"temperature": 0.3}, stream=True))
    kwargs = fake_chat.instances[0].stream_calls[0]["kwargs"]
    assert kwargs["options"] == {"temperature": 0.3, "num_ctx": 32768}


# ------------------------------------------------- think / format eşlemeleri

def test_think_maps_to_reasoning_kwarg(fake_chat):
    client = _client()
    client.chat("m1", _MSGS, think=False)
    client.chat("m1", _MSGS, think=True)
    calls = fake_chat.instances[0].invoke_calls
    assert calls[0]["kwargs"]["reasoning"] is False
    assert calls[1]["kwargs"]["reasoning"] is True


def test_think_none_omits_reasoning(fake_chat):
    _client().chat("m1", _MSGS, think=None)
    kwargs = fake_chat.instances[0].invoke_calls[0]["kwargs"]
    assert "reasoning" not in kwargs


def test_format_json_passthrough(fake_chat):
    _client().chat("m1", _MSGS, format="json")
    kwargs = fake_chat.instances[0].invoke_calls[0]["kwargs"]
    assert kwargs["format"] == "json"


def test_config_passthrough_to_invoke(fake_chat):
    sentinel = {"callbacks": ["cb"]}
    _client().chat("m1", _MSGS, config=sentinel)
    assert fake_chat.instances[0].invoke_calls[0]["config"] is sentinel


# ------------------------------------------------------------- yanıt şimleri

def test_response_shim_exposes_content_and_thinking(fake_chat):
    client = _client()
    client._chat_model("m1").invoke_results.append(
        AIMessage(content="cevap", additional_kwargs={"reasoning_content": "akıl yürütme"})
    )
    res = client.chat("m1", _MSGS)
    assert res.message.content == "cevap"
    assert res.message.thinking == "akıl yürütme"


def test_response_shim_thinking_defaults_empty():
    res = _ChatResponseShim.from_ai_message(AIMessage(content="x"))
    assert res.message.thinking == ""


def test_stream_shim_separates_thinking_and_content(fake_chat):
    chunks = list(_client().chat("m1", _MSGS, stream=True))
    thinking = "".join(c.message.thinking for c in chunks if c.message.thinking)
    content = "".join(c.message.content for c in chunks if c.message.content)
    assert thinking == "düşünce "
    assert content == "merhaba dünya"
    # tools.py'nin hasattr(chunk.message, "thinking") kontrolü her chunk'ta tutmalı
    assert all(hasattr(c.message, "thinking") for c in chunks)


# ------------------------------------------------------------ retry + health

def test_retry_recovers_and_health_bookkeeping(fake_chat, monkeypatch):
    sleeps = []
    monkeypatch.setattr(pool_mod.time, "sleep", sleeps.append)
    client = _client(retries=2)
    fake = client._chat_model("m1")
    fake.invoke_results = [RuntimeError("boom"), AIMessage(content="ikinci")]
    res = client.chat("m1", _MSGS)
    assert res.message.content == "ikinci"
    assert sleeps == [0.5]  # 0.5 * attempt backoff'u korunur
    assert client.is_healthy is False  # ilk hata defteri düşürür (mevcut davranış)


def test_retry_exhausted_raises(fake_chat, monkeypatch):
    monkeypatch.setattr(pool_mod.time, "sleep", lambda _s: None)
    client = _client(retries=2)
    fake = client._chat_model("m1")
    fake.invoke_results = [RuntimeError("a"), RuntimeError("b")]
    with pytest.raises(RuntimeError, match="b"):
        client.chat("m1", _MSGS)
    assert client.is_healthy is False
    assert client._last_error == "b"


# --------------------------------------------------- kurulum ve model önbelleği

def test_chat_model_constructed_with_host_timeout_keepalive(fake_chat):
    _client().chat("m1", _MSGS)
    ctor = fake_chat.instances[0].ctor_kwargs
    assert ctor["model"] == "m1"
    assert ctor["base_url"] == "http://localhost:11434"
    assert ctor["client_kwargs"] == {"timeout": 60}
    assert ctor["keep_alive"] == "2h"


def test_keep_alive_none_omitted_from_ctor(fake_chat):
    _client(keep_alive=None).chat("m1", _MSGS)
    assert "keep_alive" not in fake_chat.instances[0].ctor_kwargs


def test_chat_model_cached_per_model_name(fake_chat):
    client = _client()
    client.chat("m1", _MSGS)
    client.chat("m1", _MSGS)
    client.chat("m2", _MSGS)
    assert len(fake_chat.instances) == 2
