"""Centralized LLM client pool for multi-block deployment.

Manages chat clients per deployment block with retry, timeout, and
health-check support. The transport is langchain-ollama's ``ChatOllama``
(a LangChain Runnable — callbacks passed via ``config`` or the ambient
LangGraph config context reach every call), but the public surface is the
raw-ollama-era ``chat()`` contract: components keep reading
``res.message.content`` / ``chunk.message.thinking``.
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Optional

import ollama
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from src.config.pipeline_loader import PipelineConfig


class _MessageShim:
    """Duck-types the raw ollama response message (``.content``/``.thinking``)."""

    __slots__ = ("content", "thinking")

    def __init__(self, content: str = "", thinking: str = "") -> None:
        self.content = content
        self.thinking = thinking


class _ChatResponseShim:
    """Duck-types ``ollama.ChatResponse`` just enough for the agent components."""

    __slots__ = ("message",)

    def __init__(self, message: _MessageShim) -> None:
        self.message = message

    @classmethod
    def from_ai_message(cls, msg: BaseMessage) -> "_ChatResponseShim":
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        thinking = msg.additional_kwargs.get("reasoning_content") or ""
        return cls(_MessageShim(content=content, thinking=thinking))


class BlockClient:
    """Chat client wrapper for a single deployment block with retry/timeout."""

    def __init__(
        self,
        host: str,
        block_name: str,
        timeout_seconds: int = 30,
        retries: int = 1,
        keep_alive: "str | int | None" = None,
        default_num_ctx: "int | None" = None,
    ) -> None:
        self.host = host
        self.block_name = block_name
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        # Every call must pin num_ctx explicitly: without it Ollama falls back to
        # the model's Modelfile context (e.g. 256K), which both forces a reload
        # (mismatch with the already-resident instance) and can OOM the runner —
        # a 31B model at 256K × parallel slots needs far more KV cache than fits.
        self.default_num_ctx = default_num_ctx
        # How long Ollama keeps the model resident after a call. None → Ollama
        # default (5m). A long value (e.g. "2h" or -1) avoids re-loading the model
        # from disk on the first query after an idle gap (the ~11s cold-start).
        self.keep_alive = keep_alive
        # Raw client kept only for health_check(); chat traffic goes through
        # ChatOllama so LangChain callbacks (Langfuse) see every generation.
        self._client = ollama.Client(host=host, timeout=timeout_seconds)
        self._chat_models: dict[str, ChatOllama] = {}
        self._healthy = True
        self._last_error: str | None = None

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def _chat_model(self, model: str) -> ChatOllama:
        if model not in self._chat_models:
            kwargs: dict[str, Any] = {
                "model": model,
                "base_url": self.host,
                # Forwarded to httpx so a stuck/loading model raises instead of
                # hanging forever.
                "client_kwargs": {"timeout": self.timeout_seconds},
            }
            if self.keep_alive is not None:
                kwargs["keep_alive"] = self.keep_alive
            self._chat_models[model] = ChatOllama(**kwargs)
        return self._chat_models[model]

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        format: str | None = None,
        stream: bool = False,
        think: bool | None = None,
        config: Optional[RunnableConfig] = None,
    ) -> Any:
        """Call chat with retry logic.

        ``config`` is an optional LangChain RunnableConfig (callbacks etc.);
        when omitted, calls made inside a LangGraph node still inherit the
        graph's config from the ambient context.

        Raw-ollama'dan bilinen iki davranış farkı:
        - ``think=None`` (reasoning gönderilmez) iken langchain-ollama,
          modelin kendiliğinden ürettiği thinking'i İSTEMCİ tarafında düşürür
          (``reasoning`` truthy olmadan reasoning_content doldurulmaz).
          Thinking isteyen çağrı ``think=True``, istemeyen ``think=False``
          geçmeli — None'a güvenme.
        - Non-stream ``invoke`` içeride stream'leyip biriktirir: httpx read
          timeout artık toplam üretimi değil, ilk-token/chunk-arası boşlukları
          sınırlar (takılı model yine yakalanır; yavaş damlayan üretim
          num_predict ile sınırlıdır).
        """
        merged_options = dict(options or {})
        if self.default_num_ctx is not None:
            merged_options.setdefault("num_ctx", self.default_num_ctx)

        # Invoke-time kwargs: `options` fully replaces ChatOllama's constructor
        # defaults and `reasoning` maps to Ollama's `think` (None → omitted),
        # so this is byte-equivalent to the raw ollama.Client call it replaced.
        call_kwargs: dict[str, Any] = {}
        if merged_options:
            call_kwargs["options"] = merged_options
        if format:
            call_kwargs["format"] = format
        if think is not None:
            call_kwargs["reasoning"] = think

        chat_model = self._chat_model(model)
        if stream:
            return self._stream(chat_model, messages, call_kwargs, config)

        for attempt in range(1, self.retries + 1):
            try:
                ai_msg = chat_model.invoke(messages, config=config, **call_kwargs)
                return _ChatResponseShim.from_ai_message(ai_msg)
            except Exception as e:
                self._healthy = False
                self._last_error = str(e)
                if attempt < self.retries:
                    time.sleep(0.5 * attempt)
                    continue
                raise

    def _stream(
        self,
        chat_model: ChatOllama,
        messages: list[dict],
        call_kwargs: dict[str, Any],
        config: Optional[RunnableConfig],
    ) -> Iterator[_ChatResponseShim]:
        """Yield shimmed chunks; thinking deltas arrive via reasoning_content."""
        try:
            for chunk in chat_model.stream(messages, config=config, **call_kwargs):
                yield _ChatResponseShim.from_ai_message(chunk)
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            raise

    def health_check(self) -> bool:
        """Quick health check by listing models."""
        try:
            self._client.list()
            self._healthy = True
            self._last_error = None
            return True
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            return False

    def __repr__(self) -> str:
        status = "healthy" if self._healthy else f"unhealthy({self._last_error})"
        return f"BlockClient({self.block_name}, {self.host}, {status})"


class LLMClientPool:
    """Pool of chat clients, one per deployment block.

    Clients are lazily initialized on first access.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._clients: dict[str, BlockClient] = {}

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "LLMClientPool":
        return cls(config)

    def get_client(self, block_name: str) -> BlockClient:
        """Get or create a client for the given block."""
        if block_name not in self._clients:
            block = self._config.get_block(block_name)
            self._clients[block_name] = BlockClient(
                host=block.host,
                block_name=block_name,
                timeout_seconds=block.timeout_seconds,
                retries=block.retries,
                keep_alive=getattr(self._config, "keep_alive", None),
                default_num_ctx=block.max_num_ctx,
            )
        return self._clients[block_name]

    def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all initialized clients."""
        return {
            name: client.health_check()
            for name, client in self._clients.items()
        }

    def get_model_for_block(self, block_name: str, model_key: str) -> str:
        """Get the model name for a specific role in a block."""
        block = self._config.get_block(block_name)
        model = block.get_model(model_key)
        if not model:
            raise ValueError(
                f"Model key '{model_key}' not found in block '{block_name}'. "
                f"Available: {list(block.models.keys())}"
            )
        return model

    def get_host(self, block_name: str) -> str:
        """Get the host URL for a block (for embedding clients, etc.)."""
        return self._config.get_block(block_name).host
