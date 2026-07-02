"""Centralized LLM client pool for multi-block deployment.

Manages Ollama clients per deployment block with retry, timeout,
and health-check support. Replaces direct ollama.Client(host=...) calls.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import ollama

from src.config.pipeline_loader import PipelineConfig


class BlockClient:
    """Ollama client wrapper for a single deployment block with retry/timeout."""

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
        # Pass timeout through to httpx so a stuck/loading model raises instead of
        # hanging forever (ollama.Client forwards **kwargs to its httpx client).
        self._client = ollama.Client(host=host, timeout=timeout_seconds)
        self._healthy = True
        self._last_error: str | None = None

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def chat(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        format: str | None = None,
        stream: bool = False,
        think: bool | None = None,
    ) -> Any:
        """Call chat with retry logic."""
        for attempt in range(1, self.retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                }
                merged_options = dict(options or {})
                if self.default_num_ctx is not None:
                    merged_options.setdefault("num_ctx", self.default_num_ctx)
                if merged_options:
                    kwargs["options"] = merged_options
                if format:
                    kwargs["format"] = format
                if think is not None:
                    kwargs["think"] = think
                if self.keep_alive is not None:
                    kwargs["keep_alive"] = self.keep_alive

                return self._client.chat(**kwargs)
            except Exception as e:
                self._healthy = False
                self._last_error = str(e)
                if attempt < self.retries:
                    time.sleep(0.5 * attempt)
                    continue
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
    """Pool of Ollama clients, one per deployment block.

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
