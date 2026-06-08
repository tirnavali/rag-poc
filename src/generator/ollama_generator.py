"""Ollama-backed generator: streaming, query expansion, and non-streaming ask."""
from __future__ import annotations

import re
from typing import Iterable

import ollama

from src.common.protocols import StreamChunk
from src.config import settings
from src.generator.prompts import (
    EXPAND_QUERY_PROMPT,
    MUFETTIS_SYS_PROMPT,
    SYS_PROMPT,
)


class OllamaGenerator:
    def __init__(self, model: str = settings.LLM_MODEL) -> None:
        self.model = model
        self.client = ollama.Client(host=settings.OLLAMA_HOST)

    def expand_query(self, query: str) -> str:
        """Expand a query for müfettiş deep-research mode via LLM."""
        prompt = EXPAND_QUERY_PROMPT.format(query=query)
        try:
            res = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": settings.LLM_TEMPERATURE_EXPAND},
            )
            expanded = res.message.content.strip()
            expanded = re.sub(r'["\']', "", expanded)
            return expanded if expanded else query
        except Exception as e:
            print(f"Query expansion error: {e}")
            return query

    def stream(
        self,
        query: str,
        context: str,
        *,
        mufettis_mode: bool = False,
        num_predict: int | None = None,
    ) -> Iterable[StreamChunk]:
        """Yield StreamChunk dicts from an Ollama streaming chat call.

        ``num_predict`` overrides the default token cap; useful when callers
        (e.g., the MCP report tool) need to fit inside a wall-clock budget.
        """
        user_msg = f"BAĞLAM:\n{context}\n\nSORU: {query}"
        sys_prompt = MUFETTIS_SYS_PROMPT if mufettis_mode else SYS_PROMPT
        token_cap = num_predict if num_predict is not None else (
            settings.LLM_NUM_PREDICT_MUFETTIS if mufettis_mode else settings.LLM_NUM_PREDICT_DEFAULT
        )
        stream = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            options={
                "temperature": settings.LLM_TEMPERATURE_MUFETTIS if mufettis_mode else settings.LLM_TEMPERATURE_DEFAULT,
                "num_predict": token_cap,
                "num_ctx": settings.LLM_NUM_CTX,
            },
            stream=True,
        )
        for chunk in stream:
            if hasattr(chunk.message, "thinking") and chunk.message.thinking:
                yield StreamChunk(type="thinking", content=chunk.message.thinking)
            if hasattr(chunk.message, "content") and chunk.message.content:
                yield StreamChunk(type="content", content=chunk.message.content)

    def answer(
        self,
        query: str,
        context: str,
        *,
        mufettis_mode: bool = False,
    ) -> tuple[str, str]:
        """Return (thinking, content) by collecting the full stream."""
        thinking = ""
        content = ""
        for chunk in self.stream(query, context, mufettis_mode=mufettis_mode):
            if chunk["type"] == "thinking":
                thinking += chunk["content"]
            else:
                content += chunk["content"]
        if not content.strip():
            content = "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."
        return thinking, content.strip()
