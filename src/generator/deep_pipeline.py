"""Shared deep-research pipeline used by both the CLI and the MCP report tool.

The CLI /müfettiş and /rapor commands consume ``run()`` (an iterator of
StreamChunks); the MCP ``generate_report`` tool consumes ``run_blocking()``
(returns the accumulated ReportResult). Keeping a single implementation
prevents the two surfaces from drifting on prompt, k, or context budget.

Pipeline:
    1. expand_query  → query + expansion (single LLM call)
    2. retrieve(mufettis_mode=True) using the expanded query (single fetch)
    3. build_context with MUFETTIS_CONTEXT_* budgets
    4. stream(MUFETTIS_SYS_PROMPT) — yields StreamChunks
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

from src.common.protocols import RetrievalResult, StreamChunk
from src.config import settings
from src.retriever.context import build_context, build_structured_context


@dataclass
class ReportResult:
    markdown: str
    sources: list[dict]
    expanded_query: Optional[str]
    timings_ms: dict[str, float]
    truncated: bool = False


class DeepPipeline:
    def __init__(self, service) -> None:
        self.service = service
        self._last: Optional[ReportResult] = None

    def run(
        self,
        query: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> Iterable[StreamChunk]:
        results, expanded, t_expand_ms, t_retrieve_ms = self._retrieve(query)
        ctx_args = {
            "max_chars": settings.MUFETTIS_CONTEXT_MAX_CHARS,
            "total_max_chars": settings.MUFETTIS_CONTEXT_TOTAL_MAX,
        }
        context = build_context(results, **ctx_args)
        sources = build_structured_context(results, **ctx_args)

        if not context.strip():
            msg = "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."
            yield StreamChunk(type="content", content=msg)
            self._last = ReportResult(
                markdown=msg,
                sources=[],
                expanded_query=expanded,
                timings_ms={"expand": t_expand_ms, "retrieve": t_retrieve_ms, "generate": 0.0},
            )
            return

        gen_start = time.perf_counter()
        markdown_buf: list[str] = []
        for chunk in self.service.generator.stream(
            query, context, mufettis_mode=True, num_predict=max_tokens
        ):
            if chunk["type"] == "content":
                markdown_buf.append(chunk["content"])
            yield chunk
        t_generate_ms = (time.perf_counter() - gen_start) * 1000

        self._last = ReportResult(
            markdown="".join(markdown_buf).strip(),
            sources=sources,
            expanded_query=expanded,
            timings_ms={
                "expand": t_expand_ms,
                "retrieve": t_retrieve_ms,
                "generate": t_generate_ms,
            },
        )

    def run_blocking(
        self,
        query: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> ReportResult:
        for _ in self.run(query, max_tokens=max_tokens):
            pass
        if self._last is None:
            raise RuntimeError("DeepPipeline.run() did not complete — _last not set")
        return self._last

    def retrieve_only(self, query: str) -> ReportResult:
        """Run only expand+retrieve (no generation) and return sources.

        Used as the graceful-degrade fallback when the generation budget is
        exceeded — Open WebUI's own LLM can synthesize from the structured
        sources instead of getting nothing.
        """
        results, expanded, t_expand_ms, t_retrieve_ms = self._retrieve(query)
        ctx_args = {
            "max_chars": settings.MUFETTIS_CONTEXT_MAX_CHARS,
            "total_max_chars": settings.MUFETTIS_CONTEXT_TOTAL_MAX,
        }
        sources = build_structured_context(results, **ctx_args)
        return ReportResult(
            markdown="",
            sources=sources,
            expanded_query=expanded,
            timings_ms={"expand": t_expand_ms, "retrieve": t_retrieve_ms, "generate": 0.0},
            truncated=True,
        )

    def _retrieve(
        self, query: str
    ) -> tuple[RetrievalResult, Optional[str], float, float]:
        t0 = time.perf_counter()
        expanded = self.service.generator.expand_query(query)
        t_expand_ms = (time.perf_counter() - t0) * 1000

        combined = (
            f"{query} {expanded}".strip()
            if expanded and expanded != query
            else query
        )
        t1 = time.perf_counter()
        result = self.service.retriever.retrieve(
            combined,
            top_k=settings.MUFETTIS_TOP_K,
            fetch_k=settings.MUFETTIS_FETCH_K,
            mufettis_mode=True,
        )
        t_retrieve_ms = (time.perf_counter() - t1) * 1000

        expanded_for_meta = expanded if expanded and expanded != query else None
        result["expanded_query"] = expanded_for_meta
        return result, expanded_for_meta, t_expand_ms, t_retrieve_ms
