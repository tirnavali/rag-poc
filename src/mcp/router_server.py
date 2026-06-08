"""Cross-domain MCP router — searches both press and parliament collections.

Use this when a query spans both sources (e.g., "1997 Refah Partisi hakkında
gazete ve meclis kayıtları"). For single-source queries prefer the focused servers.

Port 8003 — Swagger UI at http://localhost:8003/docs
MCP SSE at  http://localhost:8003/sse
REST API at http://localhost:8003/api/search  (POST, JSON)
"""
from __future__ import annotations

import asyncio
import re
from typing import Literal, Optional

from fastapi import FastAPI
from mcp.server import Server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from pydantic import BaseModel, Field

from src.generator.deep_pipeline import DeepPipeline, ReportResult
from src.generator.service import RAGService
from src.mcp._base import create_app, format_response, run_server
from src.retriever.context import build_context, build_structured_context
from src.retriever.vector_retriever import VectorRetriever
from src.config import settings


# Wall-clock budget for the full deep pipeline server-side. Open WebUI's own
# tool-call timeout dominates beyond ~60s, so keep a hard ceiling.
REPORT_BUDGET_SECONDS = 45.0
# Token cap for generate_report — tighter than the CLI MUFETTIS cap so the
# generation can finish inside the budget on local gemma4.
REPORT_MAX_TOKENS = 6000

# ---------------------------------------------------------------------------
# Mode resolution — belt-and-braces deep-mode detection
# ---------------------------------------------------------------------------

# Belt-and-braces: küçük modeller (gemma4) enum'u kaçırır ya da search_archives'ı
# generate_report yerine seçerse, sunucu query metnini tarayarak deep moduna terfi
# eder. Terminaldeki /müfettiş ve /rapor davranışıyla aynı kelime kümesi.
_DEEP_KEYWORDS = re.compile(
    r"müfetti[sş]|"
    r"derin\s+ara[sş]t[ıi]r|"
    r"detayl[ıi]\s+incele|"
    r"kapsaml[ıi]\s+(ara|incele)|"
    r"\brapor\w*",
    re.IGNORECASE,
)

Mode = Literal["normal", "deep"]


def _resolve_mode(query: str, requested: Optional[str]) -> bool:
    """Return True if deep (müfettiş) mode should be active."""
    mode = requested if requested in {"normal", "deep"} else "normal"
    if _DEEP_KEYWORDS.search(query or ""):
        mode = "deep"
    return mode == "deep"


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

mcp = Server("rag-router")
_retriever: Optional[VectorRetriever] = None
_service: Optional[RAGService] = None


def _get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        from src.config.collections import get_spec
        _retriever = VectorRetriever(get_spec(settings.DEFAULT_COLLECTION))
    return _retriever


def _get_pipeline() -> DeepPipeline:
    """Lazy-init a shared RAGService + DeepPipeline (loads Ollama clients)."""
    global _service
    if _service is None:
        _service = RAGService()
    return DeepPipeline(_service)


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_archives",
            description=(
                "Basit arama / lookup: gazete arşivi (Sabah 1997) + TBMM tutanakları "
                "(2002-2026) üzerinde RRF füzyonlu birleşik arama. Yapılandırılmış "
                "kaynak listesi döndürür; LLM'in sentezlemesi beklenir. "
                "Kullanıcı 'rapor', 'müfettiş', 'derin araştırma', 'kapsamlı inceleme' "
                "isterse generate_report aracını çağır — bu araç sadece arama yapar. "
                "Modlar: 'normal' (10 sonuç) ve 'deep' (40 sonuç + sorgu genişletme)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Arama sorgusu (Türkçe)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "deep"],
                        "default": "normal",
                        "description": (
                            "'deep' derin araştırma modudur (daha fazla sonuç, "
                            "daha geniş bağlam, sorgu genişletme). Varsayılan: 'normal'."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="generate_report",
            description=(
                "Derin arşiv araştırmasına dayalı, kaynaklı yapılandırılmış Markdown "
                "raporu üretir. Kullanıcı 'rapor hazırla', 'müfettiş gibi araştır', "
                "'derin araştırma', 'kapsamlı inceleme' istediğinde kullan. "
                "Sunucu tarafında sorgu genişletme + 40 kaynak + müfettiş prompt "
                "ile tam pipeline çalışır ve hazır rapor döner (~30-45 sn). "
                "Sadece gerçek arşiv verisine dayanır, uydurma yapmaz. "
                "Basit lookup için search_archives kullan."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Rapor için ana soru veya konu (Türkçe)",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Opsiyonel: raporun odaklanması istenen alt konu",
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Opsiyonel: 'YYYY-YYYY' veya 'YYYY' formatında tarih aralığı ipucu",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


def _compose_report_query(query: str, focus: Optional[str], date_range: Optional[str]) -> str:
    """Fold focus + date_range hints into the query string for retrieval."""
    parts = [query]
    if focus:
        parts.append(f"odak: {focus}")
    if date_range:
        parts.append(f"tarih aralığı: {date_range}")
    return "  |  ".join(parts)


def _format_report_text(query: str, result: ReportResult) -> str:
    """Build the final TextContent payload for the generate_report tool."""
    if result.truncated and not result.markdown.strip():
        body = (
            f"# Derin Araştırma Sonuçları: {query}\n\n"
            "_rapor üretimi zaman aşımı — ham kaynaklar döndürüldü. "
            "Aşağıdaki KAYNAKLAR üzerinden lütfen sentez yapın._"
        )
    elif result.truncated:
        body = (
            f"# Derin Araştırma Raporu: {query}\n\n"
            f"{result.markdown}\n\n"
            "_[üretim kesildi — token bütçesi aşıldı, kaynaklar tam listelendi]_"
        )
    elif result.markdown.strip():
        body = result.markdown
    else:
        body = "Arşivde bu soruyu yanıtlayacak yeterli bilgi bulunamadı."

    return format_response(body, result.sources, body)


async def _run_report(
    pipeline: DeepPipeline, query: str
) -> ReportResult:
    """Run the deep pipeline with a wall-clock budget; degrade on timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(pipeline.run_blocking, query, max_tokens=REPORT_MAX_TOKENS),
            timeout=REPORT_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        # Generation exceeded the budget. Return retrieval-only sources so the
        # caller (Open WebUI's LLM) can still synthesize a best-effort answer.
        return await asyncio.to_thread(pipeline.retrieve_only, query)


@mcp.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[TextContent | ImageContent | EmbeddedResource]:
    query = arguments.get("query", "")
    if not query:
        raise ValueError("query is required")

    if name == "search_archives":
        mufettis_mode = _resolve_mode(query, arguments.get("mode"))
        results = _get_retriever().retrieve(query, mufettis_mode=mufettis_mode)

        ctx_args = {
            "max_chars": settings.MUFETTIS_CONTEXT_MAX_CHARS if mufettis_mode else settings.CONTEXT_MAX_CHARS,
            "total_max_chars": settings.MUFETTIS_CONTEXT_TOTAL_MAX if mufettis_mode else settings.CONTEXT_TOTAL_MAX,
        }
        ctx = build_context(results, **ctx_args)
        sources = build_structured_context(results, **ctx_args)
        text = format_response(ctx, sources, "Arşivde bu sorguya uygun kayıt bulunamadı.")
        return [TextContent(type="text", text=text)]

    if name == "generate_report":
        composed = _compose_report_query(
            query, arguments.get("focus"), arguments.get("date_range")
        )
        result = await _run_report(_get_pipeline(), composed)
        return [TextContent(type="text", text=_format_report_text(query, result))]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# FastAPI REST interface (Swagger-documented)
# ---------------------------------------------------------------------------

class RouterSearchRequest(BaseModel):
    query: str = Field(..., description="Arama sorgusu (Türkçe)")
    mode: Mode = Field(
        "normal",
        description="'deep' derin araştırma modu; varsayılan 'normal'.",
    )


class SearchResponse(BaseModel):
    context: str = Field(..., description="Derlenmiş kaynak metni")
    sources: list[dict] = Field(default_factory=list, description="Yapılandırılmış kaynak listesi (JSON)")
    result_count: int = Field(..., description="Döndürülen belge sayısı")
    mode: Mode = Field(..., description="Gerçekte uygulanan mod (keyword fallback sonrası)")


app: FastAPI = create_app(
    mcp,
    title="Arşiv Router MCP",
    description="Çapraz kaynak arama: gazete + TBMM tutanakları, RRF füzyonu.",
)


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def api_search(req: RouterSearchRequest) -> SearchResponse:
    """Her iki arşivde arama yap ve ilgili bağlamı döndür."""
    mufettis_mode = _resolve_mode(req.query, req.mode)
    results = _get_retriever().retrieve(req.query, mufettis_mode=mufettis_mode)
    ctx_args = {
        "max_chars": settings.MUFETTIS_CONTEXT_MAX_CHARS if mufettis_mode else settings.CONTEXT_MAX_CHARS,
        "total_max_chars": settings.MUFETTIS_CONTEXT_TOTAL_MAX if mufettis_mode else settings.CONTEXT_TOTAL_MAX,
    }
    ctx = build_context(results, **ctx_args)
    sources = build_structured_context(results, **ctx_args)
    docs = results.get("documents", [[]])[0]
    return SearchResponse(
        context=ctx,
        sources=sources,
        result_count=len(docs),
        mode="deep" if mufettis_mode else "normal",
    )


class ReportRequest(BaseModel):
    query: str = Field(..., description="Rapor için ana soru veya konu (Türkçe)")
    focus: Optional[str] = Field(None, description="Opsiyonel: odak alt-konu")
    date_range: Optional[str] = Field(None, description="Opsiyonel: tarih aralığı ipucu")


class ReportResponse(BaseModel):
    markdown: str = Field(..., description="Üretilen Markdown raporu")
    sources: list[dict] = Field(default_factory=list, description="Kaynak listesi (JSON)")
    expanded_query: Optional[str] = Field(None, description="LLM tarafından genişletilen sorgu")
    timings_ms: dict[str, float] = Field(default_factory=dict, description="Aşama süreleri (ms)")
    truncated: bool = Field(False, description="Bütçe aşıldı mı?")


@app.post("/api/report", response_model=ReportResponse, tags=["Report"])
async def api_report(req: ReportRequest) -> ReportResponse:
    """Sunucu tarafında derin araştırma raporu üret. (~30-45 sn)"""
    composed = _compose_report_query(req.query, req.focus, req.date_range)
    result = await _run_report(_get_pipeline(), composed)
    return ReportResponse(
        markdown=result.markdown,
        sources=result.sources,
        expanded_query=result.expanded_query,
        timings_ms=result.timings_ms,
        truncated=result.truncated,
    )


if __name__ == "__main__":
    run_server(app, default_port=8003)
