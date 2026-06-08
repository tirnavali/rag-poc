"""MCP server for the 1997 newspaper press archive (gazete arşivi).

Port 8001 — Swagger UI at http://localhost:8001/docs
MCP SSE at  http://localhost:8001/sse
REST API at http://localhost:8001/api/search  (POST, JSON)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server import Server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from pydantic import BaseModel, Field

from src.mcp._base import create_app, format_response, run_server
from src.retriever.context import build_context, build_structured_context
from src.retriever.vector_retriever import VectorRetriever
from src.config import settings
from src.config.collections import get_spec

# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

mcp = Server("rag-press-archive")
_retriever: Optional[VectorRetriever] = None


def _get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever(get_spec("press_nomic"))
    return _retriever


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_press_archive",
            description=(
                "1997 yılı gazete arşivinde vektör arama yapar. Sabah gazetesinden haber ve "
                "köşe yazılarını döndürür. Gazete haberleri, basın kupürleri, yazar "
                "yazıları veya belirli bir tarihte yayımlanmış içerik için kullanın. "
                "TBMM tutanakları için search_parliament_minutes aracını kullanın."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Arama sorgusu (Türkçe)",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@mcp.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[TextContent | ImageContent | EmbeddedResource]:
    if name != "search_press_archive":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query", "")
    if not query:
        raise ValueError("query is required")

    results = _get_retriever().retrieve(query)
    ctx_kwargs = {
        "max_chars": settings.CONTEXT_MAX_CHARS,
        "total_max_chars": settings.CONTEXT_TOTAL_MAX,
    }
    ctx = build_context(results, **ctx_kwargs)
    sources = build_structured_context(results, **ctx_kwargs)
    text = format_response(
        ctx, sources, "Gazete arşivinde bu sorguya uygun kayıt bulunamadı."
    )
    return [TextContent(type="text", text=text)]


# ---------------------------------------------------------------------------
# FastAPI REST interface (Swagger-documented)
# ---------------------------------------------------------------------------

class PressSearchRequest(BaseModel):
    query: str = Field(..., description="Arama sorgusu (Türkçe)")


class SearchResponse(BaseModel):
    context: str = Field(..., description="Derlenmiş kaynak metni")
    sources: list[dict] = Field(default_factory=list, description="Yapılandırılmış kaynak listesi (JSON)")
    result_count: int = Field(..., description="Döndürülen belge sayısı")


app: FastAPI = create_app(
    mcp,
    title="Gazete Arşivi MCP",
    description="1997 yılı Türkiye basın arşivi — vektör arama.",
)


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def api_search(req: PressSearchRequest) -> SearchResponse:
    """Gazete arşivinde arama yap ve ilgili bağlamı döndür."""
    results = _get_retriever().retrieve(req.query)
    ctx_kwargs = {
        "max_chars": settings.CONTEXT_MAX_CHARS,
        "total_max_chars": settings.CONTEXT_TOTAL_MAX,
    }
    ctx = build_context(results, **ctx_kwargs)
    sources = build_structured_context(results, **ctx_kwargs)
    docs = results.get("documents", [[]])[0]
    return SearchResponse(context=ctx, sources=sources, result_count=len(docs))


if __name__ == "__main__":
    run_server(app, default_port=8001)
