"""MCP server for TBMM parliamentary minutes (meclis tutanakları).

Port 8002 — Swagger UI at http://localhost:8002/docs
MCP SSE at  http://localhost:8002/sse
REST API at http://localhost:8002/api/search  (POST, JSON)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from mcp.server import Server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool
from pydantic import BaseModel, Field

from src.mcp._base import create_app, format_response, run_server
from src.retriever.context import build_context, build_structured_context
from src.retriever.minutes_retriever import MinutesRetriever
from src.config import settings

# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

mcp = Server("rag-parliament-minutes")
_retriever: Optional[MinutesRetriever] = None


def _get_retriever() -> MinutesRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MinutesRetriever()
    return _retriever


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_parliament_minutes",
            description=(
                "TBMM (Türkiye Büyük Millet Meclisi) genel kurul tutanaklarında arama yapar. "
                "2002–2026 arası milletvekili konuşmalarını, parti açıklamalarını ve yasama "
                "tartışmalarını içerir. Meclis konuşmaları, milletvekili sözleri, parti tutumu "
                "veya yasama süreci için kullanın. Gazete haberleri için search_press_archive "
                "aracını kullanın."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Arama sorgusu (Türkçe)",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Oturum yılı filtresi (örn. 2023, 2026)",
                    },
                    "party": {
                        "type": "string",
                        "description": "Parti adı filtresi (örn. 'CHP', 'AKP', 'DEM Parti', 'MHP', 'İYİ Parti')",
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Konuşmacı milletvekili adı filtresi",
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
    if name != "search_parliament_minutes":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query", "")
    if not query:
        raise ValueError("query is required")

    results = _get_retriever().retrieve(
        query,
        year=arguments.get("year"),
        party=arguments.get("party"),
        speaker=arguments.get("speaker"),
    )
    ctx_kwargs = {
        "max_chars": settings.CONTEXT_MAX_CHARS,
        "total_max_chars": settings.CONTEXT_TOTAL_MAX,
    }
    ctx = build_context(results, **ctx_kwargs)
    sources = build_structured_context(results, **ctx_kwargs)
    text = format_response(
        ctx, sources, "TBMM tutanaklarında bu sorguya uygun kayıt bulunamadı."
    )
    return [TextContent(type="text", text=text)]


# ---------------------------------------------------------------------------
# FastAPI REST interface (Swagger-documented)
# ---------------------------------------------------------------------------

class MinutesSearchRequest(BaseModel):
    query: str = Field(..., description="Arama sorgusu (Türkçe)")
    year: Optional[int] = Field(None, description="Oturum yılı filtresi (örn. 2026)")
    party: Optional[str] = Field(None, description="Parti adı (örn. 'CHP', 'AKP', 'DEM Parti')")
    speaker: Optional[str] = Field(None, description="Konuşmacı milletvekili adı")


class SearchResponse(BaseModel):
    context: str = Field(..., description="Derlenmiş kaynak metni")
    sources: list[dict] = Field(default_factory=list, description="Yapılandırılmış kaynak listesi (JSON)")
    result_count: int = Field(..., description="Döndürülen belge sayısı")


app: FastAPI = create_app(
    mcp,
    title="TBMM Tutanakları MCP",
    description="TBMM genel kurul tutanakları 2002–2026 — hibrit BM25 + vektör arama.",
)


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def api_search(req: MinutesSearchRequest) -> SearchResponse:
    """TBMM tutanaklarında arama yap ve ilgili bağlamı döndür."""
    results = _get_retriever().retrieve(
        req.query,
        year=req.year,
        party=req.party,
        speaker=req.speaker,
    )
    ctx_kwargs = {
        "max_chars": settings.CONTEXT_MAX_CHARS,
        "total_max_chars": settings.CONTEXT_TOTAL_MAX,
    }
    ctx = build_context(results, **ctx_kwargs)
    sources = build_structured_context(results, **ctx_kwargs)
    docs = results.get("documents", [[]])[0]
    return SearchResponse(context=ctx, sources=sources, result_count=len(docs))


if __name__ == "__main__":
    run_server(app, default_port=8002)
