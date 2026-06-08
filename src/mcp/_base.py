"""Shared FastAPI + MCP SSE factory used by all three MCP servers."""
from __future__ import annotations

import argparse
import json

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount, Route


KAYNAKLAR_HEADER = "--- KAYNAKLAR (JSON) ---"


def format_response(prose: str, sources: list[dict], empty_message: str) -> str:
    """Return MCP-friendly text: prose context + JSON provenance footer.

    A single string keeps wire compatibility — Open WebUI and other clients
    that expect one TextContent per tool call still see exactly one. The
    JSON block lets a downstream LLM cite reliably without parsing prose.
    """
    if not prose.strip():
        return empty_message
    payload = json.dumps(sources, ensure_ascii=False, indent=2)
    return f"{prose}\n\n{KAYNAKLAR_HEADER}\n{payload}"


def create_app(mcp_server: Server, title: str, description: str = "") -> FastAPI:
    """Return a FastAPI app with MCP SSE transport + /api/search REST endpoint mounted.

    Each MCP server calls this factory and then adds its own /api/search route
    on top (FastAPI lets you add routes after creation).
    """
    sse = SseServerTransport("/messages")

    async def handle_sse(scope, receive, send):
        if scope.get("method") == "POST":
            import json
            body = b""
            more_body = True
            while more_body:
                msg = await receive()
                body += msg.get("body", b"")
                more_body = msg.get("more_body", False)
            scope["query_string"] = b"session_id=default_session"

            async def mock_receive():
                return {"type": "http.request", "body": body, "more_body": False}

            await sse.handle_post_message(scope, mock_receive, send)
            return

        async with sse.connect_sse(scope, receive, send) as streams:
            await mcp_server.run(
                streams[0], streams[1], mcp_server.create_initialization_options()
            )

    async def handle_messages(scope, receive, send):
        await sse.handle_post_message(scope, receive, send)

    app = FastAPI(
        title=title,
        description=description,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/config")
    async def get_config():
        return {"VALVES": {}}
    # Mount raw ASGI SSE routes; they bypass FastAPI routing intentionally
    app.router.routes.insert(0, Mount("/sse", app=handle_sse))
    app.router.routes.insert(1, Mount("/messages", app=handle_messages))

    # Expose the shared SSE transport so servers can reference it if needed
    app.state.sse = sse
    app.state.mcp = mcp_server

    return app


def run_server(app: FastAPI, default_port: int) -> None:
    parser = argparse.ArgumentParser(description=app.title)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    print(f"{app.title} → http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
