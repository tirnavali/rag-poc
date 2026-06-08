from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.sse import SseServerTransport
from src.generator.service import RAGService
from src.config import settings
import starlette.applications
import starlette.routing
import uvicorn
import argparse

# MCP Sunucusunu Başlat
app = Server("rag-poc-mcp")

# Mevcut RAG servisimizi entegre ediyoruz
rag_service = RAGService()

@app.list_tools()
async def list_tools() -> list[Tool]:
    """MCP istemcisine (Open-WebUI, Claude vb.) sunulan yetenekleri tanımlar."""
    return [
        Tool(
            name="search_archive",
            description="Gazete arşivi ve TBMM tutanaklarında arama yapar. En alakalı belgelerden derlenmiş ham metni döndürür.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Aranacak kelime veya soru (Türkçe)."
                    },
                    "mufettis_mode": {
                        "type": "boolean",
                        "description": "Müfettiş modunu aktif eder. Daha derin arama yapar ve daha çok sonuç getirir.",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Bir araç çağrıldığında çalışacak kod."""
    if name == "search_archive":
        query = arguments.get("query")
        if not query:
            raise ValueError("Query parametresi zorunludur.")
            
        mufettis_mode = arguments.get("mufettis_mode", False)
        
        # Arama yap
        results = rag_service.retrieve(query, mufettis_mode=mufettis_mode)
        
        # Bağlamı (Context) oluştur
        ctx_args = {
            "max_chars": settings.MUFETTIS_CONTEXT_MAX_CHARS if mufettis_mode else settings.CONTEXT_MAX_CHARS,
            "total_max_chars": settings.MUFETTIS_CONTEXT_TOTAL_MAX if mufettis_mode else settings.CONTEXT_TOTAL_MAX,
        }
        
        context_str = rag_service.build_context(results, **ctx_args)
        
        if not context_str.strip():
            return [TextContent(type="text", text="Arşivde bu sorguya uygun kayıt bulunamadı.")]
            
        return [TextContent(type="text", text=context_str)]
    
    raise ValueError(f"Bilinmeyen araç: {name}")

def main():
    parser = argparse.ArgumentParser(description="RAG-poc MCP Server (SSE)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    args = parser.parse_args()

    # SSE Transport (Server-Sent Events) HTTP üzerinden MCP için
    sse = SseServerTransport("/messages")

    async def handle_sse(scope, receive, send):
        if scope["method"] == "POST":
            # Session_id var mı kontrol et
            from urllib.parse import parse_qs
            query_string = scope.get("query_string", b"").decode()
            params = parse_qs(query_string)
            
            if "session_id" not in params:
                # Session_id yoksa direkt işle (Open WebUI için bypass)
                import json
                
                # Request body'yi oku
                body = b""
                more_body = True
                while more_body:
                    message = await receive()
                    body += message.get("body", b"")
                    more_body = message.get("more_body", False)
                
                request_json = json.loads(body)
                
                # MCP app'i manuel olarak çağır
                # Not: Bu kısım basitleştirilmiştir, tam uygulama için app.handle_request gerekebilir
                # Ama en kolayı kütüphaneye session_id varmış gibi davranmak
                scope["query_string"] = b"session_id=default_session"
                
                # Yeniden gönderilebilir receive oluştur
                async def mock_receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                
                await sse.handle_post_message(scope, mock_receive, send)
                return
                
            await sse.handle_post_message(scope, receive, send)
            return
        
        async with sse.connect_sse(scope, receive, send) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )

    async def handle_messages(scope, receive, send):
        await sse.handle_post_message(scope, receive, send)
        
    import starlette.requests
    import starlette.responses

    async def handle_api_search(request: starlette.requests.Request):
        try:
            data = await request.json()
            query = data.get("query")
            if not query:
                return starlette.responses.JSONResponse({"error": "query is required"}, status_code=400)
            
            mufettis_mode = data.get("mufettis_mode", False)
            results = rag_service.retrieve(query, mufettis_mode=mufettis_mode)
            
            ctx_args = {
                "max_chars": settings.MUFETTIS_CONTEXT_MAX_CHARS if mufettis_mode else settings.CONTEXT_MAX_CHARS,
                "total_max_chars": settings.MUFETTIS_CONTEXT_TOTAL_MAX if mufettis_mode else settings.CONTEXT_TOTAL_MAX,
            }
            context_str = rag_service.build_context(results, **ctx_args)
            
            return starlette.responses.JSONResponse({"context": context_str})
        except Exception as e:
            return starlette.responses.JSONResponse({"error": str(e)}, status_code=500)
        
    # Starlette ile HTTP sunucusunu kur
    starlette_app = starlette.applications.Starlette(
        debug=True,
        routes=[
            starlette.routing.Mount("/sse", app=handle_sse),
            starlette.routing.Mount("/messages", app=handle_messages),
            starlette.routing.Route("/api/search", endpoint=handle_api_search, methods=["POST"]),
        ],
    )
    
    print(f"MCP SSE Sunucusu başlatılıyor: http://{args.host}:{args.port}/sse")
    uvicorn.run(starlette_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
