import asyncio
import json
import uuid
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

from src.generator.service import RAGService
from src.common.tracer import PipelineTracer
from src.api import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAGService instance
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    logger.info("Initializing RAG Service...")
    # Initialize without specific pipeline for now, can be configured via env vars if needed
    rag_service = RAGService()
    yield
    logger.info("Shutting down RAG Service...")

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="TBMM RAG API", lifespan=lifespan)

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount React static assets if built
dist_assets = Path("src/ui/web_app/dist/assets")
if dist_assets.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_assets.parent / "assets")), name="assets")

@app.get("/", response_class=HTMLResponse)
def get_index():
    dist_index = Path("src/ui/web_app/dist/index.html")
    if dist_index.exists():
        return dist_index.read_text(encoding="utf-8")
        
    index_path = Path("src/ui/web/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return index_path.read_text(encoding="utf-8")

class SessionCreate(BaseModel):
    title: Optional[str] = "Yeni Sohbet"

@app.post("/api/sessions")
def create_session(data: SessionCreate):
    session_id = str(uuid.uuid4())
    db.create_session(session_id, data.title)
    return {"id": session_id, "title": data.title}

@app.get("/api/sessions")
def get_sessions():
    return db.get_sessions()

@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    return db.get_messages(session_id)

@app.websocket("/api/chat/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # First message should be the query config
        data = await websocket.receive_text()
        config = json.loads(data)
        
        query = config.get("query")
        session_id = config.get("session_id")
        mufettis_mode = config.get("mufettis_mode", False)
        agent_mode = True  # Agent Mode is the sole entry point
        
        if not query or not session_id:
            await websocket.send_json({"type": "error", "content": "Missing query or session_id"})
            await websocket.close()
            return

        # Load prior conversation history before saving the current message
        _MAX_HISTORY_TURNS = 5
        raw_history = db.get_messages(session_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in raw_history[-(_MAX_HISTORY_TURNS * 2):]
            if m["role"] in ("user", "assistant")
        ]

        # Send the memory context to the frontend for transparency
        await websocket.send_json({
            "type": "memory",
            "history": chat_history,
            "max_turns": _MAX_HISTORY_TURNS,
        })

        # Save user message
        db.add_message(session_id, "user", query)
        
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # Phase callback to translate events into WS status updates
        def on_phase(name: str, block, model, details: dict):
            phase_messages = {
                "classification": "🧭 Niyet analizi yapılıyor...",
                "probe": "🔎 Ön tarama yapılıyor...",
                "clarification": "❓ Sorgu daraltılıyor...",
                "planning": "🤖 Planlama ve arama kararı alınıyor...",
                "retrieval": "🔍 Arşiv taranıyor (Çeşitlendirilmiş arama)...",
                "re_retrieval": "🔄 Yeniden arama tetiklendi (Yetersiz kaynak)...",
                "expansion": "↻ Sorgu genişletiliyor...",
                "answering": "✍️ Yanıt üretiliyor...",
                "generation": "✍️ Yanıt üretiliyor...",
                "validation": "✅ Yanıt doğrulanıyor...",
                "filter_extraction": "🧭 Arama filtreleri analiz ediliyor...",
                "context_building": "📚 Bağlam oluşturuluyor..."
            }
            msg = phase_messages.get(name, f"İşlem yapılıyor: {name}...")
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "status", "content": msg}),
                loop
            )

        # Per-phase live trace: push each completed stage (with its filled-in
        # details — reasoning/thinking/answer_preview) as it finishes, so the UI
        # shows per-stage LLM output progressively instead of all at the end.
        def on_phase_end(event):
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "trace_phase",
                    "event": {
                        "phase": event.phase,
                        "status": "success",
                        "model": event.model,
                        "elapsed": event.latency_ms / 1000.0 if event.latency_ms else 0.0,
                        "details": event.details,
                    },
                }),
                loop,
            )

        # Initialize tracer for standard mode
        tracer = PipelineTracer(on_phase=on_phase)
        
        answer_text = ""
        thinking_text = ""
        trace_events = []
        sources = []
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "content": "Arşiv asistanı başlatılıyor..."
        })
        
        def blocking_producer():
            nonlocal answer_text, thinking_text, trace_events, sources
            try:
                if agent_mode:
                    from src.config.collections import get_production_collection_keys
                    production_keys = list(get_production_collection_keys())
                    
                    def stream_callback(token_or_chunk):
                        if isinstance(token_or_chunk, dict):
                            asyncio.run_coroutine_threadsafe(
                                queue.put(token_or_chunk),
                                loop
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                queue.put({"type": "content", "content": token_or_chunk}),
                                loop
                            )
                        
                    output = rag_service.run_agent(
                        query,
                        on_phase=on_phase,
                        session_collections=production_keys,
                        stream_callback=stream_callback,
                        clarification_callback=None, # Non-interactive clarification in web UI for now
                        deep_mode=mufettis_mode,
                        on_phase_end=on_phase_end,
                        chat_history=chat_history,
                    )
                    
                    answer_text = output.answer
                    thinking_text = output.thinking
                    sources = output.sources
                    
                    if output.thinking:
                        asyncio.run_coroutine_threadsafe(
                            queue.put({"type": "thinking", "content": output.thinking}),
                            loop
                        )
                        
                    # Extract trace events from agent output
                    trace_events = [
                        {
                            "phase": ev.phase,
                            "status": "success",
                            "model": ev.model,
                            "elapsed": ev.latency_ms / 1000.0 if ev.latency_ms else 0.0,
                            "details": ev.details
                        } for ev in output.trace
                    ]
                else:
                    # Standard mode
                    for chunk in rag_service.ask_stream(query, mufettis_mode=mufettis_mode, tracer=tracer):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                        
                    # Extract trace events from our tracer
                    trace_events = [
                        {
                            "phase": ev.phase,
                            "status": "success",
                            "model": ev.model,
                            "elapsed": ev.latency_ms / 1000.0 if ev.latency_ms else 0.0,
                            "details": ev.details
                        } for ev in tracer.events
                    ]
                
                asyncio.run_coroutine_threadsafe(queue.put(None), loop) # EOF marker
            except Exception as e:
                logger.error(f"Generation error: {e}")
                asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "content": str(e)}), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Run producer in background
        future = loop.run_in_executor(None, blocking_producer)
        
        # Consume queue and send over WS
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
                
            if isinstance(chunk, dict):
                chunk_dict = chunk
            else:
                chunk_dict = {"type": chunk["type"], "content": chunk["content"]}
                
            if chunk_dict["type"] == "thinking":
                thinking_text += chunk_dict["content"]
            elif chunk_dict["type"] == "content":
                answer_text += chunk_dict["content"]
                
            await websocket.send_json(chunk_dict)
            
        # Send trace at the end
        await websocket.send_json({
            "type": "trace",
            "events": trace_events
        })
        
        # Send sources at the end
        await websocket.send_json({
            "type": "sources",
            "sources": sources
        })
        
        # Save assistant message
        db.add_message(session_id, "assistant", answer_text, sources=sources, trace=trace_events, memory=chat_history)
        
        await websocket.close()
        
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
