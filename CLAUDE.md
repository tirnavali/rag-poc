# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Interactive chat (standard mode)
python chat.py

# Interactive chat (agentic planning mode)
python chat.py --agent --pipeline pipeline.yaml

# Ingest documents
python -m src.trainer.ingestion.ingest --request manifest.json

# MCP servers (run separately, each on its own port)
python -m src.mcp.press_server       # port 8001 — gazete arşivi
python -m src.mcp.router_server      # port 8003 — çapraz arama + rapor

# Tests
pytest tests/                        # all tests
pytest tests/test_vector_retriever.py  # single file
pytest -m "not integration and not slow"  # skip heavy tests

# Evaluation & benchmarking
python -m scripts.evaluate
python -m scripts.run_benchmark

# Collection inspection (Streamlit)
streamlit run scripts/vector_explorer.py

# Reindex everything from scratch
python -m scripts.reindex_all
```

## Architecture

The system has two independent entry paths that converge at the Retriever/Generator level:

**CLI path:** `chat.py` → `src/ui/chat.py` → `RAGService` → `VectorRetriever` + `OllamaGenerator`

**MCP path:** External client (Claude Desktop, Open WebUI) → `press_server.py:8001` or `router_server.py:8003` → `VectorRetriever` / `RAGService` / `DeepPipeline`

`src/mcp/server.py` is legacy/unused.

### Configuration as source of truth

All magic numbers live in **`src/config/settings.py`** (paths, thresholds, keyword lists, retrieval params). Embedding models and collection definitions live in **`models.yaml`** — no Python changes needed to add a collection. Document type display/filter specs live in **`src/config/document_types.py`**.

The `CollectionSpec` (loaded from `models.yaml`) is the key object passed through the stack: it controls which Chroma DB to use, which embedding model to run at query time (must match index time), whether late chunking is supported, and chunking parameters.

### Retrieval

`VectorRetriever` (`src/retriever/vector_retriever.py`) is the production retriever. It wraps `VectorSearch` (ANN via ChromaDB + optional cross-encoder reranking) and returns a `RetrievalResult` TypedDict (defined in `src/common/protocols.py`). Date filters are parsed automatically from the query via `src/common/dates.py`; an explicit `where_filter` dict bypasses this.

`src/retriever/multi_source.py` handles cross-collection routing: keyword detection (via `settings.MINUTES_KEYWORDS`, `PUBLICATION_KEYWORDS`) chooses which collection(s) to search, then RRF fusion merges results.

**Mufettis mode** doubles top_k/fetch_k, increases temperature and max_tokens, and runs `DeepPipeline` (expansion → re-retrieval → judgment loop). Triggered by keywords like `müfettiş`, `derin araştırma`, or via `router_server.py`'s `generate_report` tool.

### Metadata schema

Every chunk in Chroma carries a canonical metadata dict: `source_name`, `date`, `author`, `source_title`, `topics`. `author` is the abstract term across all document types (journalist for press, speaker for parliament, signatory for bills). Legacy Turkish keys are normalized at read-time in `document_types.py`.

Filter fields per type are declared in `DocumentTypeSpec.filter_fields`; ChromaDB filtering is exact-match only. Speaker/party name searches are more reliably done via semantic (vector) text matching + cross-encoder rerank than via metadata filter — the indexed parliament `author` is the full titled label (e.g. "BAŞBAKAN RECEP TAYYİP ERDOĞAN"), so an exact-match filter for a person name zeroes out. (BM25 was dropped; retrieval is ANN + rerank only.)

### Ingestion

`IngestionPipeline` (`src/trainer/ingestion/pipeline.py`) is the orchestrator. It takes a `CollectionSpec` and a `DocumentInput`, runs a type-specific adapter (in `adapters/`), parses PDF/DOCX via `DoclingManager`, optionally embeds with `LocalLateChunkingEmbedder` (Jina v3/v4 only — disabled for Nomic v2 because 512-token context makes late chunking pointless), then upserts to Chroma. A SQLite-backed `DocumentManifest` tracks content hashes for idempotent re-ingestion.

Chunk IDs follow the format `{document_id}_{chunk_index}` — deterministic and dedup-safe.

### Agent / Orchestrator layer

`OrchestratorAgent` (`src/agent/orchestrator.py`) is an explicit state machine used in `--agent` mode: Planner → Policy → Allocator → Retrieve → Assembler → Judge → (Expand → loop) → Answer → Sanitizer → Citations. This path is separate from the simpler `RAGService` used in standard mode.

### MCP factory pattern

`src/mcp/_base.py` exports `create_app(mcp_server, title)` which returns a FastAPI app with SSE transport (`/sse`, `/messages`) plus a REST fallback (`/api/search`). All three active MCP servers use this factory and define their tools via `@mcp_server.list_tools()` / `@mcp_server.call_tool()`.

Response format is always: prose context + `--- KAYNAKLAR (JSON) ---` footer with structured source metadata — this lets MCP clients cite reliably.

## Key extension points

| What | Where |
|---|---|
| New embedding model | `models.yaml` → `model_specs` |
| New collection | `models.yaml` → `collections` + `models.yaml` → `defaults` |
| New document type | `src/config/document_types.py` + new adapter in `src/trainer/ingestion/adapters/` |
| New MCP tool | `src/mcp/router_server.py` — add to `@list_tools()` and `@call_tool()` |
| Keyword routing | `src/config/settings.py` — `MINUTES_KEYWORDS`, `PUBLICATION_KEYWORDS`, `ONERGE_KEYWORDS` |

## Environment

Copy `.env.example` to `.env`. Key vars not in `settings.py`:

```
OCR_ENGINE=easyocr          # easyocr | tesseract | mac
DOCLING_USE_GPU=auto        # auto | true | false
USE_LOCAL_LATE_CHUNKING=0   # 1 = Jina local model (CPU-heavy)
```

`RAG_ENV`, `RAG_LLM_MODEL`, `RAG_EMBED_MODEL`, `RETRIEVAL_MODE`, `USE_RERANKER` are read in `settings.py` with fallbacks. The active Ollama host/model can also be overridden at runtime via `pipeline.yaml`.

## Test markers

```
pytest -m integration   # tests that hit Ollama or ChromaDB
pytest -m slow          # benchmarks and full pipeline runs
```

Most unit tests mock the LLM and ChromaDB — safe to run offline.
