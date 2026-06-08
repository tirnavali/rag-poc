# MCP Server Architecture

## Overview

Three focused MCP servers, each owning a specific resource type and its retrieval algorithm.
The LLM decides which server to call; the router additionally applies a server-side
keyword fallback to promote `mode="deep"` when the query contains müfettiş triggers,
so small models that miss the enum still get the correct behaviour.

```
mcp-press   (port 8001) ── PressRetriever   ── press_clips.db  + press_clips_vectors
mcp-minutes (port 8002) ── MinutesRetriever ── parliament.db   + parliament_minutes_vectors
mcp-router  (port 8003) ── HybridRetriever  ── both collections (cross-domain queries)
```

## Servers

### mcp-press — Gazete Arşivi

| | |
|---|---|
| Port | 8001 |
| Swagger | http://localhost:8001/docs |
| MCP SSE | http://localhost:8001/sse |
| REST | POST http://localhost:8001/api/search |
| Source | `src/mcp/press_server.py` |
| Retriever | `src/retriever/press_retriever.py` |

**Tool: `search_press_archive`**

```json
{
  "query": "28 Şubat 1997 MGK toplantısı",
  "year": 1997,
  "author": "Can Dündar",
  "publication": "Sabah"
}
```

- `year`: filters by `tarih_year` in ChromaDB and `TARIH LIKE '1997%'` in FTS5
- `author`: SQL `YAZARLAR LIKE '%Can Dündar%'`
- `publication`: SQL `GAZETE_ADI LIKE '%Sabah%'`
- If `year` is omitted, extracted automatically from query text

---

### mcp-minutes — TBMM Tutanakları

| | |
|---|---|
| Port | 8002 |
| Swagger | http://localhost:8002/docs |
| MCP SSE | http://localhost:8002/sse |
| REST | POST http://localhost:8002/api/search |
| Source | `src/mcp/minutes_server.py` |
| Retriever | `src/retriever/minutes_retriever.py` |

**Tool: `search_parliament_minutes`**

```json
{
  "query": "deprem sonrası acil önlemler",
  "year": 2023,
  "party": "CHP",
  "speaker": "özgür özel"
}
```

- `year`: filters by `date_year` in ChromaDB and `date LIKE '2023%'` in FTS5
- `party`: ChromaDB `$eq` filter on `party` field (AND with year if both provided)
- `speaker`: ChromaDB `$eq` filter + prepended to FTS query for BM25 boost
- Coverage: 2002–2026 genel kurul oturumları

---

### mcp-router — Çapraz Kaynak Router

| | |
|---|---|
| Port | 8003 |
| Swagger | http://localhost:8003/docs |
| MCP SSE | http://localhost:8003/sse |
| REST | POST http://localhost:8003/api/search |
| Source | `src/mcp/router_server.py` |
| Retriever | `src/retriever/hybrid.py` (HybridRetriever) |

**Tool: `search_archives`**

```json
{
  "query": "1997 Refah Partisi hakkında gazete ve meclis kayıtları",
  "mode": "normal"
}
```

- Searches both collections simultaneously; RRF (Reciprocal Rank Fusion) merges results
- `mode`: `"normal"` (default, 10 sonuç / 12k bağlam) veya `"deep"` (40 sonuç / 25k bağlam + sorgu genişletme)
- **Server-side keyword fallback**: query'de `müfettiş | derin araştır | detaylı incele | kapsamlı ara` varsa `mode` otomatik `"deep"` olur. LLM enum'u kaçırsa bile doğru mod seçilir.
- `/api/search` REST endpoint'i aynı `mode` sözleşmesini ve fallback'ı kullanır; yanıtta gerçekten uygulanan `mode` alanı döner.

---

## When to Use Which Server

| Query type | Use |
|---|---|
| Gazete haberi, köşe yazısı, basın kupürü | `mcp-press` |
| Meclis konuşması, parti tutumu, milletvekili | `mcp-minutes` |
| Hem gazetede hem mecliste geçen konu | `mcp-router` (`mode: "normal"`) |
| Derinlemesine araştırma (müfettiş modu) | `mcp-router` + `mode: "deep"` (veya query'de "müfettiş/derin/detaylı/kapsamlı") |

---

## Running Locally

```bash
# Each in a separate terminal
python -m src.mcp.press_server    # → http://localhost:8001/docs
python -m src.mcp.minutes_server  # → http://localhost:8002/docs
python -m src.mcp.router_server   # → http://localhost:8003/docs
```

## Docker Compose

```bash
docker-compose up mcp-press mcp-minutes mcp-router
```

## Connecting Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "press-archive": {
      "url": "http://localhost:8001/sse"
    },
    "parliament-minutes": {
      "url": "http://localhost:8002/sse"
    },
    "archive-router": {
      "url": "http://localhost:8003/sse"
    }
  }
}
```

## Connecting Open WebUI

Tek-tool + enum + keyword fallback tasarımı küçük modellerin (Gemma-4) tool seçimini güvenilir hale getirir. Kurulum:

1. **Tool sunucusunu ekle**: Settings → Tools → Add Tool Server → `http://host.docker.internal:8003/sse` (docker-compose içinde open-webui konteynerı için `host.docker.internal`; local dev için `http://localhost:8003/sse`). Daha önce `search_all_archives` / `search_archive` ile bağlıysan eski bağlantıyı silip yeniden ekle — yeni tool adı `search_archives`.

2. **Workspace → Models → Gemma-4 (veya kullandığın model) → System Prompt** — aşağıyı yapıştır:

   ```
   Sen Türkçe bir gazete arşivi ve TBMM tutanakları araştırma asistanısın.

   Kullanıcının her mesajında aşağıdaki kuralları UYGULA:

   1) Her soruya önce `search_archives` aracını çağırarak başla; aracı çağırmadan yanıt üretme.
   2) Kullanıcı mesajında şu ifadelerden biri geçerse `mode="deep"` ile çağır:
      - "müfettiş", "müfettis"
      - "derin araştırma", "derin incele"
      - "detaylı incele", "kapsamlı ara"
      Aksi her durumda `mode="normal"`.
   3) Araç döndürdüğü bağlamdaki kaynaklara `[P-<id>]` (gazete) ve `[M-<id>]` (meclis) formatında atıf ver.
   4) Bağlamda yoksa "arşivde bulunamadı" de; uydurma.
   5) Derin modda daha uzun, karşılaştırmalı, zaman çizgili bir analiz üret.
   ```

3. Test: chat'te "1980 darbesini müfettiş gibi incele" yaz; router logunda `mode="deep"` uygulandığını doğrula (enum LLM tarafından geçmese bile keyword fallback yakalar).

> Not: Open-WebUI kendi system prompt'unu yönettiği için projedeki `MUFETTIS_SYS_PROMPT` devre dışı kalır; yukarıdaki Workspace prompt'u bunun UI-tarafı ikamesidir. Tam parite için Pipelines middleware gerekir (gelecek adım).

---

## Technical Notes

- All servers share `src/mcp/_base.py` for FastAPI + SSE factory code
- PressRetriever uses `tarih_year` metadata field (gazete ChromaDB)
- MinutesRetriever uses `date_year` metadata field (minutes ChromaDB)
- Exact dates in queries (e.g. "28 Şubat 1997") are converted to year-level filters
  — prevents filtering out articles published a few days before/after an event date
- Each server initializes its retriever lazily on first request
