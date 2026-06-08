# 30M-Page Scaling Plan

This document is the post-POC roadmap for taking the current 2-source RAG (press
clips + TBMM minutes, embedded ChromaDB, ~70k chunks) to **30M+ pages** across
many resource types (newspapers, journals, books, legislative records,
committee minutes). It is intentionally a *plan*, not an implementation — the
2-day MVP holds the line on demo quality and explicitly defers everything below.

The numbers in this document are back-of-envelope sizing, meant to bound the
problem rather than commit to specific infrastructure.

---

## 1. Vector store: Chroma → Qdrant (or Milvus)

**Why Chroma embedded breaks at scale**

The current setup uses `chromadb.PersistentClient()` against a local directory
(`data_lake/press_clips_vectors/`, `parliament_digital_born_minutes_vectors/`).
This loads the entire HNSW graph into RAM at process start, holds a single
SQLite writer lock, and has no built-in sharding or replication. At 30M chunks
this fails on three independent dimensions:

| Dimension | 70k chunks today | 30M chunks projection |
|---|---|---|
| Vector RAM (768-dim float32) | ~210 MB | **~90 GB** (single-host limit) |
| Quantized int8 RAM | ~50 MB | ~25 GB (still single-host) |
| Build time on Ollama CPU embeddings | minutes | **~17 days single-stream** |
| Concurrent ingest writers | 1 | 0 — bottleneck |

**Recommendation: Qdrant**, server mode, Docker-deployed.

- Hybrid search (BM25 + vector) is first-class via payload filters; we already
  rely on metadata filters for year/source/party/speaker, so Qdrant's payload
  indexing maps cleanly onto our existing `_vector_search` filter dicts in
  `src/retriever/{press,minutes}_retriever.py`.
- **Sharding key: `(source_db, year)`** — most parliamentary queries carry an
  implicit year filter ("2019 bütçe konuşmaları"), so this partitioning keeps
  the candidate set small without query rewriting.
- **Scalar int8 quantization** with rescoring on top-K — ~3.5x memory
  compression for negligible quality loss in our ada-002-class embedding.
- Keep SQLite (with `kupurler_fts` and `parliament_minutes_fts`) for BM25;
  cross-shard fanout for FTS at scale will eventually want OpenSearch /
  Vespa, but FTS5 holds well into the 5M-row range and is acceptable for
  Phase 1 of the migration.

**Milvus** is the alternative if cluster scale is needed beyond ~100M vectors;
it has higher operational complexity (etcd, MinIO, four services minimum)
and is overkill for the 30M target.

**Migration approach.** Dual-write during cutover (existing trainer writes to
both Chroma and Qdrant for one indexing pass), validate parity on the eval
harness golden set, then flip `HybridRetriever`/`PressRetriever`/`MinutesRetriever`
to query Qdrant. The retrievers' RRF fusion logic is independent of the vector
store, so the change is contained to `_vector_search()` in each class.

---

## 2. Embedding pipeline: distributed, resumable

**Single-stream throughput today.** `OllamaEmbeddings.embed_documents` runs
batch=20 against one Ollama host. At ~50ms per embedding on CPU with
`nomic-embed-text-v2-moe`, 30M chunks ≈ **17 days** continuous wall-clock.

**Target: ~24 hours wall-clock on 10 GPU-backed Ollama replicas.**

Architecture:

```
SQLite/JSON sources
        │
        ▼
   Ray Data pipeline (or Dask-bag if Ray adds too much weight)
        │  ├─ shard rows by (source_db, year)
        │  └─ progress checkpoint to artifacts/embed_state.json
        ▼
   load-balanced Ollama replicas (N=10) on GPU nodes
        │
        ▼
   Qdrant gRPC upsert (batched, idempotent on chunk_id)
```

**Idempotency** — chunk IDs are already `f"{sqlite_row_id}_{chunk_index}"`
(see `src/trainer/press_clips/index.py`, `src/trainer/minutes/index.py`).
Upsert is safe to repeat; resume = "rerun the pipeline." We piggyback on this
rather than building a separate state machine.

**Streaming reads from SQLite.** The press trainer was just fixed to use
`cursor.fetchmany(BATCH)` (commit on this branch). The minutes trainer still
buffers all chunks into Python lists before embedding (see TODO comment in
`src/trainer/minutes/index.py`); rewrite to a per-file streaming pass before
the migration.

**Cost model.** A 7B nomic-embed equivalent on a single A10G runs at ~5–8ms
per embedding. 30M / 8ms / 10 replicas ≈ ~10 hours; double for sequencing,
upsert, and tail. Budget 24h per full reindex, plan for partial reindexes
to be minutes (only newly-ingested chunks).

---

## 3. Reranker as quality gate

The current pipeline: BM25 + vector → RRF fusion → top-10. **Add a third
stage**: BGE cross-encoder rerank between fusion and final selection.

```
BM25 (top-50) ──┐
                ├── RRF fusion (top-50) ──► BGE-reranker-v2-m3 ──► top-10 ──► LLM
vector (top-50) ┘
```

**Why BGE-reranker-v2-m3:**
- Multilingual including Turkish (the eval fixture is heavy Turkish)
- ~568M params, fits on a single A10G with batch=16
- Cross-encoder scoring on (query, document) pair captures semantic
  similarity that bi-encoder embeddings miss — particularly important for
  parliamentary queries where exact phrasing matters

**Deploy as a separate HTTP micro-service**, not inline in the retriever.
This gives us:
- Independent horizontal scaling (rerank is the most expensive stage)
- Hot-swap to a future model without touching retriever code
- Graceful fallback: if rerank service is down, skip the stage and serve
  RRF top-10 with a header flag

**Latency budget.** ~200ms p50 for batch=50 → top-10. Within the 500ms p95
retrieval target if the rerank service is co-located.

**Quality bet.** Expect P@5 to lift 15-25% on long-tail and underspecified
queries. Validate empirically against the eval fixture before committing.

---

## 4. Metadata-first routing

A large chunk of MP queries carry strong filters: a year, a party name, a
speaker, a date range. **Use those filters to prune the candidate set before
vector search**, not after.

Today's flow (in `_vector_search`):
```
Chroma.query(query_embeddings=[v], where={year_field: 2019})
```
This works at 70k chunks because Chroma scans the in-memory index. At 30M
chunks, the where filter happens *during* HNSW traversal — on a poorly-indexed
field, the search degrades to scanning a large fraction of the graph.

**Move the filter to the front:**

1. Maintain a small SQL/Qdrant payload index over `(source_db, year, party,
   speaker)`.
2. For filtered queries, pre-resolve the candidate `kayit_no` set, then run
   vector search restricted to that set via a Qdrant `must_id` filter or by
   targeting only the relevant Qdrant shard.
3. Unfiltered queries fall through to today's path.

Most relevant for queries like "2019 AK Parti bütçe konuşmaları" — today this
runs full-corpus vector search and post-filters; with 30M chunks the candidate
set after metadata pre-filter is more like 50k, a 600x reduction.

---

## 5. Observability

The current eval harness (`src/evaluator/harness.py`) tracks retrieval and
generation latency offline, but production needs **per-query online telemetry**.

Per-query structured log emitted by `RAGService` and the MCP servers:

```json
{
  "ts": "...",
  "query_id": "...",
  "user_id_hash": "...",
  "surface": "cli|webui|mcp_router",
  "tool": "search_archives|generate_report",
  "routing_decision": ["gazete", "minutes"],
  "expansion_used": true,
  "bm25_count": 50,
  "vector_count": 50,
  "fusion_top_k": 10,
  "rerank_top_k": 10,
  "context_chars": 12000,
  "gen_tokens": 4096,
  "latency_breakdown_ms": {"expand": 1200, "retrieve": 280, "rerank": 180, "generate": 4500},
  "model": "gemma4:latest",
  "judge_score": null,
  "user_thumbs": null
}
```

**Sink:** ClickHouse for queryable analytics, or plain Parquet with a
DuckDB/Datasette UI in the lighter tier. Either is cheap to operate.

**Dashboards (Grafana or Datasette):**
- p50 / p95 latency per stage (expand, retrieve, rerank, generate)
- Recall@10 on a rotating golden subset (shadow eval — re-run nightly)
- Cost per query (`tokens × model_rate` on remote LLM, GPU-hours on local)
- Routing accuracy on questions where the user clicks `/kaynak N`
  (implicit feedback)

**Alerting:**
- p95 retrieval > 1s
- recall@10 on shadow eval drops > 5% week-on-week
- generation timeout rate > 5% on `generate_report`

---

## 6. MVP → Prod cut-line

What the 2-day POC explicitly defers:

| Capability | POC state | Production minimum |
|---|---|---|
| Vector store | Embedded Chroma, single host | Qdrant cluster (3 nodes), int8 quantized, sharded by `(source_db, year)` |
| Indexing | Sync, single-stream, OOM at 30M | Ray pipeline, 10 Ollama replicas, resumable |
| Reranker | None | BGE-reranker-v2-m3 microservice |
| BM25 | SQLite FTS5, single file | OpenSearch / Vespa cluster (Phase 2 only — FTS5 fine for Phase 1) |
| Metadata routing | Post-filter | Pre-filter via Qdrant payload index |
| Observability | Offline eval only | Per-query structured logs, dashboards, shadow eval |
| Auth | None | Per-MP identity, audit log of every report generated |
| Per-user state | In-memory chat history | Redis-backed sessions, retention policy |
| Tenancy | Single tenant | Multi-MP isolation, rate limiting per user |
| Trainer abstraction | Copy-pasted retrievers | `BaseSourceRetriever` + `SourceSpec` (deferred from MVP) |
| MCP report streaming | Synchronous, 45s budget | True SSE channel for clients that consume it |

---

## 7. Phasing

Suggested rollout order, post-demo:

**Week 1–2: foundations.** Qdrant migration + dual-write + parity validation.
Press trainer fix landed in the 2-day sprint; minutes trainer streaming
rewrite. `BaseSourceRetriever` refactor to consolidate the 3 near-duplicate
retrievers ahead of adding new sources.

**Week 3–4: quality.** BGE-reranker-v2-m3 deployment + A/B against current
top-K. Shadow eval pipeline running nightly. Per-query observability live.

**Week 5–6: scale.** Distributed embedding pipeline, ingest a journal corpus
or two as a smoke test of the 30M-page pipeline. Metadata-first routing.

**Week 7+: new sources.** With abstractions and pipeline in place, books and
journals onboard via a `SourceSpec` definition, schema, and ingest pass —
estimated 1-2 days per new source after the foundation is laid.

---

## 8. Why this order

The 2-day demo proves the **information loop is correct**: hybrid retrieval
fuses sources sensibly, the deep pipeline produces report-quality output for
MPs, the MCP surface plays well with Open WebUI. Those are the *hardest* bets
to validate; once they work, scaling is a series of mechanical migrations
with measurable acceptance criteria (parity tests, latency dashboards, eval
deltas).

Conversely, doing a Qdrant migration first would be expensive, risk
introducing regressions before the product proposition is validated, and does
not change the answer the user sees. Scale earns priority only after the
quality story is locked in.
