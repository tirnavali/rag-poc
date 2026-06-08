# src/generator

RAG pipeline'ının üretim (generation) katmanı. LLM çağrıları, filter extraction, deep-research pipeline ve sistem prompt'larını barındırır. Dış dünyaya tek giriş noktası `RAGService`'tir.

## Modüller

| Dosya | Açıklama |
|---|---|
| `service.py` | `RAGService` — retriever + generator'ı birleştiren ana façade |
| `ollama_generator.py` | `OllamaGenerator` — Ollama LLM wrapper; streaming, query expansion |
| `filter_extractor.py` | `FilterExtractor` — doğal dil sorgusundan ChromaDB filter çıkarır |
| `deep_pipeline.py` | `DeepPipeline` — `/müfettiş` ve MCP report için paylaşılan deep-research pipeline |
| `prompts.py` | Tüm sistem prompt sabitleri (`SYS_PROMPT`, `MUFETTIS_SYS_PROMPT`, `FILTER_SYSTEM_PROMPT`, vb.) |

## Temel Sınıflar

### `RAGService` (`service.py`)

Sistemin tek dış arayüzü. `chat.py`, `app.py` ve MCP araçları buradan başlar.

| Metot | Açıklama |
|---|---|
| `retrieve(query, ...)` | Filter extraction → fallback cascade → vector retrieval |
| `ask_stream(query, ...)` | Streaming yanıt; `mufettis_mode=True` ise `DeepPipeline`'a yönlendirir |
| `ask(query, debug)` | Blocking yanıt; debug modunda chunk skorlarını yazdırır |
| `ask_from_results(query, results)` | Dışarıdan verilmiş result seti üzerinden yanıt üretir |
| `build_context(results, ...)` | `RetrievalResult`'ı LLM context string'ine dönüştürür |
| `inspect_record(source_db, chunk_id)` | `/kaynak N` komutu için kaynak kaydını döner |
| `run_agent(query, on_phase)` | Planning Agent pipeline'ını çalıştırır |

**Retrieval fallback cascade sırası:**
1. Tam filter (yıl + yazar + kaynak + …)
2. `author_dropped` — yazar/rol filter'ı düşürülmüş
3. `semantic_only` — filter yok, saf vektör araması

Cascade mantığı bilerek `RAGService`'de tutulur; `VectorRetriever` filter-agnostik kalır (katman ihlalini önler).

### `OllamaGenerator` (`ollama_generator.py`)

Ollama API wrapper'ı. Token budget ve temperature, mod bazında ayarlanır (`settings.py`).

| Metot | Açıklama |
|---|---|
| `stream(query, context, ...)` | `StreamChunk` iterator'ı döner; thinking + content chunk'larını ayırır |
| `answer(query, context, ...)` | Stream'i toplar; `(thinking, content)` tuple döner |
| `expand_query(query)` | Müfettiş modu için sorguyu genişletir; hata halinde orijinali döner |

### `FilterExtractor` (`filter_extractor.py`)

Kullanıcının doğal dil sorgusundan yıl, yazar, kaynak, dönem, birleşim gibi metadata bilgilerini çıkarır. `has_filter_hints()` ile LLM çağrısını bypass edebilir (hız optimizasyonu).

| Metot | Açıklama |
|---|---|
| `has_filter_hints(query)` | LLM'e gitmeden önce hızlı regex + keyword kontrolü |
| `extract(query)` | `ExtractedFilterResponse` (refined_query + filters) döner |
| `to_chroma_filter(filters)` | `FilterCriteria` → ChromaDB `where` dict |
| `fallback_chain(criteria)` | Sıralı relaxation adayları listesi döner |

### `DeepPipeline` (`deep_pipeline.py`)

`/müfettiş`, `/rapor` (CLI) ve `generate_report` (MCP) tarafından paylaşılan implementasyon. İki tüketim modu vardır:

| Metot | Açıklama |
|---|---|
| `run(query, ...)` | Streaming — `StreamChunk` iterator'ı döner (CLI için) |
| `run_blocking(query, ...)` | Blocking — `ReportResult` döner (MCP için) |
| `retrieve_only(query)` | Generation olmadan expand + retrieve; bütçe aşımında graceful fallback |

**Pipeline adımları:**
```
expand_query()
    ↓
retrieve(mufettis_mode=True)  [MUFETTIS_TOP_K sonuç]
    ↓
build_context()  [MUFETTIS_CONTEXT_* bütçeleri]
    ↓
generator.stream(MUFETTIS_SYS_PROMPT)
    ↓
ReportResult { markdown, sources, timings_ms }
```

## Sorgu Akışı

```
ask_stream(query)
    │
    ├─ mufettis_mode=False ──► retrieve() → build_context() → generator.stream()
    │                              │
    │                              └─ filter extraction → fallback cascade → VectorRetriever
    │
    └─ mufettis_mode=True ───► DeepPipeline.run()
                                   │
                                   └─ expand_query → retrieve → build_context → stream
```

## Kullanım

```python
from src.generator.service import RAGService

service = RAGService()

# Streaming yanıt
for chunk in service.ask_stream("1996 yılında Ege krizinde kim ne dedi?"):
    if chunk["type"] == "content":
        print(chunk["content"], end="", flush=True)

# Müfettiş modu (deep research)
for chunk in service.ask_stream("Kardak krizi", mufettis_mode=True):
    if chunk["type"] == "content":
        print(chunk["content"], end="", flush=True)

# Blocking
thinking, answer = service.ask("Deniz Baykal 1996")
```
