# `src/common` — RAG Altyapı Katmanı

Bu paket, Türk gazete arşivi ve TBMM tutanak korpusu üzerinde çalışan RAG pipeline'ının paylaşılan altyapısını içerir. Her modül tek bir sorumluluğa sahip olacak şekilde tasarlanmıştır ve birbirinden bağımsız import edilebilir. Üst katmanlar (`retriever`, `generator`, `agent`) bu modüllere bağımlıdır; alt yönde bağımlılık yoktur.

---

## Tip Sistemi

### `schemas.py`
Pipeline genelinde kullanılan Pydantic veri şemaları.

| Sınıf | Açıklama |
|---|---|
| `Message` | LLM mesaj formatı (role, content, tool_call_id) |
| `SearchResult` | Vektör DB hit: document, metadata, distance |
| `FilterCriteria` | Sorgudan çıkarılan filtreler: yıl, yazar, kaynak, dönem, belge türü |
| `ExtractedFilterResponse` | LLM filtre çıkarım yanıtı: refined_query + filters |
| `RoutingDecision` | Sorgu yönlendirme kararı: intents + reasoning |

### `protocols.py`
`typing.Protocol` ve `TypedDict` tanımlamaları. Retriever, Generator ve Evaluator bileşenlerinin uygulaması gereken arayüzleri belirtir; somut sınıflara bağımlılık olmadan tip güvencesi sağlar.

| İsim | Tür | Açıklama |
|---|---|---|
| `RetrievalResult` | TypedDict | Retriever çıktısı (documents, metadatas, distances, is_minutes, ...) |
| `StreamChunk` | TypedDict | LLM stream olayı: `{"type": "thinking"\|"content", "content": str}` |
| `Retriever` | Protocol | `retrieve()` ve `inspect_record()` arayüzü |
| `Generator` | Protocol | `expand_query()`, `answer()`, `stream()` arayüzü |

---

## Erişim Katmanı

### `sqlite_io.py`
SQLite bağlantı yardımcısı. `connect(db_path)` salt okunur bağlantı açar, `check_same_thread=False` ile thread güvenliği sağlar. Şu an yalnızca `src/retriever/minutes_retriever.py` tarafından kullanılır.

### `chroma.py`
ChromaDB koleksiyon yardımcıları. Tüm `chromadb.*` importları bu dosyada toplanmıştır; başka bir vektör veritabanına geçildiğinde yalnızca bu dosya değiştirilir.

| Fonksiyon | Açıklama |
|---|---|
| `open_collection(path, name)` | Var olan koleksiyonu açar |
| `open_or_create_collection(path, name)` | Yoksa cosine similarity ile oluşturur |
| `query_collection(...)` | `where` filtreli vektör sorgusu |
| `where_year_filter(years, field)` | Yıl bazlı ChromaDB `where` dict'i üretir |

### `filter_translators.py`
Adapter deseni. `FilterCriteria` (schemas.py) nesnelerini ChromaDB'nin `where` filtre sözlüğüne çevirir. `BaseFilterTranslator` soyut sınıfı üzerinden genişletilebilir (Qdrant, pgvector vb. için).

---

## Embedding ve Chunking

### `embeddings.py`
Embedding factory. İki backend destekler:
- **OllamaEmbeddings + L2 normalizasyonu** (`nomic-embed-text-v2-moe`) — cosine similarity için birim norm vektörler üretir
- **LocalLateChunkingEmbedder** (Jina v3) — late chunking için tembel import kullanır, döngüsel bağımlılığı önler

`build_embedder(model)` ve `build_embedder_for_spec(spec)` factory fonksiyonları kullanılır.

### `chunking.py`
LangChain `RecursiveCharacterTextSplitter` sarmalayıcısı.

| Fonksiyon | Açıklama |
|---|---|
| `build_text_splitter(chunk_size, chunk_overlap)` | Splitter nesnesi döndürür |
| `split_with_offsets(text, chunk_size, chunk_overlap)` | Her chunk için `(text, (start, end))` tuple listesi döndürür — late chunking için zorunlu |

### `span_resolver.py`
`chunk_id → (start_char, end_char)` çözümleyici. Chunk ID formatı `{doc_id}_{N}`. Parse önbelleğini (JSON) okuyarak chunk'ın tam metin içindeki karakter aralığını bulur. `src/evaluator/benchmark.py` ve `scripts/vector_explorer.py` tarafından kullanılır.

---

## LLM Yardımcıları

### `llm_utils.py`
LLM yanıtlarından JSON çıkarma ve doğrulama.

| Fonksiyon | Açıklama |
|---|---|
| `extract_json_from_text(text)` | Markdown bloğu (`\`\`\`json`) veya ham JSON çıkarır |
| `parse_llm_response(response, schema)` | JSON çıkarır, Pydantic şemasına karşı doğrular; geçersizse `ValueError` veya `ValidationError` fırlatır |

### `llm_client_pool.py`
Çoklu Ollama istemci havuzu. Birden fazla inference bloğuna (farklı host/port) yönetilmiş erişim sağlar.

| Sınıf | Açıklama |
|---|---|
| `BlockClient` | Tek blok için Ollama istemci sarmalayıcısı; retry, timeout, health-check |
| `LLMClientPool` | Tüm blokları yönetir; `get_client(block_name)` ile istemci döndürür |

Pipeline'da doğrudan `ollama.Client(host=...)` çağrısı yerine bu pool kullanılır.

---

## Metin İşleme

### `text.py`
Türkçe metin yardımcıları.

| Fonksiyon | Açıklama |
|---|---|
| `normalize_tr(text)` | Türkçe büyük/küçük harf dönüşümü (İ/I, ı/i), noktalama temizleme |
| `extract_relevant_windows(text, query, window_size, max_total)` | Sorgu terimlerini içeren bağlam pencerelerini birleştirir; eşleşme yoksa metnin başını döndürür |

`extract_relevant_windows` context bloat'u önlemek için kullanılır — tam belge yerine ilgili kesimler LLM'e gönderilir.

### `dates.py`
Türkçe tarih ifadelerini çıkarır ve normalize eder.

| Fonksiyon | Açıklama |
|---|---|
| `extract_dates(query)` | `{"exact_dates": [...], "years": [...]}` döndürür |
| `normalize_iso_date(raw)` | `YYYY-MM-DD` formatına çevirir (ISO, noktalı, eğik çizgili, Türkçe metin) |
| `extract_year(iso_date)` | ISO tarihten yıl int'i çıkarır; geçersizse 0 döndürür |

BM25 ve ChromaDB filtrelerinde yıl bazlı arama için kullanılır.

---

## Pipeline İzleme

### `tracer.py`
`PipelineTracer` ve `_PhaseContext`. Her pipeline aşamasının (filter_extraction, retrieval, context_building, generation) süresini otomatik olarak ölçer.

```python
tracer = PipelineTracer()
with tracer.phase("retrieval", block="gpu1", model="nomic-embed-text-v2-moe") as ctx:
    results = retriever.retrieve(query)
    ctx.update_details(result_count=len(results))
tracer.print_trace(console)
```

- İstisna durumunda latency kaydedilir, istisna bastırılmaz (`__exit__` `True` döndürmez).
- `on_phase` callback ile UI'ye canlı ilerleme bildirimi gönderilebilir.
- Agent pipeline için `planning → retrieval → re_retrieval → answering → validation` aşamalarını da destekler.

---

## Parsing Alt Paketi (`parsing/`)

PDF ve yapılandırılmış belgelerden chunk üretimi için araçlar.

### `parsing/docling_manager.py`
Ana parser. `DoclingManager` sınıfı PDF'leri OCR ile işler (EasyOCR, Tesseract veya macOS OCR), iki kademeli önbellek yazar (atom önbelleği + chunk önbelleği) ve `(full_text, chunks)` tuple'ı döndürür.

Üç chunking modu:
1. **Greedy** — `packer.py:greedy_pack()` ile karakter bazlı paketleme
2. **Author-aware** — `author_extractor.py` + `segment_pack.py` ile yazar geçişlerinde soft-split
3. **HybridChunker** — Docling'in token tabanlı chunk'layıcısı + `_min_token_merge()`

### `parsing/packer.py`
`greedy_pack(atoms, min_chars, max_chars)` — `List[str]` atomlarını `min_chars`/`max_chars` sınırlarına göre birleştirir. Kısa paragrafların ayrı chunk olarak kalmasını engeller.

### `parsing/author_extractor.py`
Durum makinesi tabanlı yazar metadata çıkarıcı. `tag_atoms()` atom listesini gezerek yazar geçişlerini tespit eder ve her atom'a `author`/`author_role` metadata'sı ekler. `tag_chunks_post_hoc()` önceden oluşturulmuş chunk'lara post-hoc etiketleme uygular.

### `parsing/segment_pack.py`
`segment_aware_pack()` — çok yazarlı belgeler için yazar geçişlerinde soft-split yapan paketleyici. Char-weighted birincil yazar, yazar listesi ve segment bilgileri metadata olarak eklenir.

### `parsing/extractors/`
Belge türüne özgü `AuthorSegmentExtractor` uygulamaları. `get_extractor(document_type)` factory fonksiyonu ile erişilir.

| Dosya | Belge Türü | Tespit Ettiği |
|---|---|---|
| `tutanak.py` | TBMM tutanağı | Başkan, milletvekili, bakan konuşma geçişleri |
| `gazete.py` | Gazete/dergi | "Yazan:" byline'ları, muhabir imzaları |
| `onerge.py` | Önerge/kanun teklifi | Milletvekili teklif sahipleri, imzacılar |
| `noop.py` | Varsayılan | Geçiş tespit etmez; tüm atomlar `initial_author`'ı devralır |

---

## Çapraz Referans

| Modül | Bağımlı Olduğu `src/common` Modülleri |
|---|---|
| `filter_translators.py` | `schemas.py` (FilterCriteria) |
| `embeddings.py` | — (dış: `langchain_ollama`, `trainer` lazy import) |
| `parsing/docling_manager.py` | `parsing/packer.py`, `parsing/author_extractor.py`, `parsing/segment_pack.py`, `parsing/extractors/` |
| `parsing/segment_pack.py` | `parsing/author_extractor.py` (TaggedAtom) |
| `parsing/extractors/*` | `parsing/author_extractor.py` (AuthorSegmentExtractor, AuthorTransition) |
| Diğer tüm modüller | Bağımsız |
