# RAG-poc Mimari Kılavuzu

> Bu doküman, sistem mimarisinin katmanlı refaktör sonrası "tek doğru kaynağı" (single source of truth) dır. Tüm yeni geliştirmeler burada açıklanan kurallara uymalıdır.

---

## Genel Bakış

Yerel LLM tabanlı, çok-koleksiyonlu hibrit RAG (Retrieval-Augmented Generation) sistemidir. Farklı belge tiplerini (tutanak, gazete kupürü, kanun teklifi, PDF rapor) tek bir arama ve yanıt üretim altyapısında birleştirir. Temel bileşenler:

- **Yerel Dil Modelleri (Local LLMs)**: Ollama üzerinden (`gemma4:latest` vb.)
- **Vektör Veritabanı**: ChromaDB (gömülü modda)
- **Metin Arama**: SQLite FTS5
- **Hibrit Arama**: Vektör benzerliği + BM25 (RRF veya Rerank ile birleştirilmiş)
- **Gelişmiş Veri Yükleme**: IBM Docling ve Late Chunking (Bağlamsal Embedding)

---

## Katman Diyagramı

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             Giriş Noktaları                              │
│                                                                          │
│  ┌─────────────────────────────┐   ┌──────────────────────────────────┐ │
│  │     CLI / chat.py           │   │       MCP İstemcileri            │ │
│  │     src/ui/*.py             │   │  (Claude Desktop, Open WebUI, …) │ │
│  └──────────────┬──────────────┘   └────────────────┬─────────────────┘ │
└─────────────────┼────────────────────────────────────┼───────────────────┘
                  │ kullanır                           │ SSE (port 8001/8002/8003)
                  ▼                                    ▼
┌─────────────────────────────┐   ┌──────────────────────────────────────┐
│    Generator.RAGService     │   │          MCP Sunucu Katmanı          │
│  src/generator/service.py   │   │                                      │
│  src/generator/             │   │  press_server.py   (port 8001)       │
│    deep_pipeline.py         │   │  minutes_server.py (port 8002)       │
└──────────────┬──────────────┘   │  router_server.py  (port 8003)       │
               │                  │  _base.py  (paylaşılan altyapı)      │
               │                  └────────────────┬─────────────────────┘
               │                                   │ doğrudan çağırır
               └───────────────────┬───────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          Retriever Katmanı                               │
│  src/retriever/                                                          │
│  vector_retriever.py    minutes_retriever.py    query_parser.py          │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ kullanır
┌──────────────────────────────────▼───────────────────────────────────────┐
│                          Generator Katmanı                               │
│  src/generator/ollama_generator.py    prompts.py                         │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ her ikisi de kullanır
┌──────────────────────────────────▼───────────────────────────────────────┐
│                           Common Katmanı                                 │
│  src/common/{text,dates,chunking,embeddings,chroma,sqlite_io,parsing,    │
│              protocols}.py                                               │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ okur
┌──────────────────────────────────▼───────────────────────────────────────┐
│                           Config Katmanı                                 │
│  src/config/{settings.py, collections.py}    models.yaml                │
└──────────────────────────────────────────────────────────────────────────┘

Evaluator  →  kullanır  →  RAGService (Retriever + Generator)
Trainer    →  kullanır  →  Common / Ingestion (Veri Yükleme Hattı)
```

---

## Katman Sorumlulukları

### `src/config/`
Sistemdeki tüm "sabitler" burada toplanmıştır:
- `settings.py`: Global ayarlar (DB yolları, genel eşikler, ortam değişkenleri).
- `collections.py`: Koleksiyon bazlı ayarlar (parçalama boyutu, kullanılan modeller, meta veri şemaları).

### `src/common/`
Katmanlar arası paylaşılan durumsuz yardımcı araçlar:
- `text.py`: Türkçe normalizasyon (`normalize_tr`) ve anahtar kelime penceresi çıkarma.
- `dates.py`: Türkçe tarih ayrıştırma ve ISO formatına dönüştürme.
- `embeddings.py`: `Ollama` ve yerel `Jina v3` embedding fabrikaları.
- `parsing/`: `DoclingManager` (yapısal döküman analizi) ve `packer.py` (akıllı parçalama).
- `chroma.py` / `sqlite_io.py`: Veritabanı bağlantı yardımcıları.

### `src/trainer/ingestion/` (Yeni Veri Yükleme Hattı)
Modern, yapısal analize dayalı veri yükleme süreci:
- `adapters/`: Farklı belge tiplerini ortak `DocumentInput` şemasına dönüştüren adapter sınıfları.
- `pipeline.py`: Uçtan uca veri yükleme akışı (Docling -> Late Chunking -> Chroma).
- `manifest.py`: Hangi dosyaların yüklendiğini takip eden SQLite tabanlı manifest.

### `src/retriever/`
Arama ve bilgi getirme mantığı:
- `query_parser.py`: Sorgu analizi, tarih ayıklama ve kaynak yönlendirme.
- `vector_retriever.py`: Vektör araması (ANN) ve Cross-Encoder reranking tabanlı üretim getiricisi.
- `minutes_retriever.py`: Uzun yapısal belgeler için özelleşmiş hibrit (BM25 + Vektör + RRF) getirici.
- `reranker.py`: Sonuçların doğruluğunu artıran Cross-Encoder modeli.

### `src/generator/`
LLM etkileşimi:
- `service.py` (`RAGService`): UI ve scriptler için ana giriş noktası. Retriever ve Generator'ı birleştirir.
- `ollama_generator.py`: Ollama üzerinden metin üretimi ve sorgu genişletme.
- `deep_pipeline.py`: "Müfettiş modu" için çok adımlı akıl yürütme hattı.

### `src/mcp/`
Harici MCP istemcileri (Claude Desktop, Open WebUI vb.) için SSE tabanlı üç bağımsız sunucu:
- `press_server.py` (port 8001): Gazete arşivini `VectorRetriever` üzerinden sunar; `search_press_archive` aracını tanımlar.
- `minutes_server.py` (port 8002): TBMM tutanaklarını `MinutesRetriever` üzerinden sunar; yıl/parti/konuşmacı filtrelerini destekler.
- `router_server.py` (port 8003): İki arşivi çapraz sorgular; `RAGService` + `DeepPipeline` kullanır ve `generate_report` aracını da sunar.
- `_base.py`: FastAPI + SSE transport fabrikası — üç sunucunun paylaştığı ortak altyapı.
- `server.py`: Eski (legacy) uygulama; artık kullanılmıyor.

### `src/evaluator/`
Çevrimdışı değerlendirme araçları:
- Geri getirme metrikleri (Precision@K, Recall@K, MRR).
- `LLMJudge`: Ollama tabanlı yanıt kalitesi puanlama.
- Latans raporları ve regresyon testleri.

---

## CLI Referansı

| Komut | Açıklama |
|---|---|
| `python chat.py` | İnteraktif sohbet arayüzü |
| `python -m src.mcp.press_server` | Gazete arşivi MCP sunucusunu başlat (port 8001) |
| `python -m src.mcp.minutes_server` | TBMM tutanakları MCP sunucusunu başlat (port 8002) |
| `python -m src.mcp.router_server` | Çapraz arama + rapor MCP sunucusunu başlat (port 8003) |
| `python -m src.trainer.ingestion.ingest --request manifest.json` | Belge yükleme hattı |
| `python -m scripts.ingest_onerge` | Kanun teklifleri için veri yükleme hattı |
| `python -m scripts.evaluate` | Değerlendirme sistemini çalıştır |
| `python -m scripts.reindex_all` | Tüm koleksiyonları sıfırdan yeniden dizinle |
| `python -m pytest tests/` | Birim testleri çalıştır |

---

## Ortam Değişkenleri

| Değişken | Varsayılan | Amacı |
|---|---|---|
| `RAG_ENV` | `local` | `local` veya `remote` Ollama kullanımı |
| `RAG_LLM_MODEL` | `gemma4:latest` | Üretim için kullanılacak LLM |
| `RAG_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Vektörleştirme modeli |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` veya `vector` arama modu |
| `USE_RERANKER` | `1` | Reranker kullanımını açar/kapatır |
| `USE_LOCAL_LATE_CHUNKING`| `0` | Yerel Jina v3 (Late Chunking) kullanımını açar |

---

## Parçalama ve Bağlam (Chunking)

Sistem artık statik parçalama yerine, döküman yapısını anlayan akıllı paketleme (`packer.py`) ve bağlamsal embedding (`Late Chunking`) kullanmaktadır. Ayarlar `src/config/collections.py` içerisinden koleksiyon bazlı yönetilir.

---

## Yeni Veri Kaynağı Ekleme

1. `src/config/collections.py` içerisine yeni koleksiyon tanımını ekleyin.
2. `src/trainer/ingestion/adapters/` altında veriye özel bir adapter sınıfı oluşturun.
3. `scripts/` altında `ingest_new_source.py` scriptini oluşturun.
4. `src/retriever/query_parser.py` içerisine yönlendirme anahtar kelimelerini ekleyin.
