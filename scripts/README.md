# scripts/

Bu klasör, projenin tüm pipeline adımlarını, değerlendirme araçlarını ve yardımcı scriptlerini bir **Python paketi** olarak barındırır.

## Neden `scripts/` Paketi?

`scripts/` klasörü Python paketi olarak yapılandırılmıştır (`__init__.py` içerir). Bu sayede scriptler hem doğrudan çalıştırılabilir (`python scripts/chat.py`) hem de modül olarak çağrılabilir (`python -m scripts.chat`). Modül çağrımı önerilir: proje kökünden bağımsız olarak çalışır çünkü tüm dosya yolları `settings.py` içindeki `PROJECT_ROOT`'a göre mutlak olarak çözümlenir.

Proje kökündeki eski scriptler (`data_prep.py`, `setup_fts.py`, `embed_and_index.py`, `text_minutes_parser.py`, `embed_and_index_text_minutes.py`) hâlâ çalışır ancak deprecated sayılır. Yeni işler için `scripts/` kullanılmalıdır.

---

## Veri Pipeline'ı — Çalıştırma Sırası

Gazete yeniden ingest (jina-v3 + late chunking):

```bash
# 1. CSV'den manifest JSON oluştur
python -m scripts.ingest --csv gazete-rag-001.csv --collection gazete_arsivi_jina_v3 --output gazete_ingestion.json

# 2. Manifest'i pipeline'a gönder
python -m src.trainer.ingestion.ingest --request gazete_ingestion.json
```

TBMM tutanakları:

```bash
python -m scripts.parse_minutes_urls <url1> [<url2> ...]   # HTML → JSON
python -m src.trainer.ingestion.ingest --request tutanaklar_ingestion.json
```

Hepsini tek seferde yeniden kur:

```bash
python -m scripts.reindex_all
```

---

## Script Referansı

### Uygulama

#### `chat.py`
Ana sohbet arayüzü. `python chat.py` ile doğrudan ya da `python -m scripts.chat` ile çalışır. Altında `src/ui/chat.py` çalışır; Normal mod, `/müfettiş`, `/kaynak`, `/debug` komutlarını destekler.

```bash
python -m scripts.chat
# veya
python chat.py
```

---

### Ingestion

#### `parse_minutes_urls.py`
TBMM tutanak URL'lerini HTML'den ayrıştırıp JSON olarak kaydeder. `ingest_minutes.py`'dan önce çalıştırılır.

```bash
python -m scripts.parse_minutes_urls <url1> [<url2> ...] [--output tutanak/extracted]
```

| Argüman | Açıklama |
|---|---|
| `urls` | Ayrıştırılacak HTML tutanak URL'leri (bir veya fazla) |
| `--output DIR` | JSON çıktı klasörü (varsayılan: `tutanak/extracted`) |

---

#### `reindex_all.py`
Tüm pipeline'ı baştan sona çalıştırır: gazete kupürleri ve tutanaklar için 4 adımın tamamı.

```bash
python -m scripts.reindex_all
```

---

### Değerlendirme & Benchmark

#### `evaluate.py`
RAG sisteminin geri getirme kalitesini test eder. Opsiyonel olarak LLM hakem skorlaması yapar.

```bash
python -m scripts.evaluate                    # tam değerlendirme
python -m scripts.evaluate --quick            # LLM hakem olmadan, hızlı
python -m scripts.evaluate --inspect-chunks   # önce chunk kalitesini kontrol et
python -m scripts.evaluate --golden-only      # yalnızca altın etiketli sorular
```

| Argüman | Açıklama |
|---|---|
| `--queries` | Test sorgu fixture JSON dosyası (varsayılan: `tests/fixtures/eval_queries_tr.json`) |
| `--no-judge` / `--quick` | LLM hakem adımını atla |
| `--inspect-chunks` | Chunk kalite denetimi çalıştır |
| `--golden-only` | Yalnızca altın yanıtlı sorular, hakemi zorla |

---

#### `benchmark.py`
Farklı koleksiyonlar veya embedding modelleri arasında geri getirme kalitesini A/B karşılaştırır. YAML config ile çalışır.

```bash
python -m scripts.run_benchmark --config experiments/jina_v3_vs_nomic.yaml
python -m scripts.run_benchmark --config experiments/jina_v3_vs_nomic.yaml --output artifacts/result.json
```

| Argüman | Açıklama |
|---|---|
| `--config, -c` | YAML deney dosyası (zorunlu) |
| `--output, -o` | JSON çıktı dosyası (opsiyonel) |

**YAML şeması:**
```yaml
name: Deney adı
collections: [koleksiyon1, koleksiyon2]
fixture: tests/fixtures/eval_queries_docling_d20.json
top_k: [1, 3, 5, 10]
reranker: false
fetch_k: 50
```

Çıktı: Precision@k, Recall@k, Hit@k, MRR, NDCG@10.

---

#### `bench_embedding_context.py`
Farklı bağlam penceresi boyutlarında (4k, 8k, 32k token) late-chunking geri getirme kalitesini karşılaştırır.

```bash
USE_LOCAL_LATE_CHUNKING=1 python scripts/bench_embedding_context.py \
  --doc path/to/tutanak.pdf \
  --max-tokens 4096 8192
```

| Argüman | Açıklama |
|---|---|
| `--doc` | Test edilecek uzun belge (PDF, DOCX veya TXT) — zorunlu |
| `--max-tokens` | Karşılaştırılacak pencere boyutları, boşlukla ayrılmış (varsayılan: `4096 8192`) |
| `--queries` | Altın sorgu fixture dosyası |
| `--model` | Jina embedding model adı |

> `USE_LOCAL_LATE_CHUNKING=1` ortam değişkeni zorunlu.

---

### Keşif & Araçlar

#### `vector_explorer.py`
ChromaDB koleksiyonlarını görsel olarak keşfetmek ve altın değerlendirme verisi oluşturmak için Streamlit tabanlı web arayüzü.

```bash
streamlit run scripts/vector_explorer.py
```

**Sekmeler:**
- **Tab 1 — Vektör Arama:** ChromaDB'yi sorgula, sonuçları gör, altın veri kuyruğuna ekle
- **Tab 2 — Altın Veri Oluşturucu:** Arama sonuçlarından otomatik doldurulan değerlendirme sorguları oluştur
- **Tab 3 — Mevcut Verilere Gözat:** `tests/fixtures/eval_*.json` dosyalarındaki tüm altın veriyi incele

> Streamlit paketi gerekmektedir.

---

#### `compare_ocr.py`
Bir PDF üzerinde farklı OCR motorlarının çıktı kalitesini karşılaştırır.

```bash
python scripts/compare_ocr.py <pdf_dosyası> [easyocr,tesseract,mac]
```

| Argüman | Açıklama |
|---|---|
| `pdf_path` | Test edilecek PDF dosyası (zorunlu) |
| motorlar | Virgülle ayrılmış motor listesi (varsayılan: `easyocr,tesseract`) |

Çıktı: `output/<dosya_adı>_<motor>.txt` metin dosyaları + karakter/chunk istatistikleri.

---

#### `try_docling.py`
Docling PDF ayrıştırma pipeline'ını test eder; `tutanak/raw/` klasöründeki PDF'leri ChromaDB `tbmm_minutes_docling_test` koleksiyonuna indeksler.

```bash
python scripts/try_docling.py
```

> Deneysel script. Klasör yapısından dönem/yıl/birleşim metadata'sını ayrıştırır (D20, Y1, B1_... formatı).

