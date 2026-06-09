# Gazete Arşivi RAG (Retrieval-Augmented Generation) Sistemi

Türkçe gazete arşivi ve TBMM tutanakları üzerinde hibrit arama ve LLM yanıt üretimi yapan yerel RAG sistemi.

> **Mimari detaylar için:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Hızlı Başlangıç

```bash
# 1. Ortamı kur
bash setup_env.sh
source .venv/bin/activate

# 2. Ollama modellerini indir
ollama pull nomic-embed-text-v2-moe
ollama pull gemma4:e2b
ollama pull qwen3.5:9b

# 3. Belgeleri indeksle (örnek manifest ile)
python -m src.trainer.ingestion.ingest --request ornek_ingestion.json

# 4. Sohbeti başlat (agent modu)
python chat.py --agent

# Alternatif: Görsel arayüz
streamlit run scripts/vector_explorer.py
```

### Vector Explorer ile Hızlı Test

Belgeleri indeksledikten sonra `streamlit run scripts/vector_explorer.py` ile tarayıcıda açılan arayüzden:

- **Koleksiyon seçimi** — sidebar'dan bir veya birden fazla koleksiyon seçin
- **Sorgu** — arama kutusuna doğal dil sorgusu yazın
- **Auto Filter Extractor** — yıl, yazar, dönem gibi filtreleri sorgudaki doğal dilden otomatik çıkarır (`"Deniz Baykal'ın 1996 konuşmaları"` → `year=1996, author=Deniz Baykal`)
- **Reranker** — cross-encoder ile sonuçları yeniden sıralar; `top_k` ve `fetch_k` slider'larından ayarlanır
- **Sorgu limitleri** — `Top K` (LLM'e gidecek sonuç sayısı) ve `Fetch K` (reranker'a beslenen aday havuzu) arayüzden değiştirilebilir

---

## İlk İndeksleme — Hızlı Örnek

Aşağıdaki adımlar, `ornek_ingestion.json` ile TBMM tutanaklarını `tbmm_tutanaklar_nomic_v2` koleksiyonuna indeksler. Örnek dosya; 20. dönem (1996) ve 23. dönem (2007) TBMM birleşim tutanaklarını kapsar.

### 0. Koleksiyon Seç veya Oluştur

```bash
# Kayıtlı koleksiyonları listele
python -m src.trainer.ingestion.ingest --list-collections
```

Çıktıda mevcut koleksiyonlar, embedding modelleri ve context boyutları görünür. Uygun bir koleksiyon varsa adını not edin.

Yeni bir koleksiyon gerekiyorsa interaktif sihirbazı çalıştırın:

```bash
python -m src.trainer.ingestion.ingest --add-collection
```

Sihirbaz koleksiyon anahtarı, ChromaDB adı, embedding modeli ve chunk parametrelerini sorar; onayda `models.yaml`'a yazar. Python kodu değişikliği gerekmez.

> **Yeni embedding modeli eklemek:** `models.yaml`'daki `model_specs` bloğuna yeni bir giriş ekleyin; sihirbaz bir sonraki çalıştırmada bu modeli listeler. HuggingFace modelleri (örn. `jinaai/jina-embeddings-v4`) ilk indekslemede otomatik indirilir — kurulum gerekmez, sadece disk alanı ve bekleme süresi.

### 1. `ingest_request.json` Hazırla

```json
{
  "version": "1.0",
  "collection": "tbmm_tutanaklar_nomic_v2",
  "documents": [
    {
      "document_id": "tutanak-20-01-02-19960108",
      "document_type": "tutanak",
      "document_source": "./tutanak/tbmm20001002.pdf",
      "document_date": "1996-01-08",
      "period": 20,
      "legislative_year": 1,
      "session": 2,
      "source_name": "TBMM Tutanakları"
    }
  ]
}
```

> Tam örnek: [`ornek_ingestion.json`](ornek_ingestion.json)

### 2. Doğrula ve İndeksle

```bash
# Şema + dosya yolu kontrolü
python -m src.trainer.ingestion.ingest --validate ornek_ingestion.json

# İndeksle (idempotent — aynı dosyayı tekrar çalıştırmak güvenlidir)
python -m src.trainer.ingestion.ingest --request ornek_ingestion.json

# Durum
python -m src.trainer.ingestion.ingest --status --collection tbmm_tutanaklar_nomic_v2
```

### 3. Sorgula

```bash
python chat.py --agent
```

---

## Multi-Koleksiyon Sorgulama

Sistem, **aynı anda birden fazla koleksiyonda arama** yapabilir. Bu özellik, farklı embedding modellerini karşılaştırmak, birden fazla belge kaynağını sorgulamak ve sonuçlarını birleştirmek (RRF füzyonu) için kullanışlıdır.

### Araç: Vector Explorer (Streamlit)

```bash
streamlit run scripts/vector_explorer.py
```

Sideba'da **"Koleksiyonları Seçin"** multi-select dropdown'ı görürsünüz. Birden fazla koleksiyonu seçin; her sorgu seçili koleksiyonlarda paralel olarak çalışır ve RRF (Reciprocal Rank Fusion) ile sonuçlar birleştirilir.

**Örnek:** `gazete_arsivi`, `tbmm_tutanaklar_jina_v3`, `onerge_collection` seçiliyse, sorgunuz üç koleksiyonda eşzamanlı olarak aranır. Sonuçlar her biri **`[collection_name]`** etiketi ile gösterilir (örn. `[gazete_arsivi] Belgeler... | [tbmm_tutanaklar_jina_v3] Tutanaklar...`).

### Chat UI'de Multi-Koleksiyon

```bash
python chat.py
```

Sohbeti başlatırken **"Koleksiyon Seçimi"** isteği göreceksiniz. Varsayılan (virgülle ayrılmış): `gazete_arsivi,tbmm_minutes`. Boş Enter = varsayılanları kabul et. Veya özel koleksiyon listesi yazın.

**Yapılandırma:** `retrieval_config.yaml` (repo root) — bu dosyada her uygulama için varsayılan koleksiyonlar tanımlanır:

```yaml
default_collections:
  vector_explorer:
    - gazete_arsivi
    - tbmm_tutanaklar_jina_v3
    - onerge_collection
  
  chat:
    - gazete_arsivi
    - tbmm_minutes
    - onerge_collection
```

### Teknik Detaylar

- **Retriever:** `MultiSourceRetriever` — her koleksiyonda `VectorRetriever` instance'ı çalıştırır.
- **Füzyon:** RRF (k=60) — aynı belge birden fazla koleksiyonda bulunursa, RRF skorları birikir (daha yüksek rank).
- **Attribution:** Her sonuç `"collection": "koleksiyon_adı"` metadata alanı taşır — hangi kaynaktan geldiği anlaşılır.
- **Deduplication:** Aynı `document_id` + `chunk_index` birden fazla koleksiyonda görülürse, sonuçta bir kez yer alır; RRF skoru birikir.

### Retrieval Modları

`MultiSourceRetriever` üç retrieval modu destekler:

| Method | Amaç | Kullanım Alanı |
|--------|------|----------------|
| `retrieve()` | RRF füzyonu ile koleksiyonlar arası sıralama | Çapraz koleksiyon ranking, deduplication |
| `retrieve_per_collection()` | Koleksiyon bazlı gruplanmış sonuçlar | vector_explorer UI, koleksiyon bazlı analiz |
| `retrieve_balanced()` | Her koleksiyondan eşit sayıda sonuç | chat.py, LLM context oluşturma |

**`retrieve_balanced()`** multi-collection modda chat.py tarafından kullanılır. Her seçili koleksiyondan `context_weight` chunk (varsayılan 5) getirir, collection + doc_type attribution ile birleştirir. RRF füzyonu yok — her kaynak LLM context'te eşit temsil edilir.

**`context_weight`** `CollectionSpec`'te varsayılan 5 olarak tanımlıdır. Call time'da `per_collection_k` parametresi ile override edilebilir.

Örnek:
```python
# Varsayılan: her koleksiyonun context_weight değerini kullanır
results = retriever.retrieve_balanced("sorgu")

# Override: koleksiyon başına 10 sonuç
results = retriever.retrieve_balanced("sorgu", per_collection_k=10)
```

### Örnek Senaryo

1. **Gazete + Tutanak Karşılaştırması:**
   - Sorgu: `"2023 ekonomi politikası"`
   - Gazete arşivinden güncel haberler + TBMM tutanaklarından resmi tartışmalar bir sorguda görürsünüz.

2. **Model Değerlendirmesi:**
   - `tbmm_minutes_docling_jina_v3` vs `tbmm_minutes_docling_jina_v4` — aynı sorguda iki modelin sonuçlarını yan yana karşılaştırın.

3. **Belge Tipi Filtreleme:**
   - Sadece `tutanak` ve `onerge` seçin — gazete haberleri dışlayın.

---

## Doküman İndeksleme Rehberi

Sistem, **JSON manifest** üzerinden çalışır. Her belge `ingest_request.json` ile tanımlanır.
Metadata klasör yapısından değil, JSON'dan gelir.

### Temel Kavramlar

| Kavram | Açıklama |
|---|---|
| `ingest_request.json` | İndekslenecek belgelerin listesi. Tek giriş noktası. |
| `document_id` | Her belgenin deterministik kimliği. Dışarıdan sağlanır. |
| `CollectionSpec` | Koleksiyon + model tanımı. `src/config/collections.py`'te kayıtlı. |
| `DocumentAdapter` | Belge tipine göre parser (tutanak, press_clip, pdf_report). |
| `Manifest` | SQLite tablosu. Hangi belgelerin işlendiğini takip eder. |

### Adım Adım: İlk İndeksleme

#### 1. Model ve Koleksiyon Seçin

```bash
# Mevcut koleksiyonları ve modelleri gör
python -m src.trainer.ingestion.ingest --list-collections

# Mevcut belge tiplerini gör
python -m src.trainer.ingestion.ingest --list-types
```

Çıktı:
```
Anahtar (YAML)    ChromaDB Adı                  Model                    Context   Dim   Late Chunking   Chunk Sayısı
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
gazete_arsivi       gazete_arsivi                 nomic-embed-text-v2-moe    512    768       ✗               1 240
tbmm_minutes     tbmm_minutes                  nomic-embed-text-v2-moe    512    768       ✗                   —
tbmm_minutes_docling_jina_v3   tbmm_minutes_docling_jina_v3  jinaai/jina-embeddings-v3 8192   1024       ✓               8 431
tbmm_minutes_docling_jina_v4   tbmm_minutes_docling_jina_v4  jinaai/jina-embeddings-v4 32768  1024       ✓                   —
```

- **Anahtar (YAML)** — `ingest_request.json`'daki `collection` alanına yazılan değer.
- **ChromaDB Adı** — vector_explorer UI ve `--status` çıktısında görünen ad.
- `—` chunk sayısı koleksiyonun henüz indekslenmediğini gösterir.

#### 2. `ingest_request.json` Oluşturun

Her belge için bir JSON nesnesi. Metadata anahtarları **İngilizce**, değerler **Türkçe**.

**Örnek — TBMM Tutanakları (PDF):**

```json
{
  "version": "1.0",
  "collection": "tbmm_minutes_docling_jina_v4",
  "batch_id": "tutanak-d20-v1",
  "documents": [
    {
      "document_id": "tbmm-20-1-1-19960108",
      "document_source": "tutanak/raw/D20/Y1/B1_1996-01-08/tbmm20001005.pdf",
      "document_type": "tutanak",
      "document_date": "1996-01-08",
      "period": 20,
      "legislative_year": 1,
      "session": 1,
      "author": null,
      "source_name": "TBMM Tutanakları",
      "title": "6. Birleşim",
      "metadata": {
        "oturum_baskani": "Mustafa Kalemli",
        "katip_uyeler": ["Adil Biçer"],
        "acilma_saati": "15:00"
      }
    },
    {
      "document_id": "tbmm-20-1-2-19960118",
      "document_source": "tutanak/raw/D20/Y1/B2_1996-01-18/tbmm20001002.pdf",
      "document_type": "tutanak",
      "document_date": "1996-01-18",
      "period": 20,
      "legislative_year": 1,
      "session": 2,
      "ocr": false
    }
  ]
}
```

**Örnek — PDF Rapor (URL kaynaklı):**

```json
{
  "version": "1.0",
  "collection": "tbmm_minutes_docling_jina_v4",
  "batch_id": "tutanak-d26-v1",
  "documents": [
    {
      "document_id": "tbmm-26-1-1-20170316",
      "document_source": "https://www5.tbmm.gov.tr/tutanaklar/TUTANAK/TBMM/d26/c001/tbmm26001001.pdf",
      "document_type": "tutanak",
      "document_date": "2017-03-16",
      "period": 26,
      "legislative_year": 1,
      "session": 1,
      "source_name": "TBMM Tutanakları",
      "title": "1. Birleşim"
    }
  ]
}
```

> **Not:** URL kaynaklı belgeler `data_lake/downloads/{collection}/{document_id}/` altına indirilir. İlk çalıştırmada dosya indirilir, sonraki çalıştırmalarda sunucunun `ETag`/`Last-Modified` başlıkları sayesinde değişmemişse tekrar indirilmeden atlanır.

**Örnek — Kanun Teklifi/Önerge (URL kaynaklı):**

```json
{
  "version": "1.0",
  "collection": "onerge_jina_v4",
  "batch_id": "onerge-d28-y4",
  "documents": [
    {
      "document_id": "onerge-28-4-2-3684",
      "document_source": "https://cdn.tbmm.gov.tr/KKBSPublicFile/D28/Y4/T2/WebOnergeMetni/bed7d724-7013-436f-afee-344673b618b5.pdf",
      "document_type": "kanun_teklifi",
      "document_date": "2026-05-11",
      "period": 28,
      "legislative_year": 4,
      "author": "Nevroz UYSAL ASLAN",
      "author_role": "Şırnak Milletvekili",
      "title": "Hukuksal Savunma ve Beyanlarda Dil Seçme Serbestliğinin Sağlanması Amacıyla Bazı Kanunlarda Değişiklik Yapılmasına Dair Kanun Teklifi",
      "source_name": "TBMM Kanun Teklifleri",
      "metadata": {
        "esas_no": "2/3684",
        "ozet": "Teklif ile; mahkemelerde ana dilinde veya kişinin kendisini daha iyi ifade edebileceği başka bir dilde sözlü beyanda bulunulabilmesine ve avukatın bu dilde savunma yapabilmesine ilişkin çeşitli düzenlemeler yapılması öngörülmektedir.",
        "durum": "KOMİSYONDA",
        "detay_url": "/Yasama/KanunTeklifi/4bc50c42-7c10-4346-a85a-019e1755060d"
      }
    }
  ]
}
```

> **Not:** `kanun_teklifi` için `document_source` zorunludur (PDF URL/ path) ve `metadata.esas_no` zorunludur. `document_id` önerisi: `onerge-{period}-{legislative_year}-{esas_no}`.

**Örnek — Gazete Kupürleri (inline metin):**

```json
{
  "version": "1.0",
  "collection": "gazete_arsivi",
  "batch_id": "press-week-20",
  "documents": [
    {
      "document_id": "press-324542",
      "document_source": null,
      "document_type": "press_clip",
      "document_date": "1998-04-07",
      "author": "Ahmet Hakan",
      "author_role": "köşe yazarı",
      "source_name": "Hürriyet",
      "title": "Seçim öncesi son anket sonuçları",
      "topics": "siyaset, seçim, anket",
      "metadata": {
        "dokuman_metni": "Tam makale metni burada...",
        "kayit_no": 324542
      }
    }
  ]
}
```

#### 3. Doğrulayın

```bash
python -m src.trainer.ingestion.ingest --validate ingest_request.json
```

#### 4. İndeksleyin

```bash
# İlk indeksleme
python -m src.trainer.ingestion.ingest --request ingest_request.json

# İkinci çalıştırma — aynı dosya (atlanır)
python -m src.trainer.ingestion.ingest --request ingest_request.json
# Çıktı: 2 atlandı, 0 işlendi

# Sadece değişmiş/yeni belgeleri işle
python -m src.trainer.ingestion.ingest --request ingest_request.json --only-changed

# Önce diff göster (işlem yapmadan)
python -m src.trainer.ingestion.ingest --diff ingest_request.json
```

#### 5. Durumu Kontrol Edin

```bash
# Genel durum
python -m src.trainer.ingestion.ingest --status

# Koleksiyon bazında
python -m src.trainer.ingestion.ingest --status --collection tbmm_minutes_docling_jina_v4

# Tip bazında
python -m src.trainer.ingestion.ingest --status --document-type tutanak
```

#### 6. Silme (Gerekirse)

```bash
python -m src.trainer.ingestion.ingest --delete tbmm-20-1-1-19960108
```

---

### `ingest_request.json` Alanları

| Alan | Türkçe Açıklama | Örnek |
|---|---|---|
| `document_id` | Belgenin deterministik kimliği (dışarıdan) | `tbmm-20-1-1-19960108` |
| `document_source` | PDF dosya yolu veya `http(s)` URL (null = inline) | `tutanak/raw/D20/...pdf` veya `https://cdn.tbmm.gov.tr/...pdf` |
| `document_type` | Adapter seçici | `tutanak`, `press_clip`, `pdf_report` |
| `collection` | Hedef koleksiyon | `tbmm_minutes_docling_jina_v4` |
| `document_date` | ISO tarih | `1996-01-08` |
| `year` | Yıl (sorgu filtresi için) | `1996` |
| `period` | TBMM dönemi | `20` |
| `legislative_year` | TBMM yasama yılı | `1` |
| `session` | TBMM birleşim no | `1` |
| `author` | Yazar / konuşmacı | `Mustafa Kalemli` |
| `author_role` | Rol | `başkan`, `köşe yazarı` |
| `source_name` | Kaynak adı | `Hürriyet`, `TBMM Tutanakları` |
| `title` | Başlık | `6. Birleşim` |
| `topics` | Konu etiketleri | `siyaset, seçim` |
| `ocr` | OCR kontrolü. `false` = dijital doğumlu PDF (metin katmanı var). Varsayılan: `true` | `false` |
| `metadata` | Tip-spesifik ekstralar | `{acilma_saati: "15:00"}` |

**Kural:** Sorgu/ filtrede kullanacağınız alanlar yukarıdaki canonical alanlarda olmalı.
`metadata` sadece provenance (köken bilgisi) içindir — üzerinde filtre yapılamaz.

---

### Koleksiyon Yönetimi

#### Yeni Koleksiyon Ekleme — `--add-collection` Sihirbazı

Yeni bir koleksiyonu en hızlı yol interaktif sihirbazla eklemektir; `models.yaml`'ı elle düzenlemenize gerek kalmaz.

```bash
python -m src.trainer.ingestion.ingest --add-collection
```

Sihirbaz sizi 5 adımda yönlendirir:

1. **Koleksiyon adı** — hem `models.yaml`'daki kayıt anahtarı hem ChromaDB koleksiyon adı (`snake_case`, benzersiz). `ingest_request.json`'daki `collection` alanına yazılan değerdir.
2. **ChromaDB dizini** — vektörlerin kaydedileceği dizin (örn. `data_lake/press_clips_vectors`)
3. **Embedding modeli** — `models.yaml`'daki kayıtlı modeller listesinden seçim
4. **Belge tipi** — `tutanak`, `gazete`, `onerge`, `custom`
5. **Chunk parametreleri** — varsayılanları kabul edebilir ya da özelleştirebilirsiniz

Sonunda özet gösterilir ve onayınızla `models.yaml`'a yazılır. İsteğe bağlı olarak bu koleksiyonu o belge tipi için varsayılan yapabilirsiniz.

#### Yeni Koleksiyon Tanımlama (Manuel)

Otomasyon veya CI ortamları için `models.yaml` doğrudan da düzenlenebilir — Python kodu değişikliği gerekmez.

```yaml
# models.yaml

model_specs:
  jinaai/jina-embeddings-v4:          # 1. Yeni model varsa ekle
    max_context_tokens: 32768
    overlap_tokens: 256
    embed_dim: 1024
    supports_late_chunking: true
    description: "Jina v4: 32K context, uzun dokümanlar için ideal."

collections:
  tbmm_minutes_docling_jina_v4:                    # 2. Koleksiyonu tanımla
    collection_name: tbmm_minutes_docling_jina_v4
    chroma_path: data_lake/parliament_digital_born_minutes_vectors
    sqlite_path: data_lake/parliament_digital_born_minutes.db
    embed_model: jinaai/jina-embeddings-v4
    doc_type: tutanak
    min_chunk_chars: 400
    max_chunk_chars: 1500
    max_chunk_tokens: 512
    min_chunk_tokens: 384
```

Kaydettikten sonra `get_spec("tbmm_minutes_docling_jina_v4")` ile kullanabilirsiniz.

#### Örnek Manifest Dosyaları

Proje kökünde hazır referans dosyaları bulunur:

| Dosya | Açıklama |
|---|---|
| `ornek_tutanak.json` | 19. dönem TBMM tutanakları — URL kaynaklı toplu indeksleme örneği |
| `ornek_onerge_manifest.json` | 28. dönem kanun teklifleri — `kanun_teklifi` tipi örneği |
| `ornek_ingestion.json` | Yerel PDF yolu kullanan karma tutanak listesi |

Bu dosyaları kendi `ingest_request.json`'ınız için başlangıç noktası olarak kullanabilirsiniz.

#### Python API ile İndeksleme

```python
from src.config.collections import get_spec
from src.trainer.ingestion.adapters.base import DocumentInput
from src.trainer.ingestion.pipeline import IngestionPipeline

spec = get_spec("tbmm_minutes_docling_jina_v4")
pipe = IngestionPipeline(spec=spec)

doc = DocumentInput(
    document_id="tbmm-20-1-1-19960108",
    document_type="tutanak",
    collection_name="tbmm_minutes_docling_jina_v4",
    document_source="tutanak/raw/D20/...pdf",
    document_date="1996-01-08",
    period=20,
    session=1,
)

result = pipe.run_document(doc)
print(result.status)  # "done", "skipped", "failed"
```

---

### DevOps Rehberi

**Sizin sorumluluğunuz:** `ingest_request.json` üretmek.
**Sistemin sorumluluğu:** doğrulamak, indekslemek, takip etmek.

```bash
# CI/CD pipeline örneği
# 1. Yeni tutanakları tarayıcıdan çek, JSON üret
python your_scraper.py --output ingest_new_period.json

# 2. Doğrula
python -m src.trainer.ingestion.ingest --validate ingest_new_period.json

# 3. Sadece yeni/değişmişleri işle (idempotent)
python -m src.trainer.ingestion.ingest --request ingest_new_period.json --only-changed

# 4. Rapor al
python -m src.trainer.ingestion.ingest --status --collection tbmm_minutes_docling_jina_v4
```

---

### Performans ve Önbellekleme

#### Parse Cache (Ayrıştırma Önbelleği)
PDF belgelerinin Docling ile ayrıştırılması ve OCR yapılması maliyetli bir işlemdir. Sistem, aynı belgenin (veya aynı belgenin farklı modellerle) tekrar tekrar parse edilmesini önlemek için bir önbellek mekanizması kullanır.

*   **Klasör:** `data_lake/parse_cache/`
*   **Mantık:** Dosya içeriği + OCR motoru + paketleme parametrelerinin birleşimiyle benzersiz bir **MD5 hash** (Cache Key) oluşturulur.
*   **Avantaj:** Aynı tutanak dosyasını hem Jina hem de Nomic koleksiyonuna ingest ederken, OCR işlemi sadece ilk seferinde yapılır. İkinci seferde milisaniyeler içinde önbellekten okunur.

#### Paketleme Analizi (Packer Analysis)
Sistem, Docling'den çıkan çok küçük metin parçalarını (atom) birleştirerek anlamlı bloklar oluşturur. Bu stratejinin (Greedy Packer) verimliliğini ve döküman bazlı ihtiyacını analiz etmek için hazırlanan araçtır:

*   **Script:** `scratch/analyze_packer.py`
*   **Kullanım:** `python scratch/analyze_packer.py`
*   **Ne Yapar?** `data_lake/parse_cache/` altındaki tüm atom dosyalarını tarar ve şu istatistikleri üretir:
    *   **İhtiyaç Skoru:** 500 karakter altındaki parçaların tüm parçalara oranı (%90+ ise paketleme kritiktir).
    *   **Verimlilik:** Paketleme sonrası VectorDB'deki kayıt sayısında ne kadar tasarruf sağlandığı (%80+ azalma beklenir).
    *   **Bağlam Kalitesi:** Ortalama parça uzunluğunun (örneğin 180'den 1250 karaktere) nasıl iyileştiği.

---

### Chunk-Düzeyi Yazar/Konuşmacı Metadata Çıkarımı

Çok yazarlı dökümanlarda (TBMM tutanağı, kanun teklifi, önerge) bir chunk yazar değişiminin ortasına denk gelebilir. Sistem her chunk için canonical `author` + `author_role` metadata'sı otomatik üretir.

**Mimari (3 aşamalı):**

```
Docling atoms_data
  ↓ tag_atoms(atoms, EXTRACTORS[doc_type], initial_author=doc.author)
tagged_atoms (her atom: author + author_role + segment_index + is_continuation)
  ↓ segment_aware_pack (soft split-on-author-change)
chunks (author, author_role, authors_in_chunk, segment_indices, starts_mid_segment)
  ↓ [opsiyonel] LlmAuthorValidator (author=None chunk'lar)
  ↓ embed + chroma.upsert
```

**Tip → Extractor eşlemesi** (`src/common/parsing/extractors/__init__.py`):

| document_type | Extractor | Pattern |
|---|---|---|
| `tutanak` | TutanakAuthorExtractor | `BAŞKAN —`, `AD SOYAD (İl) —`, `BAKAN AD —` |
| `onerge` / `kanun_teklifi` | OnergeAuthorExtractor | `... Milletvekili X ve N arkadaşı`, `(İmza: ...)` |
| `press_clip` / `gazete` | GazeteAuthorExtractor | `Yazan: X`, `X — Hürriyet` |
| `pdf_report` | NoopAuthorExtractor | DocumentInput.author miras |
| Bilinmeyen tip | NoopAuthorExtractor | Otomatik fallback |

**Yeni chunk metadata alanları** (ChromaDB filter-edilebilir):
- `author` — canonical (chunk'taki char-ağırlıklı birincil yazar)
- `author_role` — rol
- `authors_in_chunk: list[str]` — chunk'ı kapsayan tüm yazarlar
- `segment_indices: list[int]` — chunk'ı kapsayan turn/segment aralığı
- `starts_mid_segment: bool` — chunk yazar değişiminin ortasından başlıyor mu

**LLM Backstop (opsiyonel)**: Regex pattern başarısız olduğunda (OCR gürültüsü, edge case) local Ollama LLM ile yedek çıkarım. Default kapalı.

```bash
# Tüm tipler için LLM backstop aktif
AUTHOR_VALIDATOR_ENABLED=1 python -m src.trainer.ingestion.ingest --request ingest.json
```

Sadece `author is None` chunk'lar için çalışır. Prompt'lar tip bazlı: `settings.AUTHOR_VALIDATOR_PROMPTS[doc_type]`.

**Query filter örneği**:
```python
# Belirli yazarın chunk'larını çek
collection.query(
    query_texts=["Kardak krizi"],
    where={"author": "Deniz Baykal"},
    n_results=10,
)
```

**Yeni tip eklemek**: `src/common/parsing/extractors/yeni_tip.py` yaz, `EXTRACTORS` registry'sine ekle. Schema değişmez, canonical `author` alanı tüm tipler için tek.

---

### OCR Yapılandırması

#### Belge başına OCR kontrolü

Dijital doğumlu PDF'lerde (metin katmanı olan) OCR'ı devre dışı bırakın — daha hızlı, aynı kalite:

```json
{
  "document_id": "rapor-2024-001",
  "document_type": "tutanak",
  "document_source": "rapor.pdf",
  "ocr": false
}
```

`ocr` alanı belirtilmezse varsayılan `true` — OCR her zaman çalışır. Taranmış belgeler için `ocr: true` bırakın veya hiç yazmayın.

#### Global OCR motoru

PDF'ler için OCR motoru `OCR_ENGINE` ortam değişkeni ile değiştirilebilir.

| Motor | Değer | Notlar |
|---|---|---|
| EasyOCR (varsayılan) | `easyocr` | Türkçe modeli ile en iyi kalite, ilk çalıştırmada ~300 MB model indirir |
| Tesseract CLI | `tesseract` | `brew install tesseract tesseract-lang` gerektirir |
| macOS Vision | `mac` | Sıfır kurulum, yalnızca macOS |

```bash
# Motor seçimi (tek seferlik)
OCR_ENGINE=tesseract python -m src.trainer.ingestion.ingest --request ingest.json

# Motorları karşılaştır
python scripts/compare_ocr.py data/ornek.pdf
```

## Dinamik Agentic Filtre Çıkarıcı (Filter Extractor)

Sistem, kullanıcıların Türkçe doğal dilde yazdıkları arama sorgularından metadata filtrelerini otomatik olarak çıkaran ve arama sorgusunu semantik arama için sadeleştiren agentic bir **Filter Extractor** yapısına sahiptir.

### Mimari ve Çalışma Mantığı

1. **Hızlı Ön Kontrol (Fast-Preprocessor / Bypass):** Her arama sorgusunda LLM çağrısı yapmak gecikmeye (latency) neden olacağı için sorgu önce regex ve anahtar kelime eşleşmeleriyle hızlı bir ön kontrolden geçer. Eğer sorgu filtre ipucu (yıl, dönem, birleşim, yazar adı, gazete adı vb.) içermiyorsa LLM çağrısı doğrudan bypass edilir ve normal vektör aramasına geçilir.
2. **LLM ile Yapısal Filtre Çıkarma (Ollama):** Filtre ipucu içeren sorgular, Türkçe dil yapısına ve meclis tutanakları/gazete küpürleri bağlamına optimize edilmiş özel bir sistem yönergesiyle Ollama LLM'e beslenir.
3. **Pydantic Şema Doğrulaması:** LLM çıktısı, `src/common/schemas.py` altında tanımlanan `FilterCriteria` ve `ExtractedFilterResponse` Pydantic modelleriyle sıkı bir şekilde doğrulanır ve tip dönüşümleri (örneğin yıl ve dönem bilgilerinin integer'a cast edilmesi) otomatik gerçekleştirilir.
4. **Veritabanından Bağımsız Filtre Çevirisi (Decoupled Translation):** Sistem, satıcı kilidini (vendor lock-in) önlemek ve ileride veri tabanının değiştirilmesini (örn. Qdrant, Milvus, pgvector) kolaylaştırmak için modüler bir çevirici (translator) mimarisine sahiptir. Filtrelerin veri tabanına özgü sorgulara dönüştürülmesi `src/common/filter_translators.py` altındaki sınıflar tarafından üstlenilir:
   - `BaseFilterTranslator`: Tüm veri tabanı çeviricileri için soyut temel sınıf.
   - `ChromaFilterTranslator`: Filtreleri ChromaDB'nin `$eq` ve `$and` içeren `where` koşullarına dönüştüren sınıf.

### Çıkarılan Filtre Şeması (Pydantic)

Çıkarılan filtre kriterleri aşağıdaki alanları destekler:

- `year` (int): Belgenin yılı (örn. `1996`).
- `author` (str): Belirli bir yazar veya konuşmacının adı (örn. `Deniz Baykal`).
- `author_role` (str): Konuşmacının rolü veya unvanı (örn. `milletvekili`, `başbakan`).
- `source_name` (str): Belgenin yayınlandığı kaynak/gazete adı (örn. `Hürriyet`).
- `period` (int): TBMM yasama dönemi (örn. `20`).
- `session` (int): TBMM birleşim numarası (örn. `7`).
- `document_type` (tutanak | press_clip | pdf_report | kanun_teklifi): Belgenin tipi.

### Kullanım ve Entegrasyon

#### 1. RAG Katmanında Kullanım

Filter Extractor, `RAGService.retrieve()` içerisinde varsayılan olarak aktiftir. Bir sorgu geldiğinde filtreleri otomatik olarak ayıklar ve `VectorRetriever.retrieve()` metoduna `where_filter` parametresi olarak aktarır.

```python
from src.generator.filter_extractor import FilterExtractor

fe = FilterExtractor()

# 1. Filtreleri doğal dilden çıkarma
extracted = fe.extract("Deniz Baykal'ın 1996 yılındaki Ege adaları hakkındaki konuşmaları")
print(extracted.refined_query) 
# Çıktı: "Ege adaları hakkındaki konuşmaları" (Arama için temizlenmiş sorgu)

print(extracted.filters)
# Çıktı: FilterCriteria(year=1996, author='Deniz Baykal', document_type='tutanak', ...)

# 2. ChromaDB filtre biçimine çevirme
chroma_filter = fe.to_chroma_filter(extracted.filters)
print(chroma_filter)
# Çıktı: {'$and': [{'year': {'$eq': 1996}}, {'author': {'$eq': 'Deniz Baykal'}}, {'document_type': {'$eq': 'tutanak'}}]}
```

#### 2. Vector Explorer ile Görsel Test

Geliştirilen filtreleme mekanizması, `scripts/vector_explorer.py` üzerindeki Streamlit arayüzüne tam entegredir.
- Vektör keşfi yaparken **"⚙️ Auto Filter Extractor Aktif"** seçeneğini işaretleyerek, yazdığınız doğal dil sorgularından hangi filtrelerin çıkarıldığını ve Chroma DB formatına nasıl dönüştürüldüğünü görsel olarak anlık takip edebilirsiniz.

#### 3. Farklı Bir Veritabanına Geçiş Yapma

İleride ChromaDB yerine başka bir veri tabanı kullanmak istediğinizde, tek yapmanız gereken `src/common/filter_translators.py` dosyasında `BaseFilterTranslator` sınıfından türeyen yeni bir sınıf tanımlamaktır:

```python
from src.common.filter_translators import BaseFilterTranslator
from src.common.schemas import FilterCriteria

class QdrantFilterTranslator(BaseFilterTranslator):
    def translate(self, filters: FilterCriteria):
        # Qdrant uyumlu filtre nesnesini burada kurgulayın
        ...
```

---

## Agentic Pipeline (Planning Agent)

Standart RAG akışına ek olarak, sorguyu çok aşamalı bir **Planning Agent** üzerinden
çalıştıran agentic bir pipeline mevcuttur. Ajan, sorguyu analiz edip arama planı üretir,
birden fazla koleksiyonda arama yapar, sonuç yetersizse otomatik genişletilmiş arama
(re-retrieval) tetikler, yanıtı üretir ve bir doğrulayıcı (sanitizer) ile kontrol eder.

### Akış (4 Faz)

```
Sorgu
  ↓ FAZ 1: Planlama       (planner LLM → SearchPlan: intent, koleksiyonlar, query_drafts, filtreler)
  ↓ FAZ 2: Retrieval      (her draft için VectorSearch + reranker, koleksiyon başına)
  ↓ FAZ 2b: Re-retrieval  (sonuç < eşik ise filtreleri gevşeterek tekrar ara)
  ↓ FAZ 3: Answering      (answering LLM → bağlamdan yanıt üretir)
  ↓ FAZ 4: Validation     (sanitizer LLM → kriter kontrolü; gerekirse düzeltme denemesi)
  ↓ FAZ 4b: Quality Re-retrieval  (yanıt yetersizse gap-fill arama → tekrar yanıt)
AgentOutput (answer, thinking, plan, validation, trace, sources, re_retrieved, quality_re_retrieved)
```

### Çalıştırma

```bash
# Etkileşimli sohbet — agent modu
python chat.py --agent --pipeline pipeline.yaml

# Tek seferlik test harness (app.py)
python app.py --agent --pipeline pipeline.yaml
```

`--pipeline` verilmezse proje kökündeki `pipeline.yaml` kullanılır. Dosya yoksa
`RAGService.run_agent()` bir `RuntimeError` fırlatır.

### Yapılandırma

Sistem üç konfigürasyon dosyası kullanır; her birinin sorumluluğu ayrıdır:

| Dosya | Sorumluluk | Kullanan |
|---|---|---|
| `models.yaml` | Embedding model özellikleri + collection registry | `src/config/collections.py` (import-time) |
| `pipeline.yaml` | Agent deployment block'ları, LLM model atamaları, retrieval parametreleri | `src/agent/pipeline_loader.py` → agent sistemi |
| `src/config/settings.py` | Ortam değişkeni override'ları, legacy defaults, filter extraction | Eski `RAGService`, `FilterExtractor`, ingestion script'leri |

`pipeline.yaml` koleksiyon **tanımı içermez** — koleksiyonlar yalnızca `models.yaml`'da
yaşar. Planner, mevcut koleksiyon kataloğunu (`get_collection_catalog()`) prompt'una alır.

---

#### `src/config/settings.py` — Ortam Değişkenleri

Ortam değişkenleri ile runtime'da override edilebilir:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `RAG_ENV` | `local` | `local` veya `remote` — deployment seçimi |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API adresi |
| `RAG_LLM_MODEL` | `gemma4:latest` | Eski `RAGService` için; **agent pipeline kullanmaz** |
| `RAG_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Ingestion script fallback; aktif kullanım `models.yaml`'da |
| `RAG_FILTER_LLM_MODEL` | `gemma4:e2b` | `FilterExtractor` için — agentic pipeline dışında |
| `REMOTE_OLLAMA_HOST` | — | `RAG_ENV=remote` olduğunda Ollama host adresi |
| `RAG_DEFAULT_COLLECTION` | `tbmm_tutanaklar_nomic_v2` | `RAGService` varsayılan collection'ı |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` veya `vector` |
| `USE_RERANKER` | `1` | `1` = aktif, `0` = devre dışı |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `pipeline.yaml`'da tanımlı değilse fallback |
| `DEBUG_RAG` | `0` | `1` = verbose logging |

```bash
# Örnek: uzak sunucu, debug açık
RAG_ENV=remote REMOTE_OLLAMA_HOST=http://172.20.0.143:11434 DEBUG_RAG=1 python chat.py
```

---

#### `models.yaml` — Embedding & Collection Registry

**`model_specs`** bloğu: her embedding modelinin kapasitesini tanımlar.

```yaml
model_specs:
  jinaai/jina-embeddings-v3:
    max_context_tokens: 8192
    overlap_tokens: 128
    embed_dim: 1024
    supports_late_chunking: true
    description: "Jina v3: 8K context, late chunking destekler."
```

**`collections`** bloğu: collection registry — Python değişikliği gerekmez, sadece YAML düzenlemesi yeterli.

```yaml
collections:
  tbmm_tutanaklar_nomic_v2:
    collection_name: tbmm_tbmm_tutanaklar_nomic_v2
    chroma_path: data_lake/parliament_digital_born_minutes_vectors
    sqlite_path: data_lake/parliament_digital_born_minutes.db
    embed_model: nomic-embed-text-v2-moe
    doc_type: tutanak
    min_chunk_chars: 400
    max_chunk_chars: 1500
    max_chunk_tokens: 512
    min_chunk_tokens: 384
```

**`defaults`** bloğu: her `doc_type` için varsayılan collection key'i.

```yaml
defaults:
  gazete: gazete_arsivi_jina_v3
  tutanak: tbmm_tutanaklar_nomic_v2
  onerge: tbmm_onerge_docling_jina_v3
```

---

#### `pipeline.yaml` — Agent & Deployment Config

**`deployment_blocks`** her rolü ayrı bir makine/modele yönlendirir:

```yaml
deployment_blocks:
  fast-01:   # planlama + doğrulama (küçük/hızlı model)
    host: http://localhost:11434
    models: {planner: qwen3.5:9b, sanitizer: qwen3.5:9b}
    timeout_seconds: 15
  gpu-01:    # yanıt üretimi
    models: {answer: gemma4:latest}
    max_num_ctx: 32768
    max_num_predict: 4096
```

> **Not:** `models` altındaki tag'ler Ollama'da çekili olmalı (`ollama pull <model>`).
> Çekili değilse runtime'da "model not found" alırsınız.

`retrieval` bloğundaki `reranker.model` değeri, `settings.py`'deki `RERANK_MODEL` env var'ının
önünde gelir — YAML tanımlıysa settings fallback devreye girmez.

### Trace Çıktısı

Her sorgu için faz bazlı gecikme ve metadata `PipelineTracer` ile konsola basılır:

```
[PIPELINE] 2026-05-21 15:09:00 | trace_id=a1b2c3d4e5f6
[PIPELINE] ┌─ PHASE 1: Planning
[PIPELINE] │  block: fast-01 | model: qwen3.5:9b | latency: 1.2s
[PIPELINE] │  intent: temporal | resources: tbmm_minutes
[PIPELINE] ├─ PHASE 2: Retrieval
[PIPELINE] │  tbmm_minutes: 5 results | latency: 0.8s
[PIPELINE] ├─ PHASE 3: Answering
[PIPELINE] │  block: gpu-01 | model: gemma4:latest | context: 8200 chars | latency: 6.4s
[PIPELINE] ├─ PHASE 4: Validation
[PIPELINE] │  sanitizer: PASS | checks: [addresses_query✓, backed_by_sources✓, ...]
[PIPELINE] └─ TOTAL: 8.4s
```

### Re-retrieval, Fallback ve Sanitizer

- **Re-retrieval (miktar)**: Toplam sonuç `re_retrieval.trigger_min_results` altındaysa, planner
  filtreleri gevşeterek (yazar düşür → yıl düşür → semantik) yeni bir plan üretir.
- **Re-retrieval (kalite)**: Yanıt `addresses_query` kontrolünü geçemezse veya "bulunamadı /
  kaynaklarda yer almıyor" gibi Türkçe kalıplar içeriyorsa, `on_quality_failure: true` ile
  hedefli bir gap-fill araması tetiklenir. Validation sorunları ve yetersiz yanıt planner'a
  iletilir; model eksik bilgiyi spesifik sorgularla arar.
- **Fallback**: Planner LLM tamamen başarısız olursa, `agent.planner.fallback` altındaki
  varsayılan koleksiyon/sorgu kullanılır.
- **Sanitizer**: Yanıtı `validation_criteria` kriterlerine göre kontrol eder; başarısızsa
  `max_retries` kadar düzeltme dener. LLM hatasında fail-open davranır (yanıt geçer).

### İlgili Modüller

| Modül | Sorumluluk |
|---|---|
| `src/agent/planner.py` | `PlanningAgent` — tüm akışı orkestre eder |
| `src/agent/tools.py` | `SearchTool`, `ContextBuilderTool`, `AnswerTool` |
| `src/agent/sanitizer.py` | `SanitizerAgent` — doğrulama + düzeltme |
| `src/agent/tracer.py` | `PipelineTracer` — faz bazlı gözlemlenebilirlik |
| `src/agent/schemas.py` | Pydantic kontratlar (`SearchPlan`, `AgentOutput`, ...) |
| `src/config/pipeline_loader.py` | `pipeline.yaml` → `PipelineConfig` |
| `src/common/llm_client_pool.py` | `LLMClientPool` — blok başına Ollama client (retry/timeout) |

---

### Tool Mimarisi

`src/agent/tools.py` dosyasındaki tool'lar, `PlanningAgent` tarafından doğrudan örneklenerek kullanılır.
Bir "tool registry" ya da dinamik dispatch mekanizması yoktur — her tool `planner.py`'nin `__init__`
metodunda sabit olarak bağlanır:

```python
# src/agent/planner.py — __init__
self._search_tool   = SearchTool(config, client_pool)   # FAZ 2: vektör arama
self._context_tool  = ContextBuilderTool(config)         # FAZ 3: context birleştirme
self._answer_tool   = AnswerTool(client_pool, config)    # FAZ 3: yanıt üretimi
self._sanitizer     = SanitizerAgent(client_pool, config) # FAZ 4: doğrulama
```

**Tool'lar nasıl seçiliyor?** Planner LLM, hangi koleksiyona bakılacağını (SearchPlan) üretir;
`SearchTool` bu plana göre `VectorSearch` örneği oluşturur. Tool seçimi dinamik değil,
faz sırasına göre deterministiktir.

#### Mevcut Tool'lar

| Tool Sınıfı | Faz | Görev |
|---|---|---|
| `SearchTool` | FAZ 2 | Planner'ın ürettiği `SearchPlan`'daki her `query_draft` için `VectorSearch` + reranker çalıştırır. Collection başına `VectorSearch` instance'ı önbelleğe alır. |
| `ContextBuilderTool` | FAZ 3 | Birden fazla collection'dan gelen sonuçları birleştirir, `distance_threshold`'a göre filtreler, `max_chars`/`total_max_chars` limitlerini uygular. |
| `AnswerTool` | FAZ 3 | Answering block'undaki LLM'i stream modunda çağırır, `thinking` + `content` çiftini döner. `mufettis_mode` ile farklı system prompt kullanılabilir. |

#### Yeni Tool Ekleme

1. **`src/agent/tools.py`'ye sınıf yaz:**

```python
class MyCustomTool:
    """Açıklama."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, input: str) -> str:
        # implementasyon
        ...
```

2. **`planner.py`'nin `__init__`'ine ekle:**

```python
from src.agent.tools import MyCustomTool

# __init__ içinde:
self._my_tool = MyCustomTool(config)
```

3. **`run()` metodundaki ilgili faza çağrıyı ekle:**

```python
# Örnek — FAZ 3 sonrasına yeni bir adım:
my_result = self._my_tool.run(answer)
```

4. **Gerekiyorsa `AgentOutput` şemasına alan ekle** (`src/agent/schemas.py`).

> **Not:** Tool'ların `pipeline.yaml` ile entegrasyonu için `PipelineConfig`'ten parametre
> okuyabilirsiniz (`self._config.sanitizer`, `self._config.retrieval` vb.). Yeni bir config
> bloğu gerekiyorsa `pipeline_loader.py`'ye yeni bir `AgentConfig` veya özel bir sınıf ekleyin.

---

## Sohbet Komutları

| Komut | Açıklama |
|---|---|
| `/yardim` | Yardım menüsünü göster |
| `/debug` | Debug görünümünü aç/kapat |
| `/kaynak N` | N. kaynağın tam veritabanı kaydını göster |
| `/müfettiş soru` | Derin araştırma modu (sorgu genişletme + geniş bağlam) |
| `/temizle` | Ekranı temizle |
| `/cikis` | Çık |

---

## Agentic Orkestratör (Orchestrator)

Yeni RAG akışı: `Planner → Policy → Allocator → Retrieve → Assembler → EvidenceJudge → (Expand → Re-Judge) → Answer → Sanitizer → Citation`. Eski `re_retrieval` ve `quality_re_retrieval` döngülerinin yerini cevap öncesi kanıt-yeterlilik kontrolü (`EvidenceJudge`) ve yedek tampondan beslenen genişletme (`ExpansionPlanner`) alır. Sanitizer (metin doğrulayıcı) post-hoc çalışmaya devam eder.

**Tasarım:** [`docs/superpowers/specs/2026-05-24-agentic-orchestrator-design.md`](docs/superpowers/specs/2026-05-24-agentic-orchestrator-design.md)

### Aktivasyon

`pipeline.yaml` içinde feature flag'i çevir:

```yaml
orchestrator:
  enabled: true   # false: eski PlanningAgent / true: yeni OrchestratorAgent
```

Sohbeti yeniden başlat. `RAGService.run_agent()` artık `OrchestratorAgent`'a yönlenir.

### Kullanım

```bash
# Standart agent modu (orchestrator açıkken)
python chat.py --agent

# Özel pipeline dosyası ile
python chat.py --agent --pipeline path/to/custom.yaml
```

Başlangıçta koleksiyon seçim ekranı çıkar — seçim, orkestratörün **Policy** aşamasında planner önerileri ile kesişime alınır. Hiçbir koleksiyon seçilmezse refuse yanıtı döner.

Programatik kullanım:

```python
from src.generator.service import RAGService

svc = RAGService(pipeline_config_path="pipeline.yaml")
out = svc.run_agent(
    query="2023 yılında meclis ne konuştu?",
    session_collections=["tbmm_minutes", "gazete_arsivi"],
)

print(out.answer)
print(out.evidence_decision.action)      # answer / expand / clarify / refuse
print(out.evidence_decision.judge_type)  # heuristic / llm
print(out.expanded)                       # reserve tüketildi mi
for s in out.sources:
    print(s["index"], s["collection_name"], s["chunk_id"])
```

### `pipeline.yaml` Ayarları

#### 1. Flag

```yaml
orchestrator:
  enabled: true        # ana anahtar
```

#### 2. Policy (Koleksiyon erişimi)

```yaml
policy:
  mode: session_intersection   # planner önerileri ∩ kullanıcı seçimi
```

Tek mod var. Auth/rol bazlı modlar henüz uygulanmadı (tüm koleksiyonlar public).

#### 3. Allocation (Koleksiyon başına bütçe)

```yaml
allocation:
  defaults:    { primary: 2, reserve: 2, fetch_k: 10 }
  by_query_type:
    fact:       { primary: 2, reserve: 2, fetch_k: 10 }
    summary:    { primary: 3, reserve: 2, fetch_k: 10 }
    comparison: { primary: 3, reserve: 2, fetch_k: 12 }
    reasoning:  { primary: 3, reserve: 2, fetch_k: 12 }
    policy:     { primary: 3, reserve: 3, fetch_k: 15 }
  max_per_document: 1     # koleksiyonlar arası doküman tekilleştirme
  max_total_primary: 12   # toplam primary slot tavanı
```

- `primary`: koleksiyondan context'e konulan chunk sayısı
- `reserve`: yedekte tutulan; judge `expand` derse devreye girer
- `fetch_k`: vector top-N (≥ primary + reserve olmalı)
- `max_per_document`: aynı dokümandan kaç chunk alınır
- `max_total_primary`: LLM'e gitmeden önceki sert tavan

**Not:** Planner LLM `query_type` üretmiyor henüz — varsayılan `fact`. Şimdilik `defaults` satırını ayarla; per-tip satırları planner prompt güncellenince devreye girer.

#### 4. EvidenceJudge (Kanıt yeterliliği)

```yaml
judge:
  mode: hybrid
  heuristic:
    min_chunks: 4                    # ≥ N chunk → answer
    min_collection_coverage: 2       # ≥ N farklı koleksiyon → answer
    min_rerank_score: 0.0
  llm:
    enabled: true
    block: fast-01                   # deployment bloğu
    model_key: judge                 # fast-01.models.judge → gerçek model
    borderline_band: [2, 4]          # bu chunk aralığında LLM judge devreye girer
    max_borderline_score_floor: 0.35
    timeout_seconds: 5               # LLM timeout → heuristik fallback
  max_expand_iterations: 1           # expand→re-judge döngü sayısı
  on_low_confidence: expand
```

Heuristik açık durumları cezalandırmadan geçirir (~50ms). Sınır vakalar (örn. tek koleksiyondan 2-4 chunk) LLM judge'a düşer. LLM hatası → heuristik `expand`.

#### 5. Judge LLM modeli

Deployment bloğuna model anahtarı eklenmiş olmalı (47e52dc commit'inden sonra hazır):

```yaml
deployment_blocks:
  fast-01:
    models:
      planner: qwen3.5:9b
      sanitizer: qwen3.5:9b
      judge: qwen3.5:9b      # ← LLM judge için zorunlu
```

### Ayar Rehberi

| Belirti | Çözüm |
|---|---|
| Cevap sığ, context az | ↑ `allocation.defaults.primary` (2→3 veya 4) |
| Aynı kaynaktan tekrar eden chunk'lar | `max_per_document: 1` (zaten varsayılan) |
| Judge sürekli `expand` diyor | ↓ `judge.heuristic.min_chunks` veya ↓ `min_collection_coverage` |
| Dar ama geçerli sorularda `clarify` | ↓ `min_chunks: 2` |
| LLM judge yavaş | `judge.llm.enabled: false` (sadece heuristik) |
| Expand çalışıyor ama chunk eklenmiyor | ↑ `fetch_k` (primary+reserve'den büyük olmalı) |
| Context LLM için fazla büyük | ↓ `allocation.max_total_primary` |

### Trace ve Debug

Sohbette `/debug` aç. Orchestrator her aşama için trace event üretir:
`planning`, `policy`, `allocation`, `retrieval`, `assembly`, `judge`, `expansion`, `judge_post_expand`, `answering`, `validation`, `citation`.

Her event: `phase`, `latency_ms`, `details` (aşamaya özgü payload).

### Geri Alma

```yaml
orchestrator:
  enabled: false
```

Sohbeti yeniden başlat. Eski `PlanningAgent.run()` aktifleşir — kod değişikliği yok.

### Bilinen Eksikler (engelleyici değil)

- Planner LLM `query_type` üretmiyor → her zaman `fact`. Şimdilik `allocation.defaults` üzerinden tune et; planner prompt güncellemesi follow-up'ta.
- `stream_callback` parametresi var ama token-by-token streaming değil — cevap tamamlanınca tek seferde fire eder.
- Guardrails (input injection, citation grounding, off-topic refuse, toxicity) ayrı spec'te (`docs/superpowers/specs/2026-05-24-guardrails-design.md`), henüz uygulanmadı.

---

## MCP Sunucuları

```bash
# Seçenek 1 — 3'ünü aynı anda (arka planda)
python -m src.mcp.press_server &
python -m src.mcp.minutes_server &
python -m src.mcp.router_server &

# Seçenek 2 — Ayrı terminaller
python -m src.mcp.press_server
python -m src.mcp.minutes_server
python -m src.mcp.router_server
```

| Sunucu | Adres | Açıklama |
|---|---|---|
| mcp-press | http://localhost:8001/docs | Gazete arşivi |
| mcp-minutes | http://localhost:8002/docs | TBMM tutanakları |
| mcp-router | http://localhost:8003/docs | Her iki kaynak (çapraz sorgular) |

> **MCP mimarisi için:** [`docs/mcp_architecture.md`](docs/mcp_architecture.md)

---

## Retrieval Yapılandırması

### 3 Aşamalı Pipeline

```
Bi-encoder → ChromaDB ANN (fetch_k=100)
    + BM25 FTS (hybrid modda)
    → RRF füzyonu → ~100 aday
    → Cross-encoder reranker → top 20
    → SQLite tam metin çekimi (yalnızca 5 sonuç için)
    → LLM'e top 5
```

### Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`: BM25 + vektör RRF füzyonu \| `vector`: yalnızca ChromaDB ANN |
| `USE_RERANKER` | `1` | `1`: cross-encoder aktif \| `0`: devre dışı (yalnızca RRF) |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | HuggingFace cross-encoder modeli (~120MB, çok dilli) |
| `RERANK_FETCH_K` | `100` | Cross-encoder'a beslenen aday sayısı (artırırsan recall yükselir, yavaşlar) |
| `RERANK_COARSE_K` | `20` | Cross-encoder'dan geçen ara havuz boyutu |
| `RERANK_FINAL_K` | `5` | LLM bağlamına giden nihai sonuç sayısı |

### Mod Karşılaştırması

```bash
python -m scripts.evaluate --quick
RETRIEVAL_MODE=vector python -m scripts.evaluate --quick
USE_RERANKER=0 python -m scripts.evaluate --quick
```

---

## Değerlendirme

```bash
# Tüm metriklerle değerlendirme
python -m scripts.evaluate

# Hızlı (sadece retrieval metrikleri)
python -m scripts.evaluate --no-judge

# İndeks sağlık kontrolü
python -c "from src.evaluator.index_health import run_all_checks; import pprint; pprint.pprint(run_all_checks())"
```

---

## A/B Benchmark (Koleksiyon Karşılaştırma)

```bash
# Örnek YAML deney konfigürasyonu
# experiments/jina_v3_vs_nomic.yaml
#   collections: [tbmm_minutes_docling_jina_v3, tbmm_minutes]
#   fixture: tests/fixtures/eval_queries_docling_d20.json

python -m scripts.run_benchmark --config experiments/jina_v3_vs_nomic.yaml
```

### Token-Overlap Değerlendirme (Chroma Yöntemi)

Chunk ID bağımsız, verbatim alıntı tabanlı metrikler: token recall, precision, IoU.
Farklı chunking stratejilerini aynı ground truth üzerinde adil karşılaştırmaya imkan tanır.

**1. Golden Dataset (Altın Veri) Hazırlama**

Elinizdeki duruma göre iki farklı script kullanabilirsiniz:

**A. Sıfırdan Sentetik Veri Üretmek:** Hiç test sorunuz yoksa, rastgele metinlerden soru ve cevap (excerpt) üretin.
```bash
python scripts/generate_golden.py \
    --collection tbmm_minutes_docling_jina_v3 \
    --n 20 \
    --output tests/fixtures/golden_minutes_auto.json

# Sadece ekranda görmek için (dosyaya yazmaz):
python scripts/generate_golden.py --collection tbmm_minutes_docling_jina_v3 --n 3 --dry-run
```

**B. Mevcut Sorulara Cevap Eklemek:** Zaten sorularınız varsa (fixture), LLM ile otomatik excerpt ekleyin.
```bash
python scripts/generate_excerpts.py \
    --fixture tests/fixtures/eval_queries_docling_d20.json \
    --collection tbmm_minutes_docling_jina_v3 \
    --output tests/fixtures/eval_queries_docling_d20_excerpts.json

# Sadece ilk 2 sorguyu görmek için (dosyaya yazmaz):
python scripts/generate_excerpts.py \
    --fixture tests/fixtures/eval_queries_docling_d20.json \
    --collection tbmm_minutes_docling_jina_v3 \
    --dry-run
```

**2. Benchmark'ı çalıştır** (YAML'daki fixture'ı oluşturduğunuz dosyaya güncelleyin):

```bash
python -m scripts.run_benchmark --config experiments/jina_v3_vs_nomic.yaml
```

`excerpts` alanı fixture'a manuel de eklenebilir:

```json
{
  "id": "d20-kardak-001",
  "query": "Kardak krizinde Baykal ne açıkladı?",
  "excerpts": [
    "Dışişleri Bakanı Deniz Baykal, Yunanistan'ın Kardak Kayalıklarına ilişkin tutumunu mecliste açıkladı."
  ]
}
```

Matcher önceliği: `excerpts` → `gold_evidence_spans` → `relevant_chunk_ids` → `relevant_kayit_nos`

---

## Vektör Veritabanı Görselleştirme

ChromaDB içindeki chunk'ları, meta verileri ve vektör arama sonuçlarını canlı olarak incelemek için özel Streamlit panelini kullanabilirsiniz:

```bash
# Gerekli kütüphaneyi kurun
pip install streamlit

# Paneli başlatın
streamlit run scripts/vector_explorer.py
```

Bu panel üzerinden:
*   Koleksiyonlar arasında (Jina v3 vs Nomic v2) hızlıca geçiş yapabilir,
*   Herhangi bir arama terimiyle vektör arama testi yapabilir,
*   Veritabanındaki ilk 50 kaydı tablo halinde inceleyebilirsiniz.

---

## Teknoloji Yığını

| Bileşen | Teknoloji |
|---|---|
| Embedding | `nomic-embed-text-v2-moe` (Ollama) / `jinaai/jina-embeddings-v3/v4` (HuggingFace) |
| LLM | `gemma4:latest` (Ollama) |
| Vektör arama | ChromaDB (embedded, PersistentClient) |
| Tam metin arama | SQLite FTS5 + BM25 |
| Hibrit füzyon | Reciprocal Rank Fusion (RRF, k=60) |
| Reranker | Cross-encoder (`mmarco-mMiniLMv2`, çok dilli) |
| Arayüz | Rich (terminal) |
| Belge ayrıştırma | Docling (PDF/DOCX) |

---

## Proje Yapısı

```
src/
├── config/          # Sabitler, path'ler, CollectionSpec / MODEL_SPECS kayıt defteri
├── common/          # Paylaşılan yardımcılar — detay: src/common/README.md
│   └── parsing/
│       ├── docling_manager.py    # PDF/DOCX → yapısal parçalar
│       ├── author_extractor.py   # State machine: atom→author propagation
│       ├── segment_pack.py       # Author-aware packer
│       ├── llm_author_validator.py  # Opsiyonel LLM backstop
│       └── extractors/           # Per-type pattern'ler (tutanak, gazete, onerge)
├── trainer/         # Veri yükleme ve indeksleme
│   └── ingestion/
│       ├── adapters/             # DocumentAdapter'lar (tutanak, press_clip, pdf_report)
│       ├── pipeline.py           # IngestionPipeline (manifest + dedup)
│       ├── manifest.py           # SQLite manifest (document_manifest.db)
│       ├── ingest.py             # CLI entry point
│       └── embedder.py           # LocalLateChunkingEmbedder (Jina v3/v4)
├── retriever/       # Hibrit arama (BM25 + vektör + RRF)
├── generator/       # LLM entegrasyonu ve RAGService
├── evaluator/       # Metrikler, LLM judge, latency, benchmark
└── ui/              # Terminal arayüzü

schemas/
└── ingest_request.schema.json     # JSON Schema doğrulama

experiments/         # YAML A/B deney konfigürasyonları
playground/          # Geçici scriptler — DB inceleme, tek seferlik debug, denemeler (production'a girmez)
scripts/             # CLI giriş noktaları (bazıları deprecated)
tests/               # Birim testler + değerlendirme sorgu seti
docs/                # Mimari dokümantasyon
data_lake/
├── document_manifest.db           # İndeksleme durum takibi
├── press_clips.db
├── parliament_digital_born_minutes.db
└── ...vectors/                    # ChromaDB koleksiyonları
```

---

## Sık Karşılaşılan Sorunlar

**`ImportError: This modeling file requires the following packages that were not found in your environment: peft`**
→ Jina v4 modeli `peft` paketini gerektirir. `pip install peft` ile kurun.
→ `requirements.txt`'e eklendi — mevcut venv için `pip install -r requirements.txt` yeterli.

**"Model 'xxx' MODEL_SPECS'te tanımlı değil"**
→ `src/config/collections.py` içindeki `MODEL_SPECS` sözlüğüne model özelliklerini ekleyin.

**"document_source bulunamadı"**
→ `ingest_request.json` içindeki `document_source` alanı doğru path'i veya URL'yi gösteriyor mu?
→ Yerel dosya ise path doğru mu? URL ise `--validate` ile erişilebilirliğini kontrol edin.

**"Bilinmeyen document_type"**
→ `document_type` şu değerlerden biri olmalı: `tutanak`, `press_clip`, `pdf_report`, `onerge`, `kanun_teklifi`.
Yeni tip eklemek için `src/trainer/ingestion/adapters/` altına adapter yazın ve `__init__.py`'ye kaydedin.
Chunk-düzeyi yazar çıkarımı için `src/common/parsing/extractors/` altına extractor ekleyin (opsiyonel — yoksa NoopAuthorExtractor fallback).

**"Aynı belge tekrar işleniyor"**
→ `document_id` değişiyor mu? ID deterministik ve sabit olmalı.
→ `content_hash` hesaplanıyor mu? Dosya değiştiyse hash değişir ve reprocess tetiklenir.

**Query-time ile index-time model uyuşmazlığı**
→ Her model için ayrı koleksiyon kullanın. Aynı koleksiyona farklı modelle yazmak
vektör uzaylarının uyuşmamasına neden olur (sessizce kötü sonuçlar).
