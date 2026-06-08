# src/trainer

RAG sisteminin ingestion (veri yükleme) katmanı. Ham belgeleri (PDF, JSON, CSV) okur, Docling ile yapısal parse eder, Late Chunking ile context-aware embedding üretir ve ChromaDB'ye yazar. Manifest tabanlı dedup ile aynı belgenin tekrar işlenmesini engeller.

Tek dış giriş noktası:

```bash
python -m src.trainer.ingestion.ingest [komut]
```

## Alt Paketler

| Paket | İçerik |
|---|---|
| [`ingestion/`](ingestion/README.md) | Docling + Late Chunking pipeline; CLI; manifest; adapter registry |
| [`press_clips/`](press_clips/README.md) | Gazete kupürleri CSV → SQLite → ChromaDB (legacy chunking, late chunking yok) |

## Ingestion Pipeline (adımlar)

```
manifest check  ──►  adapter.parse()  ──►  embed (Late Chunking)  ──►  ChromaDB upsert  ──►  manifest update
   (skip if hash eşleşir)     (Docling/inline)    (Jina v3/v4 span'leri ile)
```

1. **Manifest check** — `content_hash` veritabanındakiyle eşleşirse skip (force flag yoksa).
2. **Adapter.parse()** — belge türüne göre `(full_text, chunks[{text, span, metadata}])` döner.
3. **Embed** — `spec.supports_late_chunking=True` ise span'lerle late chunking; değilse standart Ollama embedding'e fallback.
4. **ChromaDB upsert** — `{document_id}_{chunk_index}` deterministic ID; metadata ChromaDB-uyumlu (str/int/float/bool) sanitize edilir.
5. **Manifest update** — status=done, chunk_count, ETag.

## Adapter Registry

`src/trainer/ingestion/adapters/__init__.py` içinde `document_type → AdapterClass` eşlemesi:

| Document type | Adapter | Kaynak | Parse | Late Chunking |
|---|---|---|---|---|
| `tutanak` | `TutanakPdfAdapter` | PDF (file/URL) | Docling | ✓ |
| `press_clip` | `PressClipAdapter` | `metadata.dokuman_metni` (inline) | RecursiveTextSplitter | ✓ |
| `pdf_report` | `PdfReportAdapter` | PDF (file/URL) | Docling | ✓ |
| `kanun_teklifi` | `KanunTeklifiAdapter` | PDF URL | Docling | ✓ |

Yeni tip eklemek için: `DocumentAdapter`'dan miras al, `parse()` ve `compute_content_hash()` uygula, registry'ye kaydet.

## Manifest

`settings.MANIFEST_DB` (SQLite) tek bağlantı üzerinden:

- `content_hash` — dosya için SHA-256(ilk 1MB + boyut), inline için tam metnin hash'i.
- `status` — `done` / `pending` / `failed`.
- `source_etag`, `source_last_modified` — URL kaynakları için cache invalidation.

Bağlantı tek; process-içi tek-thread varsayımı vardır.

## CLI Komutları

| Komut | Açıklama |
|---|---|
| `--request manifest.json` | Manifest'teki belgeleri ingest et |
| `--validate manifest.json` | Dry-run validasyon (parse yok, sadece schema/path kontrolü) |
| `--diff manifest.json` | Yeni / değişmiş / değişmemiş ayrımı |
| `--list-collections` | `models.yaml`'daki tüm koleksiyonları listele |
| `--list-types` | Desteklenen document_type değerleri |
| `--status` | Manifest özeti (kaç done, kaç pending, kaç failed) |
| `--delete DOCUMENT_ID` | Belgeyi ve chunk'larını sil (manifest + ChromaDB) |
| `--add-collection` | Yeni koleksiyon eklemek için interaktif wizard; `models.yaml`'a yazar |

## Minimal Manifest Örneği

```json
{
  "version": "1.0",
  "collection": "tbmm_minutes",
  "documents": [
    {
      "document_id": "tutanak_d27_y2_b15",
      "document_type": "tutanak",
      "document_source": "/path/to/D27/Y2/B15.pdf",
      "metadata": {
        "period": 27,
        "legislative_year": 2,
        "session": 15
      }
    }
  ]
}
```

```bash
python -m src.trainer.ingestion.ingest --validate manifest.json
python -m src.trainer.ingestion.ingest --request manifest.json
```

Daha fazla örnek için `ornek_tutanak.json`, `ornek_onerge_manifest.json`, `ornek_ingestion.json` dosyalarına bakın.

## Bağımlılıklar

- **Docling** (IBM) — PDF → structured text + spans
- **Jina v3 / v4** (HuggingFace transformers + PyTorch) — Late Chunking embedder
- **ChromaDB** (embedded) — vektör depolama
- **SQLite3** — manifest + press_clips arşivi
- **Ollama** — late chunking olmayan modeller için fallback embedder
- **ruamel.yaml** — `models.yaml` yorumlu YAML manipülasyonu
- **Rich** — terminal UI (wizard, tablo, panel)
