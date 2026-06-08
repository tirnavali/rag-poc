# devtools

Geliştirici araçları — production kodu değil. Bu klasördeki scriptler debug, analiz ve bakım amaçlıdır.

Ana projenin virtual environment'ını kullanır; ekstra kurulum gerekmez.

```bash
source .venv/bin/activate
```

---

## Araçlar

### `clear_collection.py` — Collection Temizleme

Bir collection'ı hem ChromaDB vector store'dan hem de SQLite manifest veritabanından siler. Sıfırdan yeniden indexleme yapmadan önce kullanılır.

```bash
python devtools/clear_collection.py <collection_adı>
# Örnek:
python devtools/clear_collection.py tutanaklar_nomic_v2
```

---

### `analyze_sessions.py` — Session Chunk İnceleme

Belirli TBMM tutanak oturumları için vector store'daki chunk içeriklerini stdout'a döker. Retrieval kalitesini manuel olarak doğrulamak için kullanılır.

```bash
python devtools/analyze_sessions.py
```

> Scriptin üstündeki `SESSION_IDS` ve `COLLECTION` sabitleri düzenlenerek farklı oturumlar incelenebilir.

---

### `analyze_packer.py` — Atom Packing Analizi

`data_lake/parse_cache` altındaki `*_atoms.json` dosyalarını okur; greedy packing algoritmasının chunk boyutu dağılımına etkisini istatistiksel olarak raporlar. Chunking stratejisini değerlendirmek için kullanılır.

```bash
python devtools/analyze_packer.py
```

> Cache dizini boşsa çıktı vermeden tamamlanır.

---

## Test Verisi

`test_data/FULL_28_2_1_19200423.json` — TBMM 20. Dönem tutanaklarından örnek bir JSON belgesi. Parser ve chunker scriptlerinde referans girdi olarak kullanılır.

---

## Notlar

- Tüm scriptlerde hardcoded path'ler `src.config.settings` veya `src.config.collections` üzerinden alınmıyor; gerekirse ilgili sabiti script başında düzenleyin.
- Bu araçlar CI pipeline'a dahil değildir; `tests/` altındaki test suite'i kullanın.
- Docling HybridChunker doğrulama scripti `tests/test_hybrid_chunker.py` olarak test klasöründe yer alır.
