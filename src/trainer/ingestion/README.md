# Modern Veri Yükleme Boru Hattı (Ingestion)

Bu klasör, projenin en güncel ve gelişmiş veri yükleme (ingestion) mimarisini içerir. Eski yöntemlerden farklı olarak, döküman yapısını anlayan (structure-aware) ve bağlamsal vektörler (late chunking) üreten bir akış kullanır.

## Temel Özellikler

*   **Yapısal Analiz**: `IBM Docling` kütüphanesi kullanılarak dökümanlardaki hiyerarşi (başlıklar, listeler, tablolar) korunarak metin ayıklanır.
*   **Bağlamsal Embedding (Late Chunking)**: Jina v3 modeli kullanılarak, metin parçalara bölünmeden önce tüm dokümanın bağlamı öğrenilir. Bu sayede her bir "chunk" kendi başına değil, tüm döküman içindeki anlamıyla vektörleştirilir.
*   **Otomatik Metadata**: Dosya yollarından (`D20/Y1/B1...`) otomatik olarak Dönem, Yasama Yılı, Birleşim ve Tarih gibi meta verileri ayıklar.
*   **Toplu İşleme**: Belirlenen bir dizindeki tüm PDF, DOCX ve TXT dosyalarını tarayıp otomatik olarak sisteme dahil edebilir.

## Bileşenler

*   **`pipeline.py`**: Uçtan uca akışı yöneten ana sınıftır (`IngestionPipeline`). Docling ile ayıklama, Late Chunking ile embedding üretimi ve ChromaDB'ye kayıt adımlarını orkestre eder.
*   **`ingest.py`**: CLI giriş noktası — `python -m src.trainer.ingestion.ingest` ile çağrılır. Manifest validation, diff, status, koleksiyon ekleme komutlarını sağlar.
*   **`manifest.py`**: SQLite tabanlı dedup & değişiklik takibi (`MANIFEST_DB`).
*   **`adapters/`**: Belge türü başına parser (`tutanak_pdf`, `press_clip`, `pdf_report`, `kanun_teklifi`).
*   **`embedder.py`**: `LocalLateChunkingEmbedder` (Jina v3/v4) — span'lere göre context-aware embedding üretir.
*   **`downloader.py`**: URL/path resolution; ETag-aware cache.

## Çalışma Akışı

1.  **Döküman Dönüştürme**: `DoclingManager` ile döküman Markdown formatına çevrilir ve yapısal parçalara ayrılır.
2.  **Paketleme (Packing)**: Çok kısa olan yapısal parçalar, semantik bütünlüğü bozmadan birleştirilir.
3.  **Vektörleştirme**: Tüm metin üzerinden "Late Chunking" yöntemiyle bağlamsal vektörler hesaplanır.
4.  **Depolama**: Metinler, vektörler ve zenginleştirilmiş meta veriler ChromaDB'ye kaydedilir.
