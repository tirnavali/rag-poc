# Faz 3: Uçtan Uca Ingestion Pipeline (Docling + Late Chunking + ChromaDB)

## Hedef
Faz 1 (Packer) ve Faz 2'de (DoclingManager) oluşturduğumuz yapısal veri çıkarma mantığını, Jina v3 (Late Chunking Embedder) ve ChromaDB ile birleştirerek üretime hazır, tam otomatik bir veri yükleme (ingestion) boru hattı kurmak.

## Mevcut Durum Özeti
- `DoclingManager`: Dokümanları okur, anlamsal olarak böler, gereksiz küçük parçaları birleştirir (Packer) ve bize `full_text` ile koordinatları (span) dönen `chunks` listesini verir.
- `LocalLateChunkingEmbedder`: Model sınırlarına kadar olan devasa `full_text`'i tek seferde okur, verdiğimiz `span` koordinatlarına göre en kaliteli, bağlamı korunmuş vektörleri üretir.

## Uygulama Planı (Önerilen Değişiklikler)

### 1. Ingestion Pipeline Sınıfı
**[YENİ] `src/trainer/ingestion/pipeline.py`**
- `IngestionPipeline` adında yeni bir orkestratör sınıf oluşturulacak.
- **Akış:**
  1. `DoclingManager.convert_and_pack()` çağrılarak PDF/Word dosyası okunur.
  2. Dönen `chunks` listesinden sadece `span` (koordinat) bilgileri ayıklanır.
  3. `LocalLateChunkingEmbedder.embed_with_late_chunking(full_text, spans)` çağrılarak yüksek kaliteli vektörler elde edilir.
  4. Vektörler, orijinal metinler (`text`) ve Docling'den gelen zengin etiketler (`metadata`) birleştirilerek ChromaDB'ye (veya SQLite + ChromaDB hibrid yapısına) kaydedilir.

### 2. İndeksleme Scriptinin Güncellenmesi
**[GÜNCELLEME] `src/trainer/minutes/index.py` (veya yeni bir script)**
- Mevcut indeksleme mekanizması, eski LangChain splitter'ları yerine bu yeni `IngestionPipeline`'ı kullanacak şekilde revize edilecek.
- Tekli dosya okuma veya klasör tarama desteği eklenecek.

### 3. Test ve Doğrulama
**[YENİ] `tests/test_ingestion_pipeline.py`**
- Uçtan uca test: Gerçek bir doküman verilecek, vektörlerin çıkarılıp çıkarılmadığı ve geçici bir ChromaDB koleksiyonuna boyut (dimension) uyumsuzluğu olmadan kaydedilip kaydedilmediği test edilecek.

---
*Not: Bu plan, context window'u şişirmemek adına yeni bir sohbete (conversation) başlarken doğrudan referans olarak kullanılabilir.*
