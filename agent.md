# Antigravity Agent Kuralları ve Durumu

Bu dosya, bu proje üzerinde çalışan AI asistanı (Antigravity) için temel kuralları, tercihleri ve mevcut proje durumunu özetler.

## Temel Prensipler
1. **Python Uzmanı:** Kodlar her zaman "Pythonic", temiz ve modüler olmalıdır.
2. **Dil:** Açıklamalar her zaman **Türkçe** olmalıdır. Kod yorumları ve değişken adları İngilizce kalabilir (proje standardı).
3. **Sanal Ortam (venv):** Her zaman `.venv` dizinindeki sanal ortam kullanılmalıdır (`.venv/bin/python`, `.venv/bin/pip`).
4. **Test Odaklı Geliştirme:** Yapılan her değişiklikten sonra ilgili testler mutlaka çalıştırılmalıdır.

## Teknik Standartlar
- **Docling:** Yapısal doküman analizi için ana araçtır.
- **Late Chunking:** Bağlamsal vektör üretimi için Jina v3 (Local) kullanılmalıdır.
- **ChromaDB:** Vektör veritabanı olarak doğrudan (PersistentClient) kullanılır.
- **Kod Temizliği:** Fonksiyonlar kısa, sınıflar tek bir sorumluluğa odaklı (Single Responsibility) olmalıdır.

## Mevcut Durum (Faz 3)
- `IngestionPipeline` sınıfı tamamlandı.
- `src/trainer/ingestion/index.py` ile genel indeksleme desteği eklendi.
- `tests/test_ingestion_pipeline.py` ile uçtan uca akış doğrulandı.

## Gelecek Planlar
- Diğer veri kaynaklarının (gazete arşivi vb.) yeni pipeline'a taşınması.
- SQLite ve ChromaDB arasındaki senkronizasyonun güçlendirilmesi.

---
*Son Güncelleme: 2026-05-11*
