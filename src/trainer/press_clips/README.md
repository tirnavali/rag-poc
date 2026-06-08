# Gazete Kupürü İşleme Modülü (Press Clips)

Bu klasör, gazete arşivinden gelen kupür verilerini (CSV formatında) işlemek, veritabanına aktarmak ve vektör tabanlı arama için indekslemek amacıyla kullanılan bileşenleri içerir.

## Bileşenler

*   **`load_csv.py`**: Ham CSV formatındaki gazete kupürü verilerini okur ve bunları SQLite veritabanına (`press_clips.db`) aktarır. Tarih formatlarını normalize eder ve eksik verileri temizler.
*   **`build_fts.py`**: SQLite üzerindeki kupür verileri için FTS5 (Full Text Search) tablosunu oluşturur. Bu sayede kelime bazlı (keyword search) aramaların çok hızlı yapılmasını sağlar.
*   **`index.py`**: SQLite'taki metinleri parçalara ayırır (chunking), Ollama üzerinden embedding'lerini üretir ve bu vektörleri metadata bilgileriyle birlikte ChromaDB'ye indeksler.

## Çalışma Akışı

1.  **Veri Yükleme**: CSV dosyası SQLite tablosuna (`kupurler`) aktarılır (`load_csv.py`).
2.  **Arama İndeksi**: Metin bazlı aramalar için FTS5 sanal tablosu ve otomatik senkronizasyon tetikleyicileri (triggers) kurulur (`build_fts.py`).
3.  **Vektör İndeksi**: Metinler anlamlı parçalara bölünerek vektör veritabanına (ChromaDB) kaydedilir (`index.py`).
