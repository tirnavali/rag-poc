# Arama ve Getirme Modülü (Retriever)

Bu klasör, kullanıcı sorularına yanıt bulmak için veritabanlarından (SQLite ve ChromaDB) en alakalı bilgileri getiren bileşenleri içerir. Sistem, "Hybrid Search" (Hibrit Arama) yaklaşımını kullanarak hem kelime eşleşmesine hem de anlamsal benzerliğe dayalı sonuçlar üretir.

## Temel Özellikler

*   **Hybrid Search**: BM25 (anahtar kelime bazlı) ve Vektör (anlam bazlı) aramayı birleştirir.
*   **Reciprocal Rank Fusion (RRF)**: Farklı arama motorlarından gelen sonuçları akıllı bir algoritma ile birleştirerek en doğru sıralamayı yapar.
*   **Akıllı Yönlendirme (Routing)**: Sorunun içeriğine göre aramayı otomatik olarak TBMM Tutanakları, Gazete Kupürleri veya her ikisine birden yönlendirir.
*   **Tarih Filtreleme**: Sorgu içindeki tarih ifadelerini (örn: "1996 yılındaki...", "20 Ocak 2000 tarihli...") otomatik algılar ve aramayı bu tarihlerle kısıtlar.

## Bileşenler

*   **`hybrid.py`**: Sistemin ana arama motorudur (`HybridRetriever`). SQLite FTS5 ve ChromaDB sorgularını yönetir ve RRF ile sonuçları birleştirir.
*   **`minutes_retriever.py`**: Sadece TBMM tutanakları üzerinde arama yapan özelleşmiş sınıftır.
*   **`press_retriever.py`**: Sadece gazete kupürleri üzerinde arama yapan özelleşmiş sınıftır.
*   **`query_parser.py`**: Kullanıcı sorgusunu analiz eder, hangi veri kaynağının kullanılacağına karar verir ve tarih bilgilerini ayıklar.
*   **`context.py`**: Getirilen sonuçları LLM'in (Dil Modeli) anlayabileceği bir "bağlam" (context) haline getirir.

## Arama Akışı

1.  **Sorgu Analizi**: Kullanıcı sorusu `query_parser.py` ile incelenir (Tarih var mı? Kaynak belirtilmiş mi?).
2.  **Paralel Arama**: 
    *   SQLite FTS5 ile anahtar kelime araması yapılır.
    *   ChromaDB ile vektör (anlam) araması yapılır.
3.  **Birleştirme (Fusion)**: RRF algoritması her iki listeden gelen sonuçları puanlayıp tek bir listede toplar.
4.  **Veri Getirme**: En yüksek puanlı kayıtların tam metinleri SQLite'tan çekilir ve kullanıcıya/modele sunulur.
