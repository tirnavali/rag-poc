# Arama ve Getirme Modülü (Retriever)

Bu klasör, kullanıcı sorularına yanıt bulmak için ChromaDB'den en alakalı bilgileri getiren bileşenleri içerir. Sistem, vektör (anlam bazlı) arama + cross-encoder yeniden sıralama (rerank) kullanır. (BM25/anahtar-kelime araması kaldırıldı; yalnızca ANN+rerank kaldı.)

## Temel Özellikler

*   **Vektör Arama + Rerank**: ChromaDB üzerinde ANN ile aday getirip cross-encoder ile yeniden sıralar.
*   **Reciprocal Rank Fusion (RRF)**: Çoklu koleksiyon (çapraz kaynak) sonuçlarını akıllı bir algoritma ile birleştirerek en doğru sıralamayı yapar.
*   **Akıllı Yönlendirme (Routing)**: Sorunun içeriğine göre aramayı otomatik olarak TBMM Tutanakları, Gazete Kupürleri veya her ikisine birden yönlendirir.
*   **Tarih Filtreleme**: Sorgu içindeki tarih ifadelerini (örn: "1996 yılındaki...", "20 Ocak 2000 tarihli...") otomatik algılar ve aramayı bu tarihlerle kısıtlar.

## Bileşenler

*   **`vector_search.py`**: Üretim ve benchmark için ortak vektör arama ilkeli (`VectorSearch`). ChromaDB ANN + cross-encoder rerank tek yoldan geçer. Tüm `chromadb.*` çağrıları `src/common/chroma` üzerinden gider (DB değişimi yalnızca o dosyayı etkiler).
*   **`vector_retriever.py`**: Üretim retriever'ı (`VectorRetriever`) — `VectorSearch`'ü tarih filtreleme, son-işleme ve `RetrievalResult` şekline sararak sunar.
*   **`reranker.py`**: Cross-encoder yeniden sıralayıcı (`CrossEncoderReranker`).
*   **`multi_source.py`**: Çoklu koleksiyon retriever'ı (`MultiSourceRetriever`) — her koleksiyonda `VectorRetriever` çalıştırır, sonuçları RRF ile füzyonlar.
*   **`query_parser.py`**: Sorgu anahtar kelimelerine göre hangi koleksiyon(lar)ın aranacağına karar verir (kaynak yönlendirme).
*   **`context.py`**: Getirilen sonuçları LLM'in anlayacağı bir "bağlam" (context) haline getirir.

## Arama Akışı

1.  **Yönlendirme**: `query_parser.py` sorgu anahtar kelimelerine göre hangi koleksiyon(lar)ın aranacağını belirler.
2.  **Getirme (koleksiyon başına)**: `VectorRetriever` → `VectorSearch` ChromaDB ANN ile `fetch_k` aday getirir; `where_filter` (FilterExtractor → `build_chroma_where`) varsa uygular. `author` filtresi, `AuthorResolver` ile koleksiyonun gerçek etiketlerine çözülüp `author $in [...]`'a çevrilir (exact-match yerine; bkz. `src/common/author_resolver.py`).
3.  **Yeniden Sıralama**: `CrossEncoderReranker` adayları yeniden puanlar (`fetch_k → coarse_k → final_k`).
4.  **Füzyon (çoklu koleksiyon)**: Birden çok koleksiyon arandıysa `MultiSourceRetriever` koleksiyon başına sonuçları RRF (k=60) ile birleştirir.
5.  **Bağlam**: `context.py` en yüksek puanlı chunk'ları model bağlamına dönüştürür.
