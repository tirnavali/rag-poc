# Faz 4, Aşama 1: Sorgu Zenginleştirme ve Metrik Ölçümü

Qdrant (Sparse/Dense/ColBERT) fikri mimari açıdan **mükemmel bir tespit**. Mevcut ChromaDB + SQLite hibrit yapısını tek potada eriteceği ve RRF'i (Reciprocal Rank Fusion) kendi içinde donanımsal hızlandırmayla yapacağı için production ortamında kesinlikle oraya geçmeliyiz.

Ancak geçiş yapmadan veya Reranker (Yeniden Sıralama) eklemeden önce **mevcut durumu ölçmemiz (Retrieval Metrikleri)** ve basit müdahalelerle (Sorgu Zenginleştirme) recall'ı (geri çağırma) ne kadar artırabildiğimizi görmemiz gerekiyor.

> [!NOTE]
> Bu doküman, Docling tabanlı yeni `data_ingest` pipeline'ı tamamlanıp veritabanı chunkları oluşturulduktan ve yeni bir Soru-Cevap (Ground Truth) veri seti üretildikten sonra uygulanmak üzere kalıcı referans olarak hazırlanmıştır.

## Hedef
1. Mevcut `HybridRetriever` altyapısına LLM destekli **Query Expansion (Sorgu Zenginleştirme)** yeteneğini entegre etmek.
2. Sistemin arama başarımını (Recall@K, MRR vb.) ölçebileceğimiz, **geliştirici dostu (developer-friendly)** ve karşılaştırmalı (A/B Testi) bir test mekanizması kurmak.

## Strateji: A/B Testli Prompt Mimarisinin Kurulması

Sorgu zenginleştirme sırasında her arama isteğinde yerel LLM (Ollama) çalışacaktır. Arama süresindeki gecikmeyi haklı çıkaracak en iyi stratejiyi bulmak için aşağıdaki yaklaşımlar script üzerinden A/B testine tabi tutulacaktır:

1. **Baseline (Zenginleştirme Yok):** Sadece orijinal kullanıcı sorgusu.
2. **HyDE (Hayali Doküman/Cevap):** LLM'e "Bu soruya verilecek örnek bir arşiv dokümanı/cevabı yaz" denir. Üretilen sahte cevap vektörize edilerek vektör araması yapılır (Dense vektörlerde çok başarılıdır).
3. **Multi-Query (Çoklu Sorgu/Eş Anlamlılar):** LLM'den sorunun 3 farklı varyasyonunu (eşanlamlılar, farklı ifade edilişleri) üretmesi istenir. BM25 (Kelime bazlı) aramalarda çok başarılıdır.
4. **Hybrid Expander (İkisi Birden):** Vektör araması için HyDE, BM25 araması için Multi-Query/Keyword üretilir.

## Uygulama Adımları (Gelecek Faz)

### 1. Veri Setinin Hazırlanması
- Docling pipeline'ı ile veritabanı chunkları tamamen oluşturulduktan sonra, yeni verilere dayalı bir `eval_dataset.json` (Soru ve Karşılık Gelen Kayıt No) oluşturulacak.

### 2. Sorgu Zenginleştirme Entegrasyonu (`hybrid.py`)
- `HybridRetriever.retrieve()` fonksiyonuna `expand_strategy` parametresi eklenecek.
- `OllamaGenerator` içerisinde hem HyDE hem de Multi-Query promptları tanımlanarak dinamik zenginleştirme altyapısı kurulacak.

### 3. Retrieval Metrikleri Test Scripti (`evaluate_retrieval.py`)
- Sistemin arama başarımını ölçecek özel bir değerlendirme scripti yazılacak.
- Script, veri setindeki soruları tüm stratejilerle (Baseline, HyDE, Multi-Query, Hybrid) çalıştıracak.
- **Çıktı Formatı:** Sonuçlar terminalde geliştirici dostu, okunabilir bir tablo (Markdown/PrettyPrint) olarak basılacak. Hangi stratejinin Recall@5 ve Recall@10 metriklerinde ne kadar fark yarattığı net bir şekilde karşılaştırılabilecek.
