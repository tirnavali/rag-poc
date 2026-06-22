# Parametre seçimleri

| \# | Parametre | Önerilen Değer Aralıkları | Deney Amacı | Beklenen Etki | Ölçülecek Metrikler | Öncelik |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | **Embedding Modeli** | Nomic v2 MoE vs. Jina v3 (retrieval LoRA) | Domain'e özgü en iyi embedderi belirle | NDCG@10, Recall@10'da %5–15 fark beklenir | Recall@5, Recall@10, NDCG@10, Latency | 🔴 1 |
| 2 | **Chunk Boyutu** | 128, 256, 512, 768, 1.024 token | Precision-context dengesi bul | Küçük chunk → yüksek precision; büyük chunk → yüksek recall | Context Precision, Context Recall, Faithfulness | 🔴 1 |
| 3 | **Chunking Stratejisi** | Fixed-size, Recursive, Sentence-based, Semantic, Parent-Child | En uygun parçalama yöntemini bul | Sentence-based: \+%15–30 retrieval accuracy\[14\] | NDCG@10, Context Recall, Answer Relevancy | 🔴 1 |
| 4 | **Hybrid Search (BM25 \+ Dense)** | Dense-only vs. BM25-only vs. RRF(k=60) vs. Weighted(α=0.3–0.7) | Keyword+semantik kombinezonunu optimize et | %8–15 accuracy artışı\[36\] | Recall@10, NDCG@10, P95 Latency | 🔴 1 |
| 5 | **Jina LoRA Adaptör Seçimi** | `retrieval.query` \+ `retrieval.passage` vs. `text-matching` vs. varsayılan | Task-specific LoRA etkisini ölç | Yanlış adaptör seçimi %10–20 kayba yol açabilir | NDCG@10, Recall@5, Context Precision | 🔴 1 |
| 6 | **Similarity Metric** | Cosine vs. Dot Product vs. L2 | Modele uygun metric bul | Hatalı metric seçimi retrieval'ı bozar\[27\] | Recall@10, Precision@10 | 🔴 1 |
| 7 | **Top-K Değeri** | K \= 3, 5, 8, 10, 15, 20 | LLM context kalitesi \+ maliyet dengesi | K=4→10 latency 2x artar\[16\] | Faithfulness, Latency, Cost/query | 🔴 1 |
| 8 | **Reranking** | Reranking yok vs. Cross-encoder vs. LLM-judge | Retrieval sonrası sıralama kalitesi | \+%28 NDCG@10\[34\]; \+300–800ms latency | NDCG@10, Faithfulness, P95 Latency | 🔴 1 |
| 9 | **Chunk Overlap Oranı** | %0, %10, %15, %20, %25 | Sınır kayıplarını minimize et | %10–20 overlap, boundary failure azaltır\[13\] | Context Recall, Faithfulness | 🟠 2 |
| 10 | **Embedding Boyutu (Matryoshka)** | Nomic: 256, 512, 768; Jina: 64, 256, 512, 1.024 | Maliyet-kalite eşiğini bul | 3x depolama tasarrufu\[4\]; minimal kalite kaybı | Recall@10, Depolama maliyeti, Latency | 🟠 2 |
| 11 | **HNSW M parametresi** | M \= 8, 16, 32, 64 | Recall/bellek dengesi | Yüksek M → yüksek recall, yüksek RAM kullanımı\[29\] | Recall@10, Index boyutu, Query latency | 🟠 2 |
| 12 | **HNSW efConstruction** | 100, 200, 400 | Index inşa kalitesi | Yüksek → daha iyi recall; trade-off: inşa süresi\[26\] | Recall@10, Index inşa süresi | 🟠 2 |
| 13 | **HNSW efSearch** | 50, 100, 200, 400 | Sorgu zamanı recall/latency dengesi | efSearch↑ → recall↑, latency↑ (doğrusal)\[33\] | Recall@10, P50/P95 Latency | 🟠 2 |
| 14 | **Metadata Filtreleme** | Filtre yok vs. category/date/source filtresi | Pre-retrieval arama alanı daraltma | Arama süresi ve maliyet düşüşü\[25\] | Query latency, Precision@K | 🟠 2 |
| 15 | **Normalisation / Veri Temizleme** | Ham metin vs. temizlenmiş metin (HTML, boilerplate kaldırma) | Vektör kalitesinin temel etkisini ölç | Gürültü azalması → \+%5–10 retrieval kalitesi\[22\] | Context Precision, Faithfulness | 🟠 2 |
| 16 | **RRF k sabitesi** | k \= 20, 60, 100, 200 | Hybrid fusion robust noktası | k=60 standart, domain'e göre ayar gerekebilir\[44\] | Recall@10, NDCG@10 | 🟠 2 |
| 17 | **Reranker aday kümesi boyutu** | 10, 25, 50, 75, 100 | Reranker için optimum aday sayısı | 50–75 arası en iyi kalite/latency dengesi\[34\] | NDCG@10, Reranking latency | 🟠 2 |
| 18 | **Parent-Child Chunk Oranı** | Parent: 512–1.024 / Child: 128–256 | Small-to-big retrieval kalibrasyonu | Precision \+ context dengesi iyileşir\[19\] | Context Precision, Context Recall | 🟡 3 |
| 19 | **Sentence Window Boyutu** | ±1, ±2, ±3 cümle | Bağlam penceresi genişliğini optimize et | Dar pencere → precision; geniş pencere → recall\[19\] | Context Recall, Faithfulness | 🟡 3 |
| 20 | **Contextual Retrieval (LLM prefix)** | Yok vs. kısa bağlam özeti ekleme | Chunk başına anlamsal zenginleştirme | Retrieval kalitesinde artış; LLM maliyeti ekler\[12\] | Context Precision, Maliyet/chunk | 🟡 3 |
| 21 | **IVF nlist / nprobe** | nlist: √N–4√N; nprobe: nlist'in %1–10'u | Büyük ölçekte hız/recall dengesi | Yüksek nprobe → 3x latency artışı riski\[33\] | Recall@10, Query latency | 🟡 3 |
| 22 | **Embedding Cache** | Cache yok vs. query embedding cache | Tekrarlayan sorgu maliyeti azaltma | %65–70 latency düşüşü tekrarlayan sorgularda\[34\] | P95 Latency, Cost/query | 🟡 3 |
| 23 | **Smart Model Routing** | Tek model vs. küçük/büyük model yönlendirme | Maliyet optimizasyonu | %60–80 maliyet düşüşü\[34\] | Cost/query, Faithfulness | 🟡 3 |
| 24 | **Context Budgeting** | Sabit K vs. token bütçe bazlı dinamik K | Context overflow önleme | Token aşımlarını engeller; tutarlılık artar\[23\] | Hata oranı, Context Precision | 🟡 3 |
| 25 | **Fine-tuning stratejisi** | Bağımsız vs. ortak vs. iki aşamalı fine-tuning | Domain adaptasyonu | EM ve F1 kalite metriklerinde eşit artış\[45\] | EM, F1, Faithfulness | 🟡 3 |

# Deneysel Parametre Seçimi

**Aşama A → Sıralı (sıfırdan ilk iyiye):** Her parametreyi bağımsız tarar, en iyiyi kilitlersin. **Aşama B → Latin Hypercube Sampling (LHS):** Kalan bağımlı parametreleri birlikte örnekler, az kombinasyonla geniş uzayı temsil edersin.

---

## Başlangıç Deney Tablosu (Elzem 7 Parametre)

Aşağıdaki 7 parametre, literatürde retrieval kalitesine en yüksek etkiyi gösteren, birbirinden görece bağımsız değişkenlerdir. Bunları **sırayla** sabitleye sabitleye ilerle:

| Sıra | Parametre | Test Edilecek Değerler | Sabit Tutulacak (başlangıç) | Neden Elzem | Ölçüt |
| :---: | :---- | :---- | :---- | :---- | :---- |
| **1** | **Chunking Stratejisi** | Recursive / Sentence-based / Parent-Child | chunk=512, overlap=%10, K=5, dense-only, no rerank | Retrieval doğruluğunu en çok belirleyen yapısal karar | Context Recall, Faithfulness |
| **2** | **Chunk Boyutu** | 256 / 512 / 1.024 token | Kazanan strateji, diğerleri sabit | Precision-recall dengesini doğrudan kontrol eder | Context Precision \+ Context Recall |
| **3** | **Embedding Modeli** | Nomic v2 MoE / Jina v3 (retrieval LoRA) | Kazanan chunk yapısı \+ boyutu | Domain uyumu %80 burada belirlenir | Recall@5, NDCG@10 |
| **4** | **Hybrid Search** | Dense-only / RRF(k=60) / Weighted α=0.5 | Kazanan model | \+%8–15 accuracy beklentisi | Recall@10, NDCG@10 |
| **5** | **Reranking** | Yok / Cross-encoder | Kazanan hybrid ayarı, K=20→5 | \+%28 NDCG@10 potansiyeli | NDCG@10, P95 Latency |
| **6** | **Top-K** | 3 / 5 / 10 | Kazanan reranker yapısı | Latency-faithfulness eşiği burada netleşir | Faithfulness, Cost/query, Latency |
| **7** | **Embedding Boyutu** (Matryoshka) | Nomic: 256/512/768 — Jina: 256/512/1024 | Tüm üsttekiler sabit | Maliyet/depolama optimizasyonu, minimal kalite kaybı | Recall@10, Depolama maliyeti |

**Toplam çalıştırılacak deney:** her adımda 2–3 değer → **yaklaşık 16–18 bağımsız run**. Tüm kombinasyonlar yerine sıralı "greedy search" uygulanır: her adımda kazanan değer bir sonraki adımın sabit parametresi olur.

---

## Aşama B: Latin Hypercube ile Rastgele Örnekleme (Opsiyonel)

Eğer ikincil parametreleri (HNSW M, efSearch, overlap oranı, RRF k) de dahil etmek istersen ve bütçen **N \= 10–20 run** ise:

```
Parametre uzayı:
  chunk_overlap  : [%0, %10, %20]
  hnsw_M         : [16, 32, 64]
  hnsw_efSearch  : [50, 100, 200]
  rrf_k          : [20, 60, 100]

LHS ile 12 kombinasyon örnekle →
  her kombinasyon bir "konfigürasyon"
  hepsini sabit (Aşama A kazananı) pipeline üzerinde çalıştır
  NDCG@10 ve Faithfulness ile sırala
```

---

## Pratik Karar Ağacı

```
Baseline kur (Recursive, 512, Dense, K=5, NoRerank, Nomic)
    │
    ▼
Chunking Stratejisi → kazananı kilitle
    │
    ▼
Chunk Boyutu → kazananı kilitle
    │
    ▼
Embedding Modeli → kazananı kilitle
    │
    ▼
Hybrid Search ekle → kazananı kilitle
    │
    ▼
Reranking ekle → latency bütçesi yeterliyse kilitle
    │
    ▼
Top-K tara → Faithfulness/latency dengesini bul
    │
    ▼
Matryoshka boyutu → kabul edilebilir kalite kaybı eşiğini bul
    │
    ▼
(Opsiyonel) LHS ile ikincil parametreleri 12 run'da tara
```

---

## Özet: Kaç Run Gerekir?

| Aşama | Run Sayısı | Kapsanan Alan |
| :---- | :---- | :---- |
| Aşama A (sıralı, greedy) | \~16–18 | En yüksek etki parametreleri |
| Aşama B (LHS, opsiyonel) | 10–12 | İkincil parametre uzayı |
| **Toplam** | **\~28–30** | Pipeline optimizasyonunun \~%90'ı |

