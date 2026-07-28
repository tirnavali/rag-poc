# Golden Set Not Defteri

Golden Q&A seti (`tests/fixtures/golden_tbmm27001001.json`) oluşturulurken karşılaşılan
problemler ve çözümleri. Yeni bir problem yaşayıp çözdüğümüzde buraya bir madde
ekleniyor — kronolojik, en yeni en üstte.

---

## 2026-07-03 — Kök neden düzeltmesi: VectorRetriever artık saf bir vector-sorgu katmanı

Bir önceki maddedeki `where_filter={}` hilesi geçici bir bypass'tı. Kullanıcı
mimari olarak doğrusunu istedi: "vector retrieval saf vector sorgusu biçiminde
çalışmalı, reranker/auto filter extraction ayrı bir katman olmalı." İnceleme
sırasında bunun sadece stil tercihi değil, **ana chat.py yolunda da canlı bir
bug** olduğu ortaya çıktı: `src/generator/service.py` zaten "Retriever
filtreden bağımsız olmalı" diye belgelenmiş bir mimari niyet taşıyordu ve
`FilterExtractor.fallback_chain()` her zaman `("semantic_only", None)` ile
biten bir "filtresiz son çare" tanımlıyordu — ama `VectorRetriever`'ın kendi
içindeki otomatik tarih çıkarımı bu `None`'ı sessizce yeniden filtreliyordu,
yani "semantic_only" kademesi hiçbir zaman gerçekten semantic-only olmuyordu.

**Çözüm** (bkz. plan: `~/.claude/plans/vector-retrieval-saf-vector-drifting-russell.md`):
- Yeni `src/retriever/filters.py` → `auto_date_where_filter(query)`: eski
  otomatik çıkarım mantığı buraya taşındı, artık **opt-in**.
- `VectorRetriever.retrieve()`: `where_filter=None` artık HER ZAMAN "filtresiz"
  demek; `rerank: bool` parametresi kalktı, yerine hazır bir `reranker` örneği
  (ya da `None`) alıyor — `settings.USE_RERANKER`'ı hiç okumuyor.
- `RAGService` (`src/generator/service.py`), `DeepPipeline`, `router_server.py`
  — eskiden sessizce aldıkları davranışı artık açıkça opt-in ediyorlar
  (reranker `__init__`'te bir kez inşa edilip önbelleğe alınıyor).
- `golden_builder.py`'deki `where_filter={}` hilesi tamamen kalktı — artık
  gereksiz, saf varsayılan zaten doğru.
- `src/mcp/press_server.py` ve `src/retriever/multi_source.py`'ye kasıtlı
  olarak dokunulmadı: hiçbir çağıran filtre/rerank opt-in'i istemiyor, saf
  varsayılanı kod değişikliği olmadan otomatik miras alıyorlar (CLAUDE.md'nin
  "hipotetik gelecek ihtiyacı için tasarlama" ilkesi).

Doğrulama: `pytest tests/ -m "not integration and not slow"` → 545 geçti, 5
hata (test_ingest_cli.py, test_ocr_quality.py) refactor öncesinde de mevcuttu,
alakasız. golden_builder'da "2012 KPSS" sorgusu artık HTTP 200 + gerçek
sonuçlarla dönüyor (uçtan uca doğrulandı).

---

## 2026-07-03 — Sorguda geçen bir yıl (ör. "2012") retrieval'ı 0 sonuca düşürüyor

**Belirti:** `"2012 KPSS sınavında soruların çalındığı iddiaları"` sorgusu 0 sonuç
döndürüyor; sorgudan sadece `2012`'yi çıkarınca (`"KPSS sınavında soruların
çalındığı iddiaları"`) sonuçlar geliyor.

**Sebep:** `VectorRetriever.retrieve()` (`src/retriever/vector_retriever.py`),
`where_filter=None` geldiğinde `src/common/dates.py`'deki `extract_dates` ile
sorgu metnindeki yılları otomatik ayrıştırıp `where_year_filter` üzerinden
`{"year": {"$eq": 2012}}` gibi bir Chroma metadata filtresi kuruyor. Bu üretim
sohbet akışı için doğru bir varsayım ("7 Temmuz 2018 birleşiminde..." gibi
belgenin kendi tarihini soran sorgular için) ama golden_builder'ın etiketleme
havuzu için yanlış: sorguda geçen yıl çoğu zaman **konuşulan olayın tarihi**,
tutanağın (belgenin) kendi tarihi değil. Koleksiyon 27. dönem tutanakları
(2018-2019) olduğundan `year=2012` filtresi hiçbir chunk'a denk gelmiyor ve
havuz sıfırlanıyor — doğrulandı: `where_filter={"year":{"$eq":2012}}` ile 0,
`where_filter=None` ile 10 sonuç.

**Çözüm:** `scripts/golden_builder.py` `_handle_retrieve()`'de `retr.retrieve(...)`
çağrısına `where_filter={}` eklendi. `{}`, `None`'dan farklı olarak
`VectorRetriever`'ın otomatik tarih ayrıştırma dalını (`if where_filter is None`)
atlatıyor ve `query_collection`'da boş dict falsy olduğundan Chroma'ya hiç
filtre gitmiyor — etiketleme havuzu artık sorgudaki hiçbir tarihe göre
daraltılmadan tüm derlemden geliyor. Üretim (`chat.py`/MCP) tarafındaki
otomatik tarih filtresi davranışı bilerek değiştirilmedi, sadece golden_builder
kapsamında bypass edildi.

---

## 2026-07-03 — Retrieval sonuçlarında "mükerrer" gibi görünen chunk'lar

**Belirti:** 30 sonuçluk retrieval havuzunda örn. 3. ve 17. sıradaki sonuçlar
neredeyse birebir aynı metni gösteriyor, farklı skorla.

**Şüphe:** Multi-collection karışması (birden fazla koleksiyondan sonuç geliyor
olabilir mi?).

**Bulgu:** Hayır — `golden_builder.py`'deki `_get_retriever()` tek bir
`CollectionSpec` üzerinden tek bir `VectorSearch` çağırıyor (`tutanaklar_nomic_chunk256_768d`).
Multi-source/RRF fusion (`src/retriever/multi_source.py`) golden builder'da hiç
devrede değil. Chroma'da mükerrer `chunk_id` de yok.

**Gerçek sebep:** Kaynağın kendisi. Bir önerge/gündem maddesi karara bağlanana
kadar birden çok birleşimin gündeminde **birebir aynı metinle** tekrar tekrar
listeleniyor (aynı sıra no ile). Her tekrar farklı `document_id`/sayfa/chunk_id
taşıdığı için ayrı ayrı indeksleniyor ve embedding bağlamı hafif farklı olduğundan
farklı skor alıyor. Bug değil.

**Çözüm:** `golden_builder.html`'e sonuç listesinde aynı metnin başka rank'larda
da geçtiğini gösteren turuncu `⧉ tekrar: #N` rozeti eklendi (`normSnippet` +
`dupeGroups` — `scripts/golden_builder.html`). Etiketleme sırasında kafa
karışıklığını önlüyor; retrieval/kod tarafında bir düzeltme gerekmedi.

---

## 2026-07-03 — Retrieval sonucuna tıklayınca hangi sonucun seçili olduğu belli değildi

**Belirti:** Solda sayfa önizlemesi açılıyor ama sağdaki listede hangi satıra
tıklandığı görsel olarak işaretlenmiyordu, kafa karıştırıyordu.

**Çözüm:** `.res` satırlarına tıklanan sonucu işaretleyen `.active` CSS class'ı
eklendi (mavi vurgu + sol kenar çizgisi); yeni arama yapılınca sıfırlanıyor.
`scripts/golden_builder.html`.

---

## 2026-07-03 — Sunucu kendi bilgisayarımda açılmadı

**Belirti:** `python -m scripts.golden_builder` sonrası `localhost:8765`
kullanıcının kendi tarayıcısında açılmadı.

**Sebep:** Komut, kullanıcının SSH ile bağlandığı paylaşımlı sunucuda
(`tbmmai-node3`, `10.20.24.13`) çalıştırılmıştı — `localhost` sunucunun kendi
localhost'u, kullanıcının makinesi değil.

**Çözüm:** Server `--host 0.0.0.0` ile tüm arayüzlere bind edildi, kullanıcı
`http://10.20.24.13:8765` üzerinden erişti. Alternatif (daha güvenli): SSH port
forwarding (`ssh -L 8765:localhost:8765 ...`). Güvenlik notu: `0.0.0.0` bind
kimlik doğrulamasız olduğu için sadece güvenilir iç ağda (TBMM ağı) yapılmalı.

---

## 2026-07-03 — `ModuleNotFoundError: No module named 'chromadb'`

**Sebep:** Server sistem `python3` ile başlatılmıştı, proje `.venv`'i
aktive edilmemişti — chromadb sadece `.venv` içine kurulu.

**Çözüm:** `.venv/bin/python -m scripts.golden_builder ...` ile başlat.
Proje her zaman `.venv` altında çalıştırılmalı, çıplak `python3`/`python` ile değil.

---

## Önceden biriken üretim kuralları (bkz. `scripts/lint_golden.py`)

Bu bölüm 2026-07-02 golden set v2 genişletme turunda çıkan, henüz bu deftere
tarihsiz not düşülmüş ama hâlâ geçerli kurallar:

- Cevap (`golden_answer`), atıfta bulunulan sayfada **birebir/yüksek örtüşmeli**
  geçmeli; özellikle sayılar aynen eşleşmeli (`answer_in_pages` bunu canlı
  kontrol ediyor, kaydı engellemiyor sadece uyarıyor).
- Yasak soru kalıpları: ordinal birleşim çıpası ("12. birleşimde..." gibi kırılgan
  referanslar), meta sorular, kâtip/idare amiri soruları, açılış saati soruları,
  "hangi ilin vekili" soruları.
- Tematik cevaplar sayfadan doğrudan alıntıyla kurulmalı, parafraz değil.
- `lint_golden.py` ID regex'i `tbmm27-\d{2}-\d{2,3}-\d{3}` — yy1'e sabit değil,
  hem 27.1 hem 27.3 dönem/yasama yılı ID'lerini kapsıyor.
- **Dikkat — olay karışması:** 2015 Suruç bombalı saldırısı ile Haziran 2018
  Suruç Şenyaşar olayı ayrı olaylar; arama sonuçlarında birbirine karışabiliyor,
  golden soru/cevap yazarken olay adı + tarihi birlikte doğrulanmalı.

---

## 2026-07-13 — Fihrist-güdümlü aday üretimi (cilt 8, 11, 12, 13, 15)

**Yöntem:** TBMM resmî cilt fihristleri (www5.tbmm.gov.tr, `tbmm27XXXfih.pdf`)
metin PDF'leri indirilip cilt başına tarandı; fihrist maddelerinden **50 yeni
aday soru** üretildi ve doğrudan fixture'a (`tests/fixtures/golden_tbmm27001001.json`)
`unverified` etiketiyle eklendi (ID'ler `tbmm27-02-BB-4NN` serisi; sayfa tohumları
canlı retrieval top-30 + rerank ile). Kapsam: birleşim 33–100
(15 Aralık 2018 – 10 Temmuz 2019, 2. yasama yılı). Fixture: 151 → 201 kayıt.

**Tasarım ilkesi (embedding kalite ölçümü):** Sorgu kelimeleri tutanaktaki
resmî dilden bilinçli olarak farklılaştırıldı — eş anlamlı köprüler kurulması
gerekiyor (örn. sorguda "tüberküloz" / belgede "verem"; "vekilliği düşmek" /
"üyeliğin kendiliğinden sona ermesi"; "saray" / "Cumhurbaşkanlığı"; "başkanlık
sistemi" / "Cumhurbaşkanlığı hükûmet sistemi"). Tip dağılımı: 18 olgusal (A),
20 anlatısal (B), 12 kısa/muallak anahtar kelime (C).

**Mükerrerlik kontrolü:** Adaylar mevcut 151 fixture sorusuna ve birbirlerine
karşı token-Jaccard (eşik 0.34) ile tarandı — çakışma yok. Ajan üretiminden
2 aday elendi: "Bolu belediye başkanı seçilen vekil" (fixture'da `tbmm27-02-68/69`
zaten var) ve "Kızılay kirada" (`tbmm27-02-90-302` "Kızılay yönetimine yönelik
iddialar" ile örtüşme riski).

**Ders:** Fihrist (cilt dizini) soru madenciliği için çok verimli — konu + birleşim
+ sayfa aralığını hazır veriyor; gündem dışı konuşmalar ve soru önergeleri
insan-gibi "muallak" sorular için en doğal malzeme. Fihrist sayfa numaraları
cilt-genelidir, tutanak PDF sayfasıyla birebir örtüşmez — golden_builder'da
sayfa işaretleme yine retrieval sonuçları üzerinden yapılmalı.

**Tohumlama bulgusu (embedding zorluk sinyali):** 50 sorunun 44'ünde beklenen
birleşim tutanağı ilk 30 sonuçta bulundu (çoğu rank#1–3). 6 soruda beklenen
belge ilk 30'a hiç girmedi (çapraz belgelerden tohumlandı, elle doğrulanmalı):
`34-402` Rakka'daki belediye araçları, `35-402` dünya şampiyonu voleybol takımı,
`49-402` Sedat Peker'in tutuklanmaması, `82-401` aşı reddi, `98-401` Kırkpınar
başpehlivan/Devlet Sporcusu, `100-401` Leyla Güven'in yemini. Bu altılı,
embedding'in eş-anlam köprüsünü kuramadığı örnekler olarak ayrıca değerli.
