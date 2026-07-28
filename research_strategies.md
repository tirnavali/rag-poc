# Araştırma Stratejisi Playbook'u

Bu dosya, agent planlayıcısının sorgu arketipine göre seçtiği araştırma
stratejilerini tanımlar. Her strateji bir `## ad` başlığı altında, basit
`key: value` alanlarıyla tanımlanır. Yeni bir strateji eklemek için aşağıya
yeni bir `## ad` bölümü eklemek yeterlidir — kod değişikliği gerekmez.

Alanlar:
- `triggers`: virgülle ayrılmış tetikleyici kelimeler (planner'a ipucu; boş
  bırakılabilir — o zaman strateji yalnızca LLM'in kendi yargısıyla seçilir).
- `query_type`: mevcut retrieval bütçe/döngü preset'lerinden biri
  (fact | summary | comparison | reasoning | policy | comprehensive).
- `answer_directive`: seçilen sistem promptunun üstüne eklenecek ek yönerge
  (boş bırakılırsa yanıt tarzı değişmez, yalnızca query_type uygulanır).

Çok-adımlı (multi-hop) stratejiler için opsiyonel alanlar — şimdilik parse edilip
`get_strategy()` ile sunulur, reflect adımı bağlanınca tüketilir:
- `mode`: `static` (varsayılan) | `adaptive`. `adaptive`, turlar arası bir varlık
  taşıyan (ör. önce kimlik çöz, sonra pivot et) prosedürel stratejiyi işaretler.
- `max_rounds`: bu stratejiye özel genişletme turu tavanı (tam sayı).
- `anchor`: önce çözülecek kesin kimlik/varlık (ör. `esas_no`).
- `target`: aranan nihai cevabın tanınır şekli (ör. `oy_dokumu`).
- `section_type`: stratejinin hedeflediği belge bölümü (SECTION_TYPES enum'u:
  `oylama` | `kanun_gorusmeleri` | `kanun_raporu` | `yazili_soru` | …). Planlamada
  `extracted_anchors`'a tohumlanır; reflect kanun kimliğini (sıra sayısı) çözünce
  orchestrator kesin `{sira_sayisi ∧ section_type}` where-filtresini kurar → o kanunun
  yalnız ilgili bölgesi getirilir (çapraz-kanun sızıntısı yok). Geçersiz değer yok sayılır.
- `exclude_seen_chunks`: `true` ise reflect hop'u, o ana dek havuza girmiş TAM chunk
  id'lerini o turun sıralanmış aday kümesinden düşürür (post-rank; ChromaDB `where`
  DEĞİL — chunk id metadata'da indekslenmez), böylece fetch_k yalnızca YENİ chunk'larla
  dolar. Kasıtlı olarak birleşim (`session`) düzeyinde DEĞİL: bir birleşimin bazı
  chunk'ları görüldü diye tüm birleşimi dışlamak, o oturumda henüz ulaşılmamış hedefi
  (ör. kimliksiz açık-oylama döküm tablosu) kalıcı olarak karartabilir. Bkz.
  `orchestrator._seen_chunk_ids` / `_run_retrieval(exclude_ids=…)`.
- `aliases`: `;` ile ayrılmış `halk_dili -> arşiv_terimi` çiftleri; sorgu hangi
  kelimeyle gelirse gelsin doğru arşiv terimine köprü kurar (term-hypothesis tohumu).
- `evidence_priority_patterns`: `;` ile ayrılmış, büyük/küçük harf duyarsız alt-dizgiler.
  Reflect LLM'ine verilen kompakt kanıt penceresi rerank-sıralı ilk N chunk'tır
  (`reflect.evidence_max_chunks`); prosedürün hedef kaydı (ör. açık-oylama duyurusu)
  tartışma/rapor chunk'larının altında kalıp pencereye hiç girmeyebilir — o zaman
  DURMA koşulu tetiklenemez ve döngü ancak dedup-fallback ile ölür. Bu desenlerden
  birini içeren chunk'lar pencerenin BAŞINA alınır (grup içi sıra korunur).
- `procedure`: serbest metin çok-adımlı reçete (reflect adımı okur).

## enumerate
triggers: tüm, hepsi, listele, kaç tane, hangileri
query_type: comprehensive
answer_directive: Eldeki TÜM ilgili kayıtları madde madde topla; kısmi kanıttan
  bile sentezle, eksikleri dürüstçe belirt, asla boş dönme.

## summarize
triggers: özetle, özet, kısaca
query_type: summary
answer_directive: 3–5 ana bulguyu öz ve yapılandırılmış biçimde ver; en yetkili
  kaynakları önceliklendir.

## comparative
triggers: karşılaştır, farkı, kıyasla
query_type: comparison
answer_directive: 2+ pozisyonu/dönemi karşıt köşelerde ele al; benzerlik ve
  farkları belirginleştir.

## analytical
triggers: neden, nasıl, analiz, değerlendir
query_type: reasoning
answer_directive: Olayları/alıntıları nedensellikle ilişkilendir; derin, gerekçeli
  sentez yap.

## factual
triggers:
query_type: fact
answer_directive:

## kanun_kabul_oylama
triggers: kaç oyla, kabul edildi, oylama sonucu, oy çokluğu, ret oyu, çekimser, madde kabul
query_type: reasoning
mode: adaptive
max_rounds: 4
anchor: esas_no
target: kabul_kaydi
section_type: oylama
exclude_seen_chunks: true
window_expand: true
aliases: genel gerekçe -> sıra sayısı raporu
evidence_priority_patterns: oylama sonucunu duyuruyorum; kullanılan oy; kabul edenler
answer_directive: İKİ OLASILIĞI AYIR. (1) Açık oylama yapıldıysa "<kanun adı> açık
  oylama sonucunu duyuruyorum: Kullanılan oy / Kabul / Ret / Çekimser" duyurusundaki
  SAYILARI ver. (2) İşaretle (el kaldırarak) geçtiyse kayıt "Kabul edenler… Etmeyenler…
  kabul edilmiştir" olur ve SAYI İÇERMEZ — bu durumda "işaretle kabul edildi; tutanakta
  sayısal döküm yok" de, ASLA sayı uydurma. Duyuru/kayıt SORULAN kanunu adıyla anmalı;
  aynı oturumdaki BAŞKA kanunun oylamasını bu kanuna ATFETME. Soru maddeye ilişkinse o
  maddenin, tümüne ilişkinse kanunun kabul kaydını göster.
procedure:
  BAĞLAM — kanun yapım süreci: (1) teklif TBMM Başkanlığına sunulur (esas no 2/N alır);
  (2) komisyonlara havale edilir, esas komisyon raporu SIRA SAYISI olarak bastırılır;
  (3) Genel Kurulda sıra sayısı üzerinden görüşülür — önce tümü üzerinde görüşme, sonra
  maddeler tek tek oylanır, EN SONDA kanunun TÜMÜ oylanır: aranan kabul kaydı görüşme
  bölgesinin SONUNDADIR; (4) kabul edilen kanun Cumhurbaşkanına gider — yayım/Resmî Gazete
  aşaması TUTANAKLARDA YER ALMAZ, "kabul edildi mi / kanunlaştı mı" sorusunun bu arşivdeki
  kanıtı Genel Kurul kabul kaydıdır (yayım kaydı arama). Cumhurbaşkanı geri gönderirse kanun
  Genel Kurulda YENİDEN görüşülür — tutanakta ikinci bir görüşme/oylama kaydı olabilir.
  Hop 1 — ÇIPA: Kanun adı + rapor bağlamını semantik ara; dönen chunk metninden esas no'yu
  ({tür}/{no}, tür 1=tasarı 2=teklif) okuyarak çıkar (ham id'yi sorgu yapma). ESAS NO BİÇİMİ:
  KANUN TEKLİFLERİ 2/N biçiminde (ör. 2/773), KANUN TASARILARI 1/N biçiminde esas numarası
  alır. 27. Dönem'den itibaren (2018, Cumhurbaşkanlığı hükümet sistemi) hükümet tasarısı
  kalktığından bu arşivdeki kanunlar TEKLİF olarak gelir → esas no hemen daima 2/N biçiminde
  beklenir; 1/N görürsen büyük olasılıkla eski dönem tasarısıdır. Esas no metinde
  "Kanun Teklifi (2/773)" / "Kanun Tasarısı (1/N)" ya da yalnızca "(2/773)" biçiminde parantez
  içinde de anılabilir — bu biçimleri de esas no olarak tanı.
  Hop 2 — DOĞRULA: Esas no + kanun adının aynı chunk'ta geçtiğini gör. Sıra sayısı/esas no
  numaralaması her yasama DÖNEMİ başında 1'den yeniden başlar (yasama YILI'nda değil) — aynı
  numara yalnızca FARKLI dönemlerde tekrar edebilir, aynı dönem içindeki farklı yıllarda
  tekildir. Çakışma varsa önce KANUN ADI benzerliği, eşitlikte dönem ile seç.
  Hop 3 — SONUÇ: Kanunun tümünün oylandığı yeri bul. ÖNCE açık-oylama duyurusunu ara
  ("<kanun adı> açık oylama sonucunu duyuruyorum … Kullanılan oy / Kabul / Ret"); yoksa
  işaretle kaydını ("teklifin/kanunun tümünü oylarınıza sunuyorum: Kabul edenler… kabul
  edilmiştir"). Bulduğun kaydın SORULAN kanuna ait olduğunu adıyla doğrula.
  DURMA: (a) sorulan kanunla adı EŞLEŞEN açık-oylama sayıları bulundu → sayıları ver, dur;
  VEYA (b) o kanunun işaretle kabul kaydı bulundu → "sayısal döküm yok, işaretle kabul" de,
  dur. Sayı yoksa var sanıp boşuna genişletme — çoğu kanun işaretle geçer; sayısal oy yalnızca
  açık-oylamaya tâbi (İçtüzük md.91 temel kanun vb.) kanunlarda bulunur.

## kanun_gorusmeleri
triggers: kanun üzerinde görüşmeler, teklif üzerinde görüşmeler, tüm görüşmeler, kanun üzerine görüş, teklif üzerine görüş, madde hakkında görüşme, maddeye muhalefet, sıra sayısı görüşmeleri
query_type: comprehensive
mode: adaptive
max_rounds: 4
anchor: esas_no + sıra_sayısı
target: gorus_kumesi
section_type: kanun_gorusmeleri
answer_directive: Görüşleri duruşa göre kümele (lehte / aleyhte / grup grup); grup adına ve
  şahsı adına konuşmaları, önerge gerekçelerini, muhalefet şerhini dahil et; konuşmacı ve
  grubunu belirt. Kaynak birleşim tutanağındaki görüşme (konuşma) bölümü olmalı — sıra
  sayısı raporundaki YAZILI gerekçe değil (o kanun_rapor_bolumu'nun işi).
procedure:
  BAĞLAM — kanun yapım süreci: teklif Başkanlığa sunulur (esas no 2/N) → komisyonlara havale →
  esas komisyon raporu sıra sayısı olarak bastırılır → GENEL KURUL görüşmesi → Cumhurbaşkanına.
  Bu stratejinin hedeflediği görüşmeler 3. aşamadır (Genel Kurul). KOMİSYONDAKİ görüşme
  tutanakları bu arşivde YOKTUR — komisyon aşamasından arşive yansıyan tek şey sıra sayısı
  RAPORUdur (birleşim tutanağının ekinde); komisyon tartışması sorulursa ancak rapordaki
  yazılı izler (muhalefet şerhi, ek görüş) gösterilebilir.
  KAPSAM: Soru "tüm görüşmeler / kanun üzerine" ise TÜM-KANUN; belirli "X. madde" ise MADDE
  kapsamı seç.
  Hop 1 — ÇIPA: Kanun adından esas no + sıra sayısını çöz (rapor/gündem chunk'ı ad+numarayı
  birlikte taşır). Madde kapsamıysa madde_no'yu da sorudan oku. ESAS NO BİÇİMİ: KANUN
  TEKLİFLERİ 2/N (ör. 2/773), KANUN TASARILARI 1/N esas numarası alır; 27. Dönem'den itibaren
  hükümet tasarısı kalktığından bu arşivde esas no hemen daima 2/N'dir. Esas no metinde
  "Kanun Teklifi (2/773)" / "Kanun Tasarısı (1/N)" ya da yalnızca "(2/773)" biçiminde parantez
  içinde de anılabilir — bu biçimleri de esas no olarak tanı. Sıra sayısı numaralaması her yasama DÖNEMİ
  başında sıfırlanır (yıl başında DEĞİL) — aynı numara farklı dönemlerde tekrar edebilir; kanun
  adıyla ve gerekirse dönemle doğrula.
  Hop 2 — LOKALİZE: Görüşme aralığını bul. İşlemler kanunu çoğu kez ADIYLA DEĞİL "N sıra
  sayılı" numarasıyla anar → sıra sayısı/esas no'yu extracted_anchors'a OKU; sistem
  `{sira_sayisi ∧ section_type=kanun_gorusmeleri}` kesin where-filtresini kurar ve o kanunun
  yalnız görüşme bölgesini getirir (numarayı sorgu metnine de göm — güvence).
  Sınır işaretleri: TÜM-KANUN'da "N sıra sayılı … görüşmelerine başlıyoruz" → nihai oylama;
  MADDE'de "X. maddeyi okutuyorum" → "X. madde kabul edilmiştir".
  Hop 3 — TOPLA: Aralıktaki tüm görüşleri topla; duruşa göre kümele.
  DURMA: Görüşme aralığı tükenene kadar topla; önceki sorguları tekrar etme, farklı grup/
  duruş açılarıyla genişlet.

## kanun_rapor_bolumu
triggers: genel gerekçe, madde gerekçesi, gerekçesi nedir, muhalefet şerhi, karşı oy yazısı, komisyon raporu, sıra sayısı raporu, tali komisyon, esas komisyon, ek görüş, kabul edilen metin
query_type: summary
mode: adaptive
max_rounds: 3
anchor: esas_no + sıra_sayısı
target: rapor_bolumu
section_type: kanun_raporu
aliases: genel gerekçe -> sıra sayısı raporu genel gerekçe bölümü
answer_directive: İstenen RAPOR bölümünü ver — bu SÖZLÜ görüş DEĞİL, sıra sayısı raporundaki
  YAZILI metindir. Bölümü ayırt et: genel gerekçe (kanunun bütününün gerekçesi), madde gerekçesi
  (ilgili maddenin taslak gerekçesi), esas komisyon raporu (kanunu asıl inceleyen komisyonun
  raporu — kullanıcı belirtmezse VARSAYILAN budur), tali komisyon raporu (varsa, görüşüne
  başvurulan İKİNCİL komisyonun raporu — esas komisyon raporuyla KARIŞTIRMA), rapora ek görüşler
  (raporun sonucuna katılan ama ek açıklama/çekince ekleyen, MUHALEFET OLMAYAN görüş — muhalefet
  şerhiyle KARIŞTIRMA), muhalefet şerhi (raporun aksine görüş bildiren yazı), komisyonun kabul
  ettiği metin (komisyon görüşmeleri sonunda şekillenen NİHAİ madde metni — teklifin ilk/orijinal
  metniyle KARIŞTIRMA). Bölümün SORULAN kanuna ait olduğunu esas no/sıra sayısı ile doğrula;
  başka kanunun gerekçesini atfetme.
procedure:
  BAĞLAM: Kanun yapım süreci: teklif TBMM Başkanlığına sunulur (esas no 2/N alır) → esas/tali
  komisyonlara havale edilir → esas komisyon raporu SIRA SAYISI olarak bastırılır → Genel Kurul
  görüşmesi → Cumhurbaşkanına (yayım aşaması tutanak dışıdır). Bu strateji 2. aşamanın YAZILI
  çıktısını (sıra sayısı raporunu) hedefler. Sıra sayısı, TBMM Başkanlığınca bastırılıp
  milletvekillerine dağıtılan komisyon raporu
  numarasıdır; her yasama DÖNEMİ başında (yasama YILI'nda değil) 1'den yeniden başlar — aynı
  numara yalnızca FARKLI dönemlerde tekrar edebilir. Kanun teklifine ilişkin bir sıra sayısı
  sırasıyla şunları taşır: tekliflerin gerekçesi ve metni, (varsa) tali komisyon raporu, esas
  komisyon raporu, (varsa) rapora ek görüşler, muhalefet şerhleri, komisyonun kabul ettiği metin.
  (Meclis araştırması/soruşturması komisyonlarına ilişkin sıra sayıları FARKLI yapıdadır — kuruluş
  önergeleri + komisyon raporu + varsa ek görüş/muhalefet şerhi, gerekçe/metin/kabul edilen metin
  YOKTUR; bu strateji yalnız KANUN sıra sayıları içindir.)
  Hop 1 — ÇIPA: Kanun adından esas no + sıra sayısını çöz. Sorudan hangi BÖLÜM istendiğini
  belirle (genel_gerekce | madde_gerekce[+madde_no] | esas_komisyon_raporu | tali_komisyon_raporu
  | ek_gorus | muhalefet_serhi | kabul_edilen_metin). ESAS NO BİÇİMİ: KANUN TEKLİFLERİ 2/N
  (ör. 2/773), KANUN TASARILARI 1/N esas numarası alır; 27. Dönem'den itibaren hükümet tasarısı
  kalktığından bu arşivde esas no hemen daima 2/N'dir. Esas no metinde "Kanun Teklifi (2/773)" /
  "Kanun Tasarısı (1/N)" ya da yalnızca "(2/773)" biçiminde parantez içinde de anılabilir — bu
  biçimleri de esas no olarak tanı. Aynı numara farklı dönemde de görülebilir — kanun adıyla ve
  gerekirse dönemle doğrula.
  Hop 2 — RAPORU BUL: O kanunun sıra sayısı raporunu bul; sıra sayısı/esas no'yu
  extracted_anchors'a OKU → sistem `{sira_sayisi ∧ section_type=kanun_raporu}` kesin
  filtresini kurar. Bölümler kendi başlıklarıyla geçer: "GENEL GEREKÇE", "MADDE <n>-",
  "MUHALEFET ŞERHİ"; tali komisyon raporu VARSA kendi komisyon adıyla ("… KOMİSYONU RAPORU")
  esas komisyon raporundan ÖNCE ayrı bir başlık altında yer alır — esas komisyonunki metnin
  asıl gövdesidir.
  Hop 3 — BÖLÜMÜ ÇIKAR: İstenen başlığın altındaki metni getir; madde gerekçesiyse "MADDE <n>"
  başlığının gerekçe paragrafını, kabul edilen metin isteniyorsa raporun sonundaki komisyonca
  kabul edilmiş nihai madde metnini getir.
  DURMA: İstenen bölüm sorulan kanun için bulununca dur. NOT: "madde gerekçesi" (yazılı rapor)
  ≠ "madde hakkındaki görüşler" (sözlü) → görüş isteniyorsa kanun_gorusmeleri stratejisi.
