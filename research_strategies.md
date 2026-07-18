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
- `aliases`: `;` ile ayrılmış `halk_dili -> arşiv_terimi` çiftleri; sorgu hangi
  kelimeyle gelirse gelsin doğru arşiv terimine köprü kurar (term-hypothesis tohumu).
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
target: oy_dokumu
aliases: genel gerekçe -> sıra sayısı raporu
answer_directive: Oy sayılarını (kabul/ret/çekimser) esas no ve kanun adıyla
  ilişkilendirerek ver; kaynak birleşim tutanağındaki oylama bölümü olmalı,
  sıra sayısı raporu değil. Soru bir maddeye ilişkinse o maddenin, kanunun
  tümüne ilişkinse kanunun kabul kaydını göster.
procedure:
  Hop 1 — ÇIPA (esas no'yu ARAMA, ÇIKAR): Kanun adı + rapor bağlamını semantik
  ara ("<kanun adı> sıra sayısı genel gerekçe esas no"). Dönen chunk METNİNDEN
  esas no'yu ({tür}/{no}, tür 1=tasarı 2=teklif) okuyarak çıkar; ham id'yi
  ("1/234") sorgu terimi yapma — embedding kesin id'de zayıf, id'yi metinden türet.
  Hop 2 — DOĞRULA/AYRIŞTIR (sıra sayısı raporu): Esas no + kanun adının aynı
  chunk'ta birlikte geçtiğini gör; rapor bölümü ikisini + maddeleri taşır.
  Aynı esas no birden çok yıla aitse önce KANUN ADI benzerliği, eşitlikte
  TARİH/DÖNEM ile seç.
  Hop 3 — HEDEF (oylama sonucu): Doğrulanan esas no'yu + kanun adını taşıyarak
  birleşimdeki "açık oylama sonucu" bölümünü ara; önceki sorguları tekrar etme.
  DURMA: Kanunun kabulü sorulduysa "kanunun tümü … kabul edilmiştir" + oy dökümü;
  belirli madde sorulduysa "<n>. madde … kabul edilmiştir" + o maddenin oyu.
  Kayıt esas no ile ilişkilendirilince dur, yoksa max_rounds'a kadar genişlet.
