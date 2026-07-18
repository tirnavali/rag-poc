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
target: kabul_kaydi
aliases: genel gerekçe -> sıra sayısı raporu
answer_directive: İKİ OLASILIĞI AYIR. (1) Açık oylama yapıldıysa "<kanun adı> açık
  oylama sonucunu duyuruyorum: Kullanılan oy / Kabul / Ret / Çekimser" duyurusundaki
  SAYILARI ver. (2) İşaretle (el kaldırarak) geçtiyse kayıt "Kabul edenler… Etmeyenler…
  kabul edilmiştir" olur ve SAYI İÇERMEZ — bu durumda "işaretle kabul edildi; tutanakta
  sayısal döküm yok" de, ASLA sayı uydurma. Duyuru/kayıt SORULAN kanunu adıyla anmalı;
  aynı oturumdaki BAŞKA kanunun oylamasını bu kanuna ATFETME. Soru maddeye ilişkinse o
  maddenin, tümüne ilişkinse kanunun kabul kaydını göster.
procedure:
  Hop 1 — ÇIPA: Kanun adı + rapor bağlamını semantik ara; dönen chunk metninden esas no'yu
  ({tür}/{no}, tür 1=tasarı 2=teklif) okuyarak çıkar (ham id'yi sorgu yapma).
  Hop 2 — DOĞRULA: Esas no + kanun adının aynı chunk'ta geçtiğini gör; aynı esas no birden
  çok yıla aitse önce KANUN ADI benzerliği, eşitlikte tarih/dönem ile seç.
  Hop 3 — SONUÇ: Kanunun tümünün oylandığı yeri bul. ÖNCE açık-oylama duyurusunu ara
  ("<kanun adı> açık oylama sonucunu duyuruyorum … Kullanılan oy / Kabul / Ret"); yoksa
  işaretle kaydını ("teklifin/kanunun tümünü oylarınıza sunuyorum: Kabul edenler… kabul
  edilmiştir"). Bulduğun kaydın SORULAN kanuna ait olduğunu adıyla doğrula.
  DURMA: (a) sorulan kanunla adı EŞLEŞEN açık-oylama sayıları bulundu → sayıları ver, dur;
  VEYA (b) o kanunun işaretle kabul kaydı bulundu → "sayısal döküm yok, işaretle kabul" de,
  dur. Sayı yoksa var sanıp boşuna genişletme — çoğu kanun işaretle geçer; sayısal oy yalnızca
  açık-oylamaya tâbi (İçtüzük md.91 temel kanun vb.) kanunlarda bulunur.

## kanun_gorusmeleri
triggers: üzerinde görüşmeler, tüm görüşmeler, ne konuşuldu, kanun üzerine görüş, madde hakkında, maddeye muhalefet, eleştiri, ne dedi
query_type: comprehensive
mode: adaptive
max_rounds: 4
anchor: esas_no + sıra_sayısı
target: gorus_kumesi
answer_directive: Görüşleri duruşa göre kümele (lehte / aleyhte / grup grup); grup adına ve
  şahsı adına konuşmaları, önerge gerekçelerini, muhalefet şerhini dahil et; konuşmacı ve
  grubunu belirt. Kaynak birleşim tutanağındaki görüşme (konuşma) bölümü olmalı — sıra
  sayısı raporundaki YAZILI gerekçe değil (o kanun_rapor_bolumu'nun işi).
procedure:
  KAPSAM: Soru "tüm görüşmeler / kanun üzerine" ise TÜM-KANUN; belirli "X. madde" ise MADDE
  kapsamı seç.
  Hop 1 — ÇIPA: Kanun adından esas no + sıra sayısını çöz (rapor/gündem chunk'ı ad+numarayı
  birlikte taşır). Madde kapsamıysa madde_no'yu da sorudan oku.
  Hop 2 — LOKALİZE: Görüşme aralığını bul. İşlemler kanunu çoğu kez ADIYLA DEĞİL "N sıra
  sayılı" numarasıyla anar → aralığı sıra sayısı/esas no ile daralt (metadata omurgası
  gelince sıra_sayısı FİLTRESİ kesin sonuç verir; şimdilik numara+konu semantik araması).
  Sınır işaretleri: TÜM-KANUN'da "N sıra sayılı … görüşmelerine başlıyoruz" → nihai oylama;
  MADDE'de "X. maddeyi okutuyorum" → "X. madde kabul edilmiştir".
  Hop 3 — TOPLA: Aralıktaki tüm görüşleri topla; duruşa göre kümele.
  DURMA: Görüşme aralığı tükenene kadar topla; önceki sorguları tekrar etme, farklı grup/
  duruş açılarıyla genişlet.

## kanun_rapor_bolumu
triggers: genel gerekçe, madde gerekçesi, gerekçesi nedir, muhalefet şerhi, karşı oy yazısı, komisyon raporu, sıra sayısı raporu
query_type: summary
mode: adaptive
max_rounds: 3
anchor: esas_no + sıra_sayısı
target: rapor_bolumu
aliases: genel gerekçe -> sıra sayısı raporu genel gerekçe bölümü
answer_directive: İstenen RAPOR bölümünü ver — bu SÖZLÜ görüş DEĞİL, sıra sayısı raporundaki
  YAZILI metindir. Bölümü ayırt et: genel gerekçe (kanunun bütününün gerekçesi), madde
  gerekçesi (ilgili maddenin taslak gerekçesi), muhalefet şerhi (karşı görüş yazısı). Bölümün
  SORULAN kanuna ait olduğunu esas no/sıra sayısı ile doğrula; başka kanunun gerekçesini atfetme.
procedure:
  Hop 1 — ÇIPA: Kanun adından esas no + sıra sayısını çöz. Sorudan hangi BÖLÜM istendiğini
  belirle (genel_gerekce | madde_gerekce[+madde_no] | muhalefet_serhi).
  Hop 2 — RAPORU BUL: O kanunun sıra sayısı raporunu bul; rapor da kanunu numarasıyla anar
  (metadata omurgası gelince sıra_sayısı filtresi kesin). Bölümler kendi başlıklarıyla geçer:
  "GENEL GEREKÇE", "MADDE <n>-", "MUHALEFET ŞERHİ".
  Hop 3 — BÖLÜMÜ ÇIKAR: İstenen başlığın altındaki metni getir; madde gerekçesiyse "MADDE <n>"
  başlığının gerekçe paragrafını.
  DURMA: İstenen bölüm sorulan kanun için bulununca dur. NOT: "madde gerekçesi" (yazılı rapor)
  ≠ "madde hakkındaki görüşler" (sözlü) → görüş isteniyorsa kanun_gorusmeleri stratejisi.
