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
