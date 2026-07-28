"""Kanun bölgesi etiketleyici birim testleri (offline, chroma'sız).

Sentetik sıralı chunk-metin listeleri + gerçek ``tutanak-27-02-06-20181010``
diliminden küçük fixture ile durum makinesini doğrular: görüşme→roll-call→terminatör
bölgesinde N/esas doğru yayılıyor mu, ön-materyal bölge başlatmıyor mu, belirsiz
(çok-N) chunk bölge değiştirmiyor mu, OCR/tarih/kanun-no tuzakları es geçiliyor mu.
"""
from src.trainer.ingestion.law_region_tagger import tag_law_regions, tag_sections


def _sira(tags):
    return [t["sira_sayisi"] for t in tags]


def _sec(tags):
    return [t["section_type"] for t in tags]


def test_basic_region_propagates_to_identityless_rollcall():
    """Görüşme açıcısı → oy duyurusu → kimliksiz roll-call satırları → terminatör.
    Kimliksiz tablo satırları da bölge yaymasıyla sira_sayisi + esas_no alır (ASIL amaç)."""
    chunks = [
        # ön-materyal: gündem indexi (çok-esas, görüşme yok) → bölge BAŞLATMAZ
        "## II.- GELEN KÂĞITLAR\nTeklifler: (2/900) (2/901) (2/902)",
        # güçlü açıcı: görüşme-başla + tek N=5 + tek esas 2/773
        "5 sıra sayılı Türkiye Kalkınma Bankası Anonim Şirketi Hakkında Kanun "
        "Teklifi (2/773) (S. Sayısı: 5) görüşmelerine başlıyoruz.",
        # görüşme metni (çıpasız) → ileri-yayma 5
        "Sayın Başkan, değerli milletvekilleri; bu teklif hakkında söz almak istiyorum.",
        # açık oy duyurusu
        "5 sıra sayılı Kanun Teklifi'nin açık oylama sonucu: Kullanılan oy sayısı: 247",
        # roll-call satırları — KİMLİK YOK → ileri-yayma ile 5/2-773 almalı
        "| Antalya | Uslu | Atay | AK PARTİ | Kabul |",
        "| İstanbul | Özdemir | sibel | CHP | Red |",
        # terminatör: farklı gündem işi → bölge kapanır
        "## X.- YAZILI SORULAR VE CEVAPLARI\nŞırnak Milletvekili Hüseyin Kaçmaz'ın sorusu",
        # terminatör sonrası → None
        "Aşağıdaki sorularımın cevaplandırılmasını saygılarımla arz ederim.",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [None, 5, 5, 5, 5, 5, None, None]
    # Kimliksiz roll-call satırları esas_no'yu haritadan aldı:
    assert tags[4]["esas_no"] == "2/773"
    assert tags[5]["esas_no"] == "2/773"
    assert "Kalkınma" in (tags[1]["kanun_adi"] or "")


def test_prematerial_index_does_not_start_region():
    """İÇİNDEKİLER'in 'VIII.- KANUN TEKLİFLERİ' TOC satırı (çıplak kanun-işi başlığı)
    ilk güçlü açıcıdan ÖNCE bölge açmaz — yalnız haritaya katkı verir."""
    chunks = [
        # TOC: kanun-işi başlığı ama görüşme/kapak/oy-başlığı YOK, started_ever False
        "## VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN DİĞER İŞLER\n"
        "1.- ... Kanun Teklifi (2/773) (S. Sayısı: 5) ... Sayfa 12",
        "## I.- GEÇEN TUTANAK ÖZETİ\nBir önceki birleşimin özeti.",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [None, None]


def test_law_heading_reanchors_only_after_started():
    """Çıplak kanun-işi başlığı (Devam) yalnız zaten bir bölge açılmışsa yeniden-çıpalar."""
    chunks = [
        # güçlü açıcı (görüşme-başla) bölgeyi açar
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",
        # AÇIKLAMALAR (Devam) terminatör DEĞİL → ileri-yayma 5 (görüşme-içi araya-girme)
        "## V.- AÇIKLAMALAR (Devam)\nSayın Başkan bir açıklama yapmak istiyorum.",
        # çıplak kanun-işi başlığı (Devam) + tek N → yeniden-çıpa 5
        "## VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN DİĞER İŞLER (Devam)\n"
        "(S. Sayısı: 5) görüşmelere devam ediyoruz.",
        "Değerli milletvekilleri oylamaya geçiyoruz.",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [5, 5, 5, 5]


def test_multi_n_chunk_does_not_switch_region():
    """Bir chunk iki farklı N içeriyorsa (belirsiz) bölge DEĞİŞTİRMEZ, current korunur."""
    chunks = [
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",
        # çıplak kanun-işi başlığı ama iki N (4 ve 5) → açıcı değil, 5 korunur
        "## VIII.- KANUN TEKLİFLERİ\n4 sıra sayılı ile 5 sıra sayılı teklifler gündemde.",
        "Görüşmeye devam.",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [5, 5, 5]


def test_esas_reverse_resolution_when_opener_lacks_explicit_number():
    """Görüşme açıcısı yalnız esas taşıyıp açık sıra sayısı taşımazsa (gerçek _603
    örneği), N esas→N ters haritasından çözülür."""
    chunks = [
        # harita kaynağı: tek N=5 + tek esas 2/773 (açıcı değil — görüşme/başlık yok)
        "Kanun Teklifi (2/773) ile Plan ve Bütçe Komisyonu Raporu (S. Sayısı: 5)",
        # açıcı yalnız esas taşır (açık N yok) + görüşme-başla → ters harita ile N=5
        "Türkiye Kalkınma Bankası Anonim Şirketi Hakkında Kanun Teklifinin (2/773) "
        "görüşmelerine başlıyoruz.",
        "| Ankara | Akdoğan | Yakçı | AK PARTİ | Kabul |",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [None, 5, 5]
    assert tags[1]["esas_no"] == "2/773"
    assert tags[2]["esas_no"] == "2/773"


def test_report_cover_is_strong_opener_from_scratch():
    """Rapor kapağı '## SIRA SAYISI: N' sıfırdan bölge açar (saf rapor PDF senaryosu)."""
    chunks = [
        "## SIRA SAYISI: 7\nBir Kanun Teklifi (2/860) ile Komisyon Raporu",
        "GENEL GEREKÇE\nBu teklif ile amaçlanan düzenleme ...",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [7, 7]
    assert tags[0]["esas_no"] == "2/860"


def test_date_and_law_number_traps_are_ignored():
    """Gerekçe metnindeki '4/11/1983 tarihli' (tarih) esas no sanılmaz; '13 sayılı
    Kanun' (kanun no) sıra sayısı sanılmaz — bölge önceki açıcıdan gelen 5'te kalır."""
    chunks = [
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",
        # tarih + kanun-no tuzakları: hiçbiri çıpa değil → ileri-yayma 5, esas 2/773 sabit
        "Banka 4/11/1983 tarihli ve 165 sayılı KHK ile, 22/6/1988 tarihli ve 13 "
        "sayılı Kanun Hükmünde Kararname ile yeniden düzenlenmiştir.",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [5, 5]
    assert tags[1]["esas_no"] == "2/773"  # 4/11 tarihine sapmadı


def test_bare_terminator_word_does_not_reset_forward_fill():
    """'kanunlaşmıştır' gibi çıplak sonlandırıcı sözcük (adsız) bölgeyi düşürmez —
    tagger yalnız başlık-rubriği terminatörlerini kullanır, ileri-yayma güvenle sürer."""
    chunks = [
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",
        "Böylece kanun teklifi kabul edilmiş ve kanunlaşmıştır.",
        "| Van | Arvas | Abdulahat | AK PARTİ | Kabul |",  # roll-call yine 5
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [5, 5, 5]


def test_empty_and_none_inputs_are_robust():
    assert tag_law_regions([]) == []
    tags = tag_law_regions(["", None])
    assert _sira(tags) == [None, None]


def test_real_slice_kalkinma_vote_region():
    """Gerçek ``tutanak-27-02-06-20181010`` diliminden sadeleştirilmiş fixture:
    görüşme başlığı (_969) → oy duyurusu (_970) → roll-call (_971.._977) → terminatör
    (_985). Kimliksiz tablo satırları 5/2-773 almalı; yazılı-soru bölümü almamalı."""
    chunks = [
        # _603 — görüşme başlangıcı (güçlü açıcı): yalnız esas 2/773 + görüşme-başla
        "## VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN DİĞER İŞLER\n"
        "Türkiye Kalkınma Bankası Anonim Şirketi Hakkında Kanun Teklifi (2/773) ile "
        "Plan ve Bütçe Komisyonu Raporunun görüşmelerine başlıyoruz.",
        # _604 — harita kaynağı: tek N=5 + tek esas 2/773 (görüşme boyunca yeniden-çıpa)
        "5 sıra sayılı Kanun Teklifi (2/773) üzerindeki görüşmeler sürüyor.",
        # _969 — görüşme (Devam) başlığı: kanun-işi başlık + (S. Sayısı: 5) + 5 sıra sayılı
        "## VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN DİĞER İŞLER (Devam)\n"
        "1.- ... Mehmet Muş'un Türkiye Kalkınma Bankası Anonim Şirketi Hakkında Kanun "
        "Teklifi (2/773) ile Plan ve Bütçe Komisyonu Raporu (S. Sayısı: 5) (Devam)\n"
        "BAŞKAN – bu şekilde, 5 sıra sayılı Teklif'in görüşmeleri tamamlanmıştır.",
        # _970 — açık oy duyurusu
        "BAŞKAN – 5 sıra sayılı Kanun Teklifi'nin açık oylama sonucu: "
        "\"Kullanılan oy sayısı : 247 Kabul",
        # _971 — roll-call başlığı + tablo başı (S. Sayısı: 5)
        "1,- (S. Sayısı: 5) Türkiye Kalkınma Bankası ... Verilen Oyların Sonucu: "
        "Kabul Edenler 218 Toplam 247 | Adana | Barut |",
        # _972,_973 — kimliksiz tablo satırları
        "| Amasya | Tuncer | Mustafa | CHP | Katılmadı |",
        "| Antalya | Uslu | Atay | AK PARTİ | Kabul |",
        # _977 — 's.s.5' işaretli tablo başlığı (OCR)
        "|  | Soyad |  | Parti | s.s.5 |\n| İstanbul | Bozkır | Volkan | AK PARTİ | Katılmadı |",
        # _985 — terminatör: yazılı sorular (farklı iş, esas 7/349)
        "## X.- YAZILI SORULAR VE CEVAPLARI\n1.- Şırnak Milletvekili ... sorusu (7/349)",
        # _986 — yazılı soru gövdesi
        "TÜRKİYE BÜYÜK MİLLET MECLİSİ BAŞKANLIĞINA Aşağıdaki sorularım ...",
    ]
    tags = tag_law_regions(chunks)
    assert _sira(tags) == [5, 5, 5, 5, 5, 5, 5, 5, None, None]
    for i in range(8):
        assert tags[i]["esas_no"] == "2/773", f"chunk {i} esas_no"


# ── tag_sections (bölüm etiketleyici — kardeş durum makinesi) ───────────────


def test_section_labels_every_chunk_and_propagates_to_rollcall():
    """Gelen kâğıt → görüşme → oy duyurusu → kimliksiz roll-call → yazılı-soru:
    HER chunk section_type alır; roll-call satırları ileri-yaymayla 'oylama' (ASIL
    amaç — reflect section_type=oylama filtresi kimliksiz tabloya ulaşsın)."""
    chunks = [
        # gelen kâğıtlar başlığı → gelen_kagit
        "## II.- GELEN KÂĞITLAR\nTeklifler: (2/900) (2/901) (2/902)",
        # güçlü açıcı: görüşme-başla (gövde) → kanun_gorusmeleri
        "5 sıra sayılı Türkiye Kalkınma Bankası Anonim Şirketi Hakkında Kanun "
        "Teklifi (2/773) (S. Sayısı: 5) görüşmelerine başlıyoruz.",
        # görüşme gövdesi (sinyal yok) → ileri-yayma kanun_gorusmeleri
        "Sayın Başkan, değerli milletvekilleri; bu teklif hakkında söz almak istiyorum.",
        # açık oy duyurusu (gövde) → oylama
        "5 sıra sayılı Kanun Teklifi'nin açık oylama sonucu: Kullanılan oy sayısı: 247",
        # kimliksiz roll-call satırları → ileri-yayma ile oylama almalı
        "| Antalya | Uslu | Atay | AK PARTİ | Kabul |",
        "| İstanbul | Özdemir | sibel | CHP | Red |",
        # terminatör başlığı → yazili_soru (kanun-tagger'da KAPATICI, burada AÇICI)
        "## X.- YAZILI SORULAR VE CEVAPLARI\nŞırnak Milletvekili Hüseyin Kaçmaz'ın sorusu",
        "Aşağıdaki sorularımın cevaplandırılmasını saygılarımla arz ederim.",
    ]
    tags = tag_sections(chunks)
    assert _sec(tags) == [
        "gelen_kagit", "kanun_gorusmeleri", "kanun_gorusmeleri",
        "oylama", "oylama", "oylama", "yazili_soru", "yazili_soru",
    ]
    # oylama breadcrumb'ı kapsayan görüşmeyi taşır (yalnız görüntü/debug):
    assert tags[3]["section_path"] == "kanun_gorusmeleri > oylama"
    # section_ord bölüm değiştikçe artan monoton int; aynı bölümde sabit:
    assert tags[4]["section_ord"] == tags[3]["section_ord"]
    assert tags[1]["section_ord"] < tags[3]["section_ord"] < tags[6]["section_ord"]


def test_vote_header_wins_over_deliberation_heading():
    """Aynı chunk hem oy DUYURUSU (gövde) hem kanun-işi başlığı taşırsa → 'oylama'
    (öncelik sırası: _VOTE_HEADER, _GORUSME'den önce)."""
    chunks = [
        # önce görüşme aç (strong_seen kapısı için)
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",
        # başlık kanun-işi ama gövdede açık oy sonucu → oylama kazanır
        "## VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN DİĞER İŞLER\n"
        "5 sıra sayılı Kanun Teklifi'nin açık oylama sonucu açıklanmıştır.",
    ]
    assert _sec(tag_sections(chunks)) == ["kanun_gorusmeleri", "oylama"]


def test_toc_headings_stay_icindekiler_until_strong_opener():
    """İÇİNDEKİLER + TOC rubrik-başlıkları (KANUN TEKLİFLERİ / YAZILI SORULAR) ilk
    GÜÇLÜ açıcıdan önce bölümü FLIP etmez → 'icindekiler' yapışır (parçalanma yok)."""
    chunks = [
        "## İÇİNDEKİLER",
        "## VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN DİĞER İŞLER  Sayfa 12",
        "## X.- YAZILI SORULAR VE CEVAPLARI  Sayfa 40",
        # ilk güçlü açıcı: gerçek görüşme → buradan itibaren kanun_gorusmeleri
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",
    ]
    assert _sec(tag_sections(chunks)) == [
        "icindekiler", "icindekiler", "icindekiler", "kanun_gorusmeleri",
    ]


def test_toc_exits_at_gecen_tutanak_then_flips_freely():
    """Aralıklı 'İ Ç İ N D E K İ L E R' başlığı yakalanır; TOC rubrik-listelemelerinde
    yapışır; ilk GEÇEN TUTANAK (birleşimin kanonik gerçek başlangıcı) TOC'u kapatır,
    ardından rubrikler SERBESTÇE akar (gerçek _27-02-06 belgesinin başı bu şablonda)."""
    chunks = [
        "## İ Ç İ N D E K İ L E R",                    # aralıklı başlık → icindekiler
        "## VII.- ÖNERİLER",                            # TOC listesi → bastırılır
        "## VIII.- KANUN TEKLİFLERİ / IX.- OYLAMALAR",  # TOC listesi (çoklu) → bastırılır
        "## I.- GEÇEN TUTANAK ÖZETİ",                   # GERÇEK başlangıç → gecen_tutanak
        "Önceki birleşimin özeti okundu.",              # sinyal yok → yayma
        "## II.- GELEN KÂĞITLAR",                       # serbest geçiş → gelen_kagit
    ]
    assert _sec(tag_sections(chunks)) == [
        "icindekiler", "icindekiler", "icindekiler",
        "gecen_tutanak", "gecen_tutanak", "gelen_kagit",
    ]


def test_page_furniture_inert_and_forward_fill_across_gaps():
    """'## Sayfa' mobilya başlığı + tanınmaz/OCR-bozuk başlık EYLEMSİZ (yaymayı bozmaz);
    Docling'e yalnız '# bastı' kadarıyla güvenilir — tanınmayan başlık yanlış etiketlemez."""
    chunks = [
        "5 sıra sayılı Kanun Teklifi (2/773) görüşmelerine başlıyoruz.",  # kanun_gorusmeleri
        "## Sayfa 42",                          # mobilya → eylemsiz
        "Görüşmelere devam ediyoruz efendim.",  # sinyal yok → yayma
        "## Bir Şey (OCR bozuk başlık glıph)",  # tanınmaz → eylemsiz
    ]
    assert _sec(tag_sections(chunks)) == ["kanun_gorusmeleri"] * 4


def test_terminator_recompose_preserves_law_tagger_matches():
    """_TERMINATOR alt-kümeden YENİDEN DERLENDİ — kanun-tagger'ın gördüğü tüm rubrik
    başlıkları hâlâ eşleşmeli (davranış-korunur guard'ı); yemin terminatör DEĞİL."""
    from src.trainer.ingestion.law_region_tagger import _TERMINATOR

    for s in [
        "YAZILI SORULAR", "SÖZLÜ SORU", "GÜNDEM DIŞI", "GELEN KÂĞITLAR",
        "GEÇEN TUTANAK ÖZETİ", "İÇİNDEKİLER", "ÖNERİLER", "SEÇİM",
        "GENEL GÖRÜŞME", "MECLİS ARAŞTIRMASI",
        "BAŞKANLIĞIN GENEL KURULA SUNUŞLARI", "TEZKERE",
    ]:
        assert _TERMINATOR.search(s), s
    assert not _TERMINATOR.search("YEMİN")  # yemin kanun-tagger terminatörü değil


def test_sections_empty_and_none_inputs_are_robust():
    assert tag_sections([]) == []
    tags = tag_sections(["", None])
    assert _sec(tags) == [None, None]
    assert tags[0]["section_ord"] is None
    assert tags[0]["section_path"] is None
