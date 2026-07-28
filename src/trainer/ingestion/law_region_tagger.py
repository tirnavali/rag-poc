"""Kanun bölgesi etiketleyici — tutanak chunk'larına ``sira_sayisi`` + ``esas_no``
metadata omurgasını koyan deterministik (LLM'siz) durum makinesi.

**Neden var:** TBMM açık-oylama roll-call tabloları (isim-isim "Kabul/Red/Katılmadı"
dökümü) ilgili kanunu ADIYLA da esas no'suyla da anmaz — yalnızca tablo başlığında
"S.S. 5" gibi bir işaret taşır, çoğu satırda o bile yoktur. Kanıt: Türkiye Kalkınma
Bankası (esas 2/773, sıra sayısı 5) açık-oylaması ``tutanak-27-02-06-20181010``
belgesinde ``_971``..``_984`` chunk'larında; kimliksiz olduğu için "2/773 açık oylama"
semantik araması bu chunk'lara ULAŞAMAZ. Bu etiketleyici ait olduğu kanunun sıra
sayısını (+ esas no'yu) her chunk'a metadata olarak koyar → Hop-2 kesin
``where={'sira_sayisi': 5}`` filtresi olur.

**Yaklaşım — iki geçiş:**
  1. **Harita (pass 1):** tüm belgeyi tarayıp ``N (sıra sayısı) → esas_no + kanun_adı``
     eşlemesini yalnızca GÜÇLÜ çıpalardan (tek-N + tek-esas taşıyan chunk) kur. Ters
     eşleme (``esas_no → N``) de tutulur: görüşme başlangıç chunk'ı çoğu kez esas no
     taşır ama açık sıra sayısı taşımaz.
  2. **Bölge ataması (pass 2):** chunk'ları sırayla gez, tek bir ``current`` sıra
     sayısı durumu tut. Bölge YALNIZCA güçlü açıcılarla başlar/değişir (kanun-işi
     bölüm başlığı / "görüşmelerine başlıyoruz" / rapor kapağı / roll-call başlığı, her
     biri tek çözülebilir N ile); kanun-dışı rubrik başlıkları (yazılı soru, gündem
     dışı, açıklamalar, yoklama, gelen kâğıtlar…) bölgeyi kapatır. Aradaki her chunk
     ``current`` ile ileri-yayılır → roll-call tablo satırları da etiketlenir.

**Tasarım ilkesi:** *Yanlış-etiketlemektense etiketlememek.* Belirsiz (çok-N) chunk'lar
bölge değiştirmez; ön-materyal (ilk güçlü açıcıdan önceki gündem/index) bölge başlatmaz;
emin olunmayan yerler ``None`` kalır (pipeline sanitizasyonu None'ı zaten düşürür).

Etiketleyici PAYLAŞILIR: hem backfill (mevcut korpus, ``scripts/backfill_law_metadata``)
hem ingest adaptörü (gelecek belgeler, ``adapters/tutanak_pdf``) aynı ``tag_law_regions``
fonksiyonunu çağırır → tek doğruluk kaynağı.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

# ── Sıra sayısı işaretleri ──────────────────────────────────────────────────
# "(S. Sayısı: 5)" / "(S.Sayısı:5)" — görüşme başlığı + roll-call başlığı
_SS_PAREN = re.compile(r"\(\s*S\.?\s*Say[ıi]s[ıi]\s*:?\s*(\d{1,4})\s*\)", re.IGNORECASE)
# "SIRA SAYISI: 5" — rapor kapağı başlığı (büyük harf)
_SS_COVER = re.compile(r"SIRA\s*SAYISI\s*:?\s*(\d{1,4})", re.IGNORECASE)
# "5 sıra sayılı" — gövde referansı (görüşme + oy duyurusu + gündem eki). "sıra"
# kelimesini ZORUNLU tutar → "13 sayılı Kanun", "4456 sayılı Kanun" gibi kanun-no
# atıflarını (gerekçe metninde bol) YANLIŞLIKLA sıra sayısı sanmaz.
_SS_BODY = re.compile(r"(\d{1,4})\s*[Ss][ıi]ra\s*[Ss]ay[ıi]l[ıi]", re.IGNORECASE)

# ── Esas no işaretleri (tarih DEĞİL) ────────────────────────────────────────
# Yalnızca parantezli "(2/773)" ya da "2/773 esas" biçimi → "4/11/1983" gibi üç
# parçalı tarihler hiçbir zaman eşleşmez (parantezsiz + üç-parça).
_ESAS_PAREN = re.compile(r"\((\d{1,3}/\d{1,4})\)")
_ESAS_KW = re.compile(r"(\d{1,3}/\d{1,4})\s*[Ee]sas")

# ── Bölge AÇICI işaretleri ──────────────────────────────────────────────────
# Kanun-işi bölüm başlığı: "VIII.- KANUN TEKLİFLERİ İLE KOMİSYONLARDAN GELEN…"
_LAW_HEADING = re.compile(
    r"KANUN\s+TEKL[İIi]F|KANUN\s+TASAR|KOM[İIi]SYONLARDAN\s+GELEN", re.IGNORECASE
)
# "…görüşmelerine başlıyoruz / başlanmıştır / devam"
_GORUSME = re.compile(
    r"görü[şs]me(?:ler)?ine\s+ba[şs]l|görü[şs]ülmesine\s+ba[şs]l|görü[şs]melerine\s+devam",
    re.IGNORECASE,
)
# Roll-call / açık oy sonuç başlığı: "…Verilen Oyların Sonucu", "açık oylama sonucu"
_VOTE_HEADER = re.compile(
    r"Verilen\s+Oyların|Oyların\s+Sonucu|aç[ıi]k\s+oylama\s+sonu", re.IGNORECASE
)

# ── Bölge KAPATICI (terminatör) rubrik başlıkları ───────────────────────────
# Yalnızca `#`-başlıklı satırlarda aranır → gövdedeki geçişleri (bir konuşmacının
# "yazılı soru önergesi" demesi) terminatör saymaz. Bunlar bir kanunun görüşmesini
# GERÇEKTEN sonlandıran, ayrı gündem işine geçiren üst-düzey rubrikler.
# KASITLI DIŞARIDA: AÇIKLAMALAR / YOKLAMA / OTURUM BAŞKANLARININ KONUŞMALARI gibi
# görüşme-İÇİ prosedürel araya-girmeler — bunlar bir kanunun görüşmesi SÜRERKEN olur;
# terminatör sayılırsa görüşme bölgesi delik deşik olur (roll-call yine güvende ama
# konuşma chunk'ları kaybolur). İleri-yayma onların üzerinden geçsin.
#
# Her terminatör rubriği AYRI derlenir ve kanonik ``section_type``'a eşlenir
# (``tag_sections`` — kardeş bölüm etiketleyici; kanun-tagger'ın KAPATICI'ları orada
# AÇICI olur). ``_TERMINATOR`` bu alt-kümeden YENİDEN DERLENİR → kanun bölgesi durum
# makinesinin (``tag_law_regions``) davranışı DEĞİŞMEZ: yalnız ``.search()`` boolean'ı
# kullanılır, dolayısıyla alternatif sırası önemsizdir.
# Harf-arası ``\s*`` → TBMM kapak/TOC başlıklarında sık görülen aralıklı dizgiyi
# ("İ Ç İ N D E K İ L E R") da yakalar; 11-harflik dizi çok özgül, false-match yok.
_R_ICINDEKILER = re.compile(
    r"İ\s*Ç\s*İ\s*N\s*D\s*E\s*K\s*[İIiı]\s*L\s*E\s*R", re.IGNORECASE
)
_R_GECEN_TUTANAK = re.compile(r"GEÇEN\s+TUTANAK", re.IGNORECASE)
_R_GELEN_KAGIT = re.compile(r"GELEN\s+K[AÂ]Ğ[Iı]T", re.IGNORECASE)
_R_YAZILI_SORU = re.compile(r"YAZILI\s+SORU", re.IGNORECASE)
_R_SOZLU_SORU = re.compile(r"SÖZLÜ\s+SORU", re.IGNORECASE)
_R_GUNDEM_DISI = re.compile(r"GÜNDEM\s+DIŞI", re.IGNORECASE)
_R_SECIM = re.compile(r"SEÇ[İIi]M", re.IGNORECASE)
_R_GENEL_GORUSME = re.compile(r"GENEL\s+GÖRÜŞME", re.IGNORECASE)
_R_MECLIS_ARASTIRMASI = re.compile(r"MECL[İIi]S\s+ARAŞTIRMASI", re.IGNORECASE)
_R_TEZKERE = re.compile(r"BAŞKANLIĞIN\s+GENEL\s+KURULA|TEZKERE", re.IGNORECASE)
_R_ONERILER = re.compile(r"ÖNER[İIi]LER", re.IGNORECASE)

# (section_type, desen) — kanun-tagger terminatörleri; ``tag_sections``'ta AÇICI.
_TERMINATOR_RUBRICS: list[tuple[str, "re.Pattern[str]"]] = [
    ("icindekiler", _R_ICINDEKILER),
    ("gecen_tutanak", _R_GECEN_TUTANAK),
    ("gelen_kagit", _R_GELEN_KAGIT),
    ("yazili_soru", _R_YAZILI_SORU),
    ("sozlu_soru", _R_SOZLU_SORU),
    ("gundem_disi", _R_GUNDEM_DISI),
    ("secim", _R_SECIM),
    ("genel_gorusme", _R_GENEL_GORUSME),
    ("meclis_arastirmasi", _R_MECLIS_ARASTIRMASI),
    ("tezkere", _R_TEZKERE),
    ("oneriler", _R_ONERILER),
]
# Kanun-tagger'ın gördüğü tek terminatör regex'i (alt-kümeden yeniden derlenir).
_TERMINATOR = re.compile(
    "|".join(p.pattern for _, p in _TERMINATOR_RUBRICS), re.IGNORECASE
)

# Kanun adı: "… Hakkında Kanun Teklifi/Tasarısı" → adı (best-effort, opsiyonel).
_KANUN_ADI = re.compile(
    r"([^.\n|]{5,110}?)\s+Hakk[ıi]nda\s+Kanun\s+(?:Teklif|Tasar)", re.IGNORECASE
)

_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$", re.MULTILINE)


def _distinct_sira(text: str) -> list[int]:
    """Metindeki tüm sıra sayısı adaylarını (parantez + kapak + gövde) topla."""
    vals: set[int] = set()
    for rx in (_SS_PAREN, _SS_COVER, _SS_BODY):
        for m in rx.finditer(text):
            try:
                vals.add(int(m.group(1)))
            except (ValueError, IndexError):
                continue
    return sorted(vals)


def _distinct_esas(text: str) -> list[str]:
    """Metindeki tüm esas no adaylarını (parantezli + 'esas' anahtarlı) topla."""
    vals: set[str] = set()
    for rx in (_ESAS_PAREN, _ESAS_KW):
        for m in rx.finditer(text):
            vals.add(m.group(1))
    return sorted(vals)


def _headings(text: str) -> str:
    """`#`-başlıklı satırların birleşimi — açıcı/terminatör başlık eşleşmesi burada
    aranır (gövde false-positive'lerinden kaçınmak için)."""
    return "\n".join(_HEADING_LINE.findall(text))


def _kanun_adi(text: str) -> Optional[str]:
    m = _KANUN_ADI.search(text)
    if not m:
        return None
    adi = re.sub(r"\s+", " ", m.group(1)).strip()
    # "…Mehmet Muş'un Türkiye Kalkınma Bankası…" — teklif/tasarı sahibi önekini at:
    # son genitive sınırından ("'un", "'nin", "'ın"…) sonrasını (asıl kanun adı) al.
    adi = re.split(r"'\w{1,4}\s+", adi)[-1]
    return adi.strip(" \t.-–—|") or None


def _build_maps(texts: list[str]) -> tuple[dict[int, str], dict[str, int], dict[int, str]]:
    """Pass 1: tek-N + tek-esas taşıyan GÜÇLÜ chunk'lardan N→esas, esas→N, N→adı
    haritalarını kur. Çakışmada çoğunluk kazanır (birkaç OCR hatasına dayanıklı)."""
    n_esas: dict[int, Counter] = {}
    n_adi: dict[int, str] = {}
    for raw in texts:
        t = raw or ""
        siras = _distinct_sira(t)
        esaslar = _distinct_esas(t)
        if len(siras) == 1 and len(esaslar) == 1:
            n = siras[0]
            n_esas.setdefault(n, Counter())[esaslar[0]] += 1
            if n not in n_adi:
                adi = _kanun_adi(t)
                if adi:
                    n_adi[n] = adi
    n_to_esas: dict[int, str] = {n: c.most_common(1)[0][0] for n, c in n_esas.items()}
    esas_to_n: dict[str, int] = {}
    for n, esas in n_to_esas.items():
        # Aynı esas'ı iki farklı N'e bağlama (ambiguity) — ilk gelen kazanır.
        esas_to_n.setdefault(esas, n)
    return n_to_esas, esas_to_n, n_adi


def _opener_n(
    text: str, headings: str, esas_to_n: dict[str, int], started_ever: bool
) -> Optional[int]:
    """Bu chunk bir bölge AÇICISI mı? Öyleyse tek çözülebilir sıra sayısını döndür.

    İki tür açıcı:
      * **Güçlü** (sıfırdan bölge açabilir): görüşme-başla / rapor kapağı (SIRA
        SAYISI: N başlığı) / roll-call (açık oy sonucu) başlığı — "bu kanunu şimdi
        işliyoruz" diyen kesin sinyaller.
      * **Devam** (yalnız zaten bir bölge açılmışsa geçerli): çıplak kanun-işi bölüm
        başlığı ("VIII.- KANUN TEKLİFLERİ …") — görüşme boyunca ~20-30 chunk'ta bir
        tekrar eden "(Devam)" yeniden-çıpaları. Ön-materyalde (İÇİNDEKİLER'in "VIII.-
        KANUN TEKLİFLERİ" TOC satırı gibi) yanlışlıkla bölge açmasın diye
        ``started_ever`` ile kapılıdır (plan: ön-materyal yalnız haritaya katkı verir).

    N önce açık sıra sayısı adaylarından, yoksa (görüşme başlangıcı çoğu kez yalnız
    esas taşır) tek-esas → N ters haritasından çözülür. Çok-N belirsizliğinde açıcı
    DEĞİL (None) → mevcut bölge korunur."""
    strong = bool(
        _GORUSME.search(text) or _VOTE_HEADER.search(text) or _SS_COVER.search(headings)
    )
    devam = bool(_LAW_HEADING.search(headings))
    if not (strong or (devam and started_ever)):
        return None
    siras = _distinct_sira(text)
    if len(siras) == 1:
        return siras[0]
    if not siras:
        esaslar = _distinct_esas(text)
        if len(esaslar) == 1 and esaslar[0] in esas_to_n:
            return esas_to_n[esaslar[0]]
    return None  # çok-N (belirsiz) ya da çözülemeyen → açıcı sayma


def tag_law_regions(
    ordered_chunk_texts: list[str], *, pages: Optional[list[Optional[int]]] = None
) -> list[dict[str, Any]]:
    """Sıralı chunk metinlerini kanun bölgelerine göre etiketle.

    Args:
        ordered_chunk_texts: bir belgenin chunk metinleri, chunk_index sırasında.
        pages: opsiyonel sayfa numaraları (API uyumu için; mevcut mantık sıra-tabanlı).

    Returns:
        Her chunk için ``{"sira_sayisi": int|None, "esas_no": str|None,
        "kanun_adi": str|None}`` — belirsiz/kapsam-dışı chunk'lar için hepsi None.
    """
    n_to_esas, esas_to_n, n_adi = _build_maps(ordered_chunk_texts)

    out: list[dict[str, Any]] = []
    current: Optional[int] = None
    started_ever = False
    for t in ordered_chunk_texts:
        text = t or ""
        headings = _headings(text)

        opener = _opener_n(text, headings, esas_to_n, started_ever)
        if opener is not None:
            # Ön-materyal (ilk güçlü açıcıdan önce) yalnız haritaya katkı verir; bir
            # açıcı görülünce bölge etiketlemesi açılır.
            current = opener
            started_ever = True
        elif started_ever and _TERMINATOR.search(headings):
            # Kanun-dışı üst rubrik → bölgeyi kapat (bu chunk etiketlenmez).
            current = None

        if current is None:
            out.append({"sira_sayisi": None, "esas_no": None, "kanun_adi": None})
        else:
            out.append({
                "sira_sayisi": current,
                "esas_no": n_to_esas.get(current),
                "kanun_adi": n_adi.get(current),
            })
    return out


# ── Bölüm (section) etiketleyici — kardeş durum makinesi ────────────────────
# Ant içme bölümü — kanun-tagger terminatörü DEĞİL (yeni), yalnız section_type için.
_R_YEMIN = re.compile(r"YEM[İIi]N", re.IGNORECASE)

# `#`-başlık rubrikleri, first-match-wins. Çıplak kanun-işi başlığı ("KANUN
# TEKLİFLERİ" / "KOMİSYONLARDAN GELEN") = ``kanun_gorusmeleri`` DEVAM açıcısı (yalnız
# güçlü açıcı görüldükten sonra geçerli — ön-materyal kapısı). Terminatör rubrikleri
# kanonik section_type'larıyla + yemin. Güçlü açıcılar (görüşme/oy DUYURULARI +
# rapor kapağı) burada DEĞİL — onlar gövdede/kapakta ayrı aranır (tag_sections).
_HEADING_RUBRICS: list[tuple[str, "re.Pattern[str]"]] = [
    ("kanun_gorusmeleri", _LAW_HEADING),
    *_TERMINATOR_RUBRICS,
    ("yemin", _R_YEMIN),
]


def _match_heading_rubric(headings: str) -> Optional[str]:
    """`#`-başlık metninde ilk eşleşen rubrik section_type'ını döndür (yoksa None)."""
    for section_type, rx in _HEADING_RUBRICS:
        if rx.search(headings):
            return section_type
    return None


def tag_sections(
    ordered_chunk_texts: list[str], *, pages: Optional[list[Optional[int]]] = None
) -> list[dict[str, Any]]:
    """Sıralı chunk metinlerini kanonik BÖLÜM (section) tiplerine göre etiketle.

    ``tag_law_regions``'ın kardeşi: aynı `#`-başlık disiplinini + ileri-yaymayı
    kullanır ama HER chunk'a bir ``section_type`` koyar (kanun-tagger'ın yalnız kanun
    bölgelerini etiketlemesinin aksine). Kanun-tagger'ın terminatörleri burada AÇICI
    olur (yazili_soru, icindekiler, gelen_kagit… birinci sınıf bölüm).

    **Sinyaller (öncelik sırasıyla):**
      1. ``_VOTE_HEADER`` (gövde) → ``oylama`` — güçlü. Roll-call tablo başlığı /
         "açık oylama sonucu" duyurusu; kimliksiz tablo satırları ileri-yayma ile
         ``oylama`` alır (ASIL senaryo — reflect ``section_type=oylama`` filtresi).
      2. ``_GORUSME`` (gövde) → ``kanun_gorusmeleri`` — güçlü ("görüşmelerine başl").
      3. ``_SS_COVER`` (başlık) → ``kanun_raporu`` — güçlü (rapor kapağı).
      4. ``#``-başlık rubrikleri (``_HEADING_RUBRICS``, first-match) → terminatör
         rubrikleri + çıplak kanun-işi başlığı (DEVAM) + yemin.

    **Güven sınırı = kanun-tagger ile aynı:** terminatör/kanun-işi/yemin rubrikleri
    YALNIZ ``#``-başlık satırlarında aranır (gövde false-positive'i yok); görüşme/oy
    DUYURULARI gövdede aranır (başlık değil — akış içinde söylenirler). Ham başlık
    metni SAKLANMAZ → tanınmayan/OCR-bozuk başlık eylemsizdir (ileri-yayar, yanlış
    kovaya etiketlemez); Docling'e yalnız "`#` bastı" kadarıyla güvenilir.

    **İÇİNDEKİLER (TOC) kapısı:** TBMM tutanağının başında İÇİNDEKİLER tüm gündemi
    sayfa-no'yla LİSTELER (aralıklı "İ Ç İ N D E K İ L E R" başlığı + "VIII.- KANUN
    TEKLİFLERİ", "X.- YAZILI SORULAR" gibi satırlar). Bu TOC satırları içerik değildir
    → ``icindekiler``'deyken bölümü FLIP ETMEZ. TOC yalnız ilk GEÇEN TUTANAK
    (birleşimin kanonik gerçek ilk gündemi) ya da bir güçlü açıcı ile kapanır; sonra
    rubrik başlıkları (gecen_tutanak/gelen_kagit/tezkere/…) SERBESTÇE geçiş yapar.
    İÇİNDEKİLER saptanamayan belgede (saf rapor PDF) kapı hiç devreye girmez.

    **Tasarım sapması (kanun-tagger'dan):** kanun-tagger şüphede ``None`` bırakır;
    bölüm-tagger İLERİ-YAYAR → kaçırılan bir sınırdan sonra bir sonraki gerçek rubrik
    yeniden çıpalayana dek AKTİF yanlış etiketleyebilir. Backfill ``--dry-run``
    histogramı bunu ölçer (``diger``/None oranı + section_type dağılımı).

    Returns:
        Her chunk için ``{"section_type": str|None, "section_ord": int|None,
        "section_path": str|None}`` — ilk rubrikten önceki chunk'lar için hepsi None.
        ``section_ord`` bölüm her DEĞİŞTİĞİNDE artan monoton int (aralık gezinme);
        ``section_path`` yalnız görüntü/debug breadcrumb'ı (oylama için kapsayan
        üst bölüm: "kanun_gorusmeleri > oylama").
    """
    out: list[dict[str, Any]] = []
    current: Optional[str] = None
    strong_seen = False
    ordinal = 0
    last_major: Optional[str] = None  # oylama'yı kapsayan üst bölüm (breadcrumb)

    for t in ordered_chunk_texts:
        text = t or ""
        headings = _headings(text)

        # Güçlü açıcılar (duyuru sinyalleri) — gövde + kapak başlığı.
        if _VOTE_HEADER.search(text):
            cand, strong = "oylama", True
        elif _GORUSME.search(text):
            cand, strong = "kanun_gorusmeleri", True
        elif _SS_COVER.search(headings):
            cand, strong = "kanun_raporu", True
        else:
            cand, strong = _match_heading_rubric(headings), False

        if cand is not None:
            if strong:
                fire = True
            elif cand == "kanun_gorusmeleri":
                # Çıplak kanun-işi başlığı = DEVAM açıcısı; yalnız güçlü açıcı sonrası
                # geçerli → İÇİNDEKİLER'deki "VIII.- KANUN TEKLİFLERİ" TOC satırının
                # kanun_gorusmeleri açmasını engeller (en zararlı TOC hatası).
                fire = strong_seen
            elif current is None:
                # İlk tanınan rubrik ön-materyali açar (tipik İÇİNDEKİLER kapağı).
                fire = True
            elif current == "icindekiler":
                # İÇİNDEKİLER (TOC) bölgesindeyiz: diğer rubrik-başlıkları içerik değil
                # TOC LİSTELEMELERİDİR (tüm gündemi sayfa-no'yla sayar) → bölümü FLIP
                # ETMEZ. TOC yalnız ilk GEÇEN TUTANAK (birleşimin kanonik gerçek ilk
                # gündemi) ya da güçlü açıcıyla kapanır → parçalanma önlenir. İÇİNDEKİLER
                # saptanamayan belgede (saf rapor PDF) bu dal hiç girilmez, güvenli.
                fire = cand == "gecen_tutanak"
            else:
                # Gövdedeyiz — rubrik başlıkları serbestçe geçiş yapar.
                fire = True
            if fire:
                if strong:
                    strong_seen = True
                if cand != current:
                    current = cand
                    ordinal += 1

        if current is None:
            out.append({"section_type": None, "section_ord": None, "section_path": None})
            continue

        if current == "oylama":
            path = f"{last_major} > oylama" if last_major else "oylama"
        else:
            path = current
            last_major = current
        out.append({
            "section_type": current,
            "section_ord": ordinal,
            "section_path": path,
        })
    return out
