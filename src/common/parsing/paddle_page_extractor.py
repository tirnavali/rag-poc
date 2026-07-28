"""Tam-sayfa PaddleOCR-VL yönlendirmesi — taranmış sayfaları Docling+EasyOCR
yerine :8080 PaddleOCR-VL pipeline'ına yollar.

Bu modül `paddle_table_extractor.py`'nin (tek bozuk tablo → :4000 ham VLM)
**tam-sayfa kardeşidir**: bir sayfanın tümünü :8080 proxy'sine gönderir; proxy
layout analizi + tablo yapısı + oto dik-çevirme yapıp temiz markdown ve semantik
bloklar (aside_text / figure_title / table / number ...) döndürür.

Neden ayrı bir yol: Docling'in layout modeli taranmış/eğik TBMM sayfalarında
tabloları ve listeleri karıştırır, TableFormer yapı çıkaramaz (num_rows=0). Kök
neden sayfa-düzeyinde olduğu için düzeltme de sayfa-düzeyinde olmalı — atom-düzeyi
tablo yükseltmesi (table_is_low_quality → VLM) bunu sonradan yamayamaz.

Yalnızca native metin katmanı zayıf (taranmış) sayfalarda devreye girer; digital-born
sayfalar dokunulmadan Docling'in native (PyPdfium) yolunda kalır — bedava ve kusursuz.

:8080 sözleşmesi (gerçek çağrıyla doğrulandı):
  POST /ocr  multipart/form-data  file=<png bytes>
  -> {"results": [{"markdown": "...", "json": "<PaddleX repr>", "annotated_image_b64": "..."}]}
  json içinde parsing_res_list blokları:  label:\\t...\\nbbox:\\t[...]\\ncontent:\\t...
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import requests

from src.config import settings

_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sayfa sınıflandırma (digital-born vs taranmış)
# ----------------------------------------------------------------------

def native_char_counts(pdf_path: str) -> Dict[int, int]:
    """Her sayfanın native (çıkarılabilir) metin karakter sayısı — {1-tabanlı sayfa: sayı}.

    Taranmış sayfalarda bu ≈0 (yalnız filigran/damga); digital-born sayfalarda yüzlerce.
    """
    counts: Dict[int, int] = {}
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            counts[i + 1] = len(doc[i].get_text("text").strip())
    finally:
        doc.close()
    return counts


def scanned_page_set(pdf_path: str, threshold: int | None = None) -> set[int]:
    """Native karakteri eşiğin altında kalan (taranmış) 1-tabanlı sayfa numaraları."""
    if threshold is None:
        threshold = settings.SCANNED_PAGE_CHAR_THRESHOLD
    return {p for p, c in native_char_counts(pdf_path).items() if c < threshold}


def page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_page_png(pdf_path: str, page_no: int, zoom: float | None = None) -> bytes | None:
    """1-tabanlı sayfayı PNG byte'a render eder (PaddleOCR-VL kendi deskew'unu yapar)."""
    if zoom is None:
        zoom = settings.SCANNED_PAGE_RENDER_ZOOM
    doc = fitz.open(pdf_path)
    try:
        if page_no < 1 or page_no > doc.page_count:
            _logger.error("render: sayfa %s aralık dışı (%s): %s", page_no, doc.page_count, pdf_path)
            return None
        return doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
    except Exception as e:  # pragma: no cover - render nadiren patlar
        _logger.error("render hatası (sayfa %s): %s", page_no, e)
        return None
    finally:
        doc.close()


# ----------------------------------------------------------------------
# :8080 yanıt ayrıştırma
# ----------------------------------------------------------------------

# parsing_res_list bloğu:  #####\nlabel:\t<lbl>\nbbox:\t[..]\ncontent:\t<...>\n#####
_BLOCK_RE = re.compile(
    r"label:\t(?P<label>\w+)\nbbox:\t(?P<bbox>\[[^\]]*\])\ncontent:\t(?P<content>.*?)\n#{6,}",
    re.DOTALL,
)
_ANGLE_RE = re.compile(r"'angle':\s*(\d+)")


def parse_paddle_blocks(json_str: str) -> List[Dict[str, Any]]:
    """PaddleX repr string'inden semantik blokları çıkarır: [{label, bbox, content}]."""
    blocks: List[Dict[str, Any]] = []
    for m in _BLOCK_RE.finditer(json_str or ""):
        blocks.append(
            {
                "label": m.group("label"),
                "bbox": m.group("bbox"),
                "content": m.group("content").strip(),
            }
        )
    return blocks


def parse_angle(json_str: str) -> int | None:
    """doc_preprocessor_res içindeki oto-deskew açısını döndürür (bilgi amaçlı)."""
    m = _ANGLE_RE.search(json_str or "")
    return int(m.group(1)) if m else None


def call_paddle_page(
    image_bytes: bytes, base_url: str, timeout: int
) -> Dict[str, Any] | None:
    """Sayfa görüntüsünü :8080/ocr'a (multipart file=) yollar.

    Döner: {markdown, blocks, angle, annotated_image_b64}. `annotated_image_b64`
    Paddle'ın layout-kutulu görselleştirmesidir (viewer önizlemesinde kullanılır;
    process_pages_with_paddle yok sayar — ek anahtar geriye uyumlu).
    """
    url = f"{base_url.rstrip('/')}/ocr"
    try:
        resp = requests.post(
            url, files={"file": ("page.png", image_bytes, "image/png")}, timeout=timeout
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            _logger.warning("Paddle :8080 boş results döndü.")
            return None
        res0 = results[0]
        raw = res0.get("json", "") or ""
        return {
            "markdown": res0.get("markdown", "") or "",
            "blocks": parse_paddle_blocks(raw),
            "angle": parse_angle(raw),
            "annotated_image_b64": res0.get("annotated_image_b64"),
        }
    except Exception as e:
        _logger.error("Paddle :8080 çağrı hatası: %s", e)
        print(f"    [PAGE-ROUTER] [API] HATA: {e}")
        return None


def preview_page(
    pdf_path: str,
    page_no: int,
    base_url: str | None = None,
    timeout: int | None = None,
    zoom: float | None = None,
) -> Dict[str, Any]:
    """Tek bir sayfayı PaddleOCR-VL ile okuyup ÖNİZLEME döndürür — hiçbir şey yazmaz.

    Debug viewer'ın "bu sayfayı Paddle ile oku" butonu için ince sarmalayıcı:
    render_page_png → call_paddle_page → blocks_to_atoms zincirini birleştirir ve
    süreyi ölçer. Cache/DB/dosya YAZMAZ (viewer salt-okuma kalır).

    Returns:
        Başarıda: {markdown, angle, blocks:[{label,content}], atoms:[{label,text,page}],
                   annotated_image_b64, elapsed_s}
        Hata: {error: "..."} (render veya API başarısız).
    """
    if base_url is None:
        base_url = settings.PADDLE_PAGE_OCR_URL
    if timeout is None:
        timeout = settings.PADDLE_PAGE_TIMEOUT
    t0 = time.time()

    img = render_page_png(pdf_path, page_no, zoom)
    if img is None:
        return {"error": f"Sayfa {page_no} render edilemedi (PDF yok ya da sayfa aralık dışı)."}

    oc = call_paddle_page(img, base_url, timeout)
    if oc is None:
        return {"error": f"PaddleOCR-VL ({base_url}) sayfa {page_no} için yanıt vermedi."}

    blocks = oc.get("blocks") or []
    atoms = blocks_to_atoms(blocks, page_no, oc.get("markdown", ""))
    furn = blocks_to_furniture(blocks)
    return {
        "markdown": oc.get("markdown", ""),  # ham Paddle çıktısı (tablolar HTML)
        # gövde: HTML→MD çevrilmiş + furniture çıkarılmış (Docling+EasyOCR hizası)
        "body_markdown": "\n\n".join(a["text"] for a in atoms),
        "angle": oc.get("angle"),
        # üst/alt bilgi artık page-metadata olarak yakalanır (Docling FURNITURE hizası)
        "page_header": furn["page_header"],
        "page_footer": furn["page_footer"],
        "blocks": [
            {"label": b.get("label"), "content": b.get("content", "")}
            for b in blocks
        ],
        "atoms": [
            {"label": a.get("label"), "text": a.get("text", ""), "page": a.get("page")}
            for a in atoms
        ],
        "annotated_image_b64": oc.get("annotated_image_b64"),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ----------------------------------------------------------------------
# Blok -> atom
# ----------------------------------------------------------------------

# Docling+EasyOCR hizası: üst/alt bilgi + sayfa numarası GÖVDEYE girmez ama
# ATILMAZ da — Docling'in FURNITURE (PAGE_HEADER/PAGE_FOOTER) yakalaması gibi
# page-metadata olarak toplanır (bkz. blocks_to_furniture). number → footer'a.
_FURNITURE_LABELS = {"header", "footer", "number"}

# İçerik-deseni: PaddleOCR-VL, TBMM running-head'ini bazen `doc_title`/`text`
# etiketliyor (Docling ise PAGE_HEADER). Etiketten bağımsız, KISA bloğun içeriği
# bu desene uyuyorsa furniture'a alınır → iki motor aynı üst/alt bilgi çıkarsın.
# Belirsiz "Türkiye Büyük Millet Meclisi" ve "(Sıra Sayısı: N)" BİLİNÇLİ dışarıda:
# ilki yanlış-pozitif riski, ikincisi sira_sayisi omurgasının gövdeden çıkardığı sinyal.
_RUNNING_HEAD_RE = re.compile(
    r"D[ÖO]NEM\s*[:.]"
    r"|C[İIıi]LT\s*[:.]"
    r"|YASAMA\s+YILI\s*[:.]"
    r"|\bB\s*:\s*\d+.{0,40}?\bO\s*:\s*\d+",   # "B: 6 ... O: 5" (birleşim + oturum)
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^[\s‒–—\-]*\d{1,4}[\s‒–—\-]*$")  # çıplak sayfa numarası

# Paddle blok etiketi -> Docling-uyumlu atom etiketi. table/picture etiketleri
# token-packer'da bölünmez (bkz. _NO_SUFFIX_LABELS); table ayrıca table_is_low_quality
# guard'ıyla korunur (extracted_by=paddle_page).
_LABEL_MAP = {
    "table": "table",
    "figure_title": "caption",
    "figure": "picture",
    "picture": "picture",
    "image": "picture",
    "chart": "picture",
    "formula": "formula",
    "aside_text": "text",
    "text": "text",
    "paragraph_title": "section_header",
    "doc_title": "title",
    "abstract": "text",
    "reference": "text",
}


def _table_to_md(tbl) -> str:
    """BeautifulSoup <table> düğümünü Markdown pipe-tablosuna çevirir (boşsa "")."""
    rows: List[List[str]] = []
    for tr in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    fmt = lambda r: "| " + " | ".join(c.replace("|", "\\|") for c in r) + " |"
    out = [fmt(rows[0]), "| " + " | ".join(["---"] * ncol) + " |"]
    out += [fmt(r) for r in rows[1:]]
    return "\n".join(out)


def _html_to_markdown(text: str) -> str:
    """PaddleOCR-VL HTML çıktısını (tablolar <table>, sarmalayıcı <div style>) Markdown'a
    çevirir → Docling'in MD pipe-tablo konvansiyonuyla hizalı, embedding'e temiz metin.

    HTML içermiyorsa metin AYNEN döner (düz MD/metin bozulmaz). bs4 yoksa güvenli
    fallback: metin aynen döner.
    """
    if not text or "<" not in text:
        return (text or "").strip()
    try:
        from bs4 import BeautifulSoup
    except Exception:  # noqa: BLE001 — bs4 yoksa HTML'i olduğu gibi bırak
        return text.strip()
    soup = BeautifulSoup(text, "html.parser")
    parts: List[str] = []
    for el in soup.contents:
        name = getattr(el, "name", None)
        if name == "table":
            md = _table_to_md(el)
            if md:
                parts.append(md)
        elif name is None:  # NavigableString
            s = str(el).strip()
            if s:
                parts.append(s)
        else:  # div / p / span ...
            for br in el.find_all("br"):
                br.replace_with("\n")
            for tbl in el.find_all("table"):
                tbl.replace_with("\n" + _table_to_md(tbl) + "\n")
            s = el.get_text().strip()
            if s:
                parts.append(s)
    return "\n\n".join(parts).strip()


def _furniture_role(block: Dict[str, Any]) -> str | None:
    """Bir bloğun furniture rolü: "header" | "footer" | None (gövde).

    Önce Paddle'ın açık etiketi (header/footer/number), sonra içerik-deseni
    (Paddle running-head'i doc_title/text sanmışsa). İçerik-deseni yalnız KISA
    (<120 char) bloklara uygulanır — uzun paragrafta 'DÖNEM:' geçse bile gövdede kalır.
    """
    label = block.get("label", "")
    if label == "header":
        return "header"
    if label in ("footer", "number"):
        return "footer"
    content = _html_to_markdown((block.get("content") or "").strip())
    if not content or len(content) > 120:
        return None
    if _PAGE_NUMBER_RE.match(content):
        return "footer"
    if _RUNNING_HEAD_RE.search(content):
        return "header"
    return None


def blocks_to_furniture(blocks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Üst/alt bilgi bloklarını page-metadata'ya toplar (Docling FURNITURE hizası).

    Rol _furniture_role ile belirlenir (açık etiket + içerik-deseni). number ve
    çıplak sayfa-numarası → footer. Döner: {"page_header": "...", "page_footer": "..."}."""
    header: List[str] = []
    footer: List[str] = []
    for b in blocks:
        role = _furniture_role(b)
        if role is None:
            continue
        content = _html_to_markdown((b.get("content") or "").strip())
        if not content:
            continue
        (header if role == "header" else footer).append(content)
    return {"page_header": " | ".join(header), "page_footer": " | ".join(footer)}


def blocks_to_atoms(
    blocks: List[Dict[str, Any]], page_no: int, markdown_fallback: str = ""
) -> List[Dict[str, Any]]:
    """Semantik blokları GÖVDE atom'una çevirir (extracted_by=paddle_page).

    Üst/alt bilgi + sayfa numarası (_FURNITURE_LABELS) gövdeye alınmaz (bunlar
    blocks_to_furniture ile page-metadata olur). Tablolar HTML→Markdown çevrilir
    (Docling MD-tablo hizası). Blok yoksa/hepsi elendiyse ve markdown_fallback
    doluysa tek bir sayfa-atomuna düşülür (veri kaybı olmasın).
    """
    atoms: List[Dict[str, Any]] = []
    for b in blocks:
        label = b.get("label", "")
        if _furniture_role(b) is not None:   # üst/alt bilgi + running-head → gövdeye girmez
            continue
        content = _html_to_markdown((b.get("content") or "").strip())
        if not content:
            continue
        atoms.append(
            {
                "text": content,
                "label": _LABEL_MAP.get(label, "text"),
                "page": page_no,
                "pages": [page_no],
                "extracted_by": "paddle_page",
            }
        )
    if not atoms and (markdown_fallback or "").strip():
        atoms.append(
            {
                "text": _html_to_markdown(markdown_fallback.strip()),
                "label": "text",
                "page": page_no,
                "pages": [page_no],
                "extracted_by": "paddle_page",
            }
        )
    return atoms


# ----------------------------------------------------------------------
# Sayfa yönlendirici (ana giriş)
# ----------------------------------------------------------------------

def process_pages_with_paddle(
    pdf_path: str,
    atoms: List[Dict[str, Any]],
    base_url: str | None = None,
    timeout: int | None = None,
    threshold: int | None = None,
    zoom: float | None = None,
    start_time: float | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Taranmış sayfaların Docling atomlarını PaddleOCR-VL tam-sayfa çıktısıyla değiştirir.

    Digital-born sayfalar dokunulmaz (Docling native atomları korunur). Bir sayfanın
    Paddle çağrısı başarısızsa o sayfanın Docling atomları korunur (veri kaybı yok).

    Returns:
        (yeni_atoms, stats). stats: total_pages, scanned_pages, paddle_calls, failed, angles.
    """
    if base_url is None:
        base_url = settings.PADDLE_PAGE_OCR_URL
    if timeout is None:
        timeout = settings.PADDLE_PAGE_TIMEOUT
    if start_time is None:
        start_time = time.time()

    scanned = scanned_page_set(pdf_path, threshold)
    total = page_count(pdf_path)
    stats: Dict[str, Any] = {
        "total_pages": total,
        "scanned_pages": sorted(scanned),
        "paddle_calls": 0,
        "failed": [],   # render/API hatası (gerçek başarısızlık — veri kaybı riski)
        "empty": [],    # API başarılı ama içerik yok (boş/ayraç sayfa — kayıp değil)
        "angles": {},
        # yönlendirilen sayfaların üst/alt bilgisi (Docling FURNITURE hizası) —
        # MarkdownConverter bunu page_furniture'a merge eder.
        "page_furniture": {},
    }
    if not scanned:
        return atoms, stats

    # Digital atomları primary page'e göre kovala (sıra korunur).
    by_page: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        by_page[a.get("page")].append(a)

    def _ts() -> str:
        return f"{time.time() - start_time:.1f}s"

    print(
        f"  [PAGE-ROUTER] {total} sayfa; taranmış: {sorted(scanned)} "
        f"(<{threshold or settings.SCANNED_PAGE_CHAR_THRESHOLD} native char)"
    )

    result: List[Dict[str, Any]] = []
    for p in range(1, total + 1):
        if p not in scanned:
            result.extend(by_page.get(p, []))
            continue

        print(f"  [PAGE-ROUTER] [{_ts()}] Sayfa {p} → PaddleOCR-VL (:8080)")
        img = render_page_png(pdf_path, p, zoom)
        oc = call_paddle_page(img, base_url, timeout) if img else None
        if oc:
            stats["paddle_calls"] += 1
            if oc.get("angle") is not None:
                stats["angles"][p] = oc["angle"]
            new_atoms = blocks_to_atoms(oc["blocks"], p, oc["markdown"])
            if new_atoms:
                furn = blocks_to_furniture(oc["blocks"])
                if furn["page_header"] or furn["page_footer"]:
                    stats["page_furniture"][p] = furn
                print(
                    f"  [PAGE-ROUTER] [{_ts()}] Sayfa {p}: {len(new_atoms)} atom "
                    f"(açı={oc.get('angle')})"
                )
                result.extend(new_atoms)
                continue
            # API başarılı ama içerik yok → boş/ayraç sayfa (gerçek hata değil).
            print(f"  [PAGE-ROUTER] [{_ts()}] Sayfa {p}: boş (içerik yok) — Docling atomları korunuyor.")
            stats["empty"].append(p)
            result.extend(by_page.get(p, []))
            continue
        # render/API hatası → gerçek başarısızlık; Docling atomlarını koru (veri kaybı yok).
        print(f"  [PAGE-ROUTER] [{_ts()}] [WARN] Sayfa {p} Paddle başarısız — Docling atomları korunuyor.")
        stats["failed"].append(p)
        result.extend(by_page.get(p, []))

    # Sayfası olmayan (nadir) atomlar sona.
    result.extend(by_page.get(None, []))
    return result, stats
