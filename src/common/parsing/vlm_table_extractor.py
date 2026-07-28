"""Bu modül PDF'lerdeki tabloları yerel bir görü-dil modeli (VLM) ile çıkarır.

Hedef profil: taranmış ve **90° döndürülmüş** tablo görüntüleri (TBMM bütçe
kanunu icmalleri gibi) — Docling'in TableFormer'ı yapı çıkaramaz, OCR çöp üretir.

Akış:
  1. Tablo bbox'ı orijinal PDF sayfasından yüksek çözünürlükte kırpılır (PyMuPDF).
  2. Kırpıntı Tesseract OSD ile tespit edilen açıya göre dik (upright) çevrilir.
  3. Çok uzun (dik) görüntüler satır-bantlarına bölünür.
  4. Her görüntü/bant yerel Ollama VLM'ine (qwen2.5vl) gönderilip markdown tabloya çevrilir.

Yalnızca "bozuk" tablolarda devreye girer (bkz. table_is_low_quality).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List

import fitz  # PyMuPDF
import requests
from PIL import Image

from src.config import settings

_logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Düşük kalite (bozuk tablo) tespiti — VLM tetikleyicisi
# ----------------------------------------------------------------------

_ALNUM_RE = re.compile(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü]")


def _looks_like_garbage(text: str, num_rows: int | None = None, num_cols: int | None = None) -> bool:
    """Markdown tablo metni OCR çöpü mü?

    Bu taranmış/döndürülmüş tablolarda OCR iki desende bozulur: (a) tek-karakter ya da
    boşlukla ayrılmış tek-haneli çöp (``| ğ | 5 1 1 1 |``), (b) mojibake kelimeler +
    çoğu hücre boş (``KMU DUZENI VE HEZMETLERI`` + boşluklar). Aşağıdaki sinyaller
    örnek belgedeki 8 bozuk tabloyu da yakalar; meşru ``0`` hücreleri (dolu + tek token)
    yanlış tetiklemez.
    """
    cells: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        for cell in line.strip("|").split("|"):
            cells.append(cell.strip())

    nonempty = [c for c in cells if c]
    if len(nonempty) < 4:
        return True

    # (1) Izgaranın çoğu boş → OCR hücreleri kaçırmış (dolu tablo ~tam dolu olmalı).
    if num_rows and num_cols and num_rows * num_cols > 0:
        if len(nonempty) / (num_rows * num_cols) < 0.5:
            return True

    # (2) Boşlukla ayrılmış tek-haneli parçalanma ("1 9 9", "8 1 5").
    frag = 0
    for c in nonempty:
        toks = c.split()
        if len(toks) >= 2 and sum(1 for t in toks if len(t) <= 1) / len(toks) >= 0.6:
            frag += 1
    if frag / len(nonempty) >= 0.5:
        return True

    # (3) Anlamlı (>=3 alnum) hücre oranı çok düşük → metin tanınmamış.
    wordlike = sum(1 for c in nonempty if len(_ALNUM_RE.sub("", c)) >= 3)
    if wordlike / len(nonempty) < 0.20:
        return True

    return False


def table_is_low_quality(atom: Dict[str, Any]) -> bool:
    """Bir tablo atomu VLM ile yeniden okunmalı mı?

    Tetikleyici: TableFormer yapı çıkaramamış (num_rows/num_cols == 0) **veya**
    OCR metni çöp görünümlü. Zaten VLM ile işlenmiş atomları (cache) atlar.
    """
    if atom.get("label") != "table":
        return False
    # Zaten bir backend'le yeniden okunmuş tabloyu (VLM / Paddle / tam-sayfa Paddle /
    # Tesseract) yeniden tetikleme — Paddle tabloları HTML <table> olduğu için
    # _looks_like_garbage yanlış-pozitif verirdi; guard bunu keser.
    if atom.get("extracted_by") in ("vlm", "paddle", "paddle_page", "tesseract"):
        return False
    num_rows = atom.get("table_num_rows")
    num_cols = atom.get("table_num_cols")
    if num_rows == 0 or num_cols == 0:
        return True
    return _looks_like_garbage(atom.get("text", ""), num_rows, num_cols)


# ----------------------------------------------------------------------
# Yön tespiti (Tesseract OSD) + dik çevirme
# ----------------------------------------------------------------------

_OSD_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")
_OSD_CONF_RE = re.compile(r"Orientation confidence:\s*([\d.]+)")


def _parse_osd(text: str) -> tuple[int | None, float | None]:
    """Tesseract OSD çıktısından (Rotate açısı, yön güveni) ayıklar."""
    rot = _OSD_ROTATE_RE.search(text)
    conf = _OSD_CONF_RE.search(text)
    rotate = int(rot.group(1)) if rot else None
    confidence = float(conf.group(1)) if conf else None
    return rotate, confidence


def deskew_to_upright(
    image_bytes: bytes,
    autorotate: bool = True,
    min_confidence: float | None = None,
    fallback_rotate: int = 90,
) -> bytes:
    """Görüntüyü dik (upright) hale getirir; PNG byte döndürür.

    Birincil: Tesseract OSD (``tesseract - - --psm 0 -l osd``) ile açı tespiti.
    Fallback (OSD yok/güvensiz): kırpıntı portre (h>w) ise — bu PDF'lerdeki
    döndürülmüş-yatay tablo profili — 90° saat yönünde çevir; aksi halde dokunma.
    Hiçbir hata ölümcül değildir: sorun olursa girdi byte'ları aynen döner.
    """
    if not autorotate:
        return image_bytes
    if min_confidence is None:
        min_confidence = settings.VLM_TABLE_OSD_MIN_CONFIDENCE

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        _logger.warning(f"deskew: görüntü açılamadı, döndürme atlanıyor: {e}")
        return image_bytes

    rotate_deg: int | None = None
    if shutil.which("tesseract"):
        print(f"    [OSD] Tesseract OSD ile yön tespiti yapılıyor...")
        t0 = time.time()
        try:
            proc = subprocess.run(
                ["tesseract", "-", "-", "--psm", "0", "-l", "osd"],
                input=image_bytes,
                capture_output=True,
                timeout=30,
            )
            osd = proc.stdout.decode("utf-8", "ignore")
            rot, conf = _parse_osd(osd)
            if rot is not None and (conf is None or conf >= min_confidence):
                rotate_deg = rot % 360
                print(f"    [OSD] {time.time()-t0:.1f}s — açı={rotate_deg}°, güven={conf:.2f}")
            elif rot is not None:
                print(f"    [OSD] {time.time()-t0:.1f}s — güven düşük ({conf:.2f} < {min_confidence}), fallback'e geçiliyor")
            else:
                print(f"    [OSD] {time.time()-t0:.1f}s — açı tespit edilemedi, fallback'e geçiliyor")
        except Exception as e:
            _logger.warning(f"OSD yön tespiti başarısız: {e}")
            print(f"    [OSD] {time.time()-t0:.1f}s — OSD hatası ({e}), fallback'e geçiliyor")
    else:
        print(f"    [OSD] Tesseract yüklü değil, aspect-ratio fallback kullanılıyor")

    if rotate_deg is None:
        w, h = img.size
        rotate_deg = fallback_rotate % 360 if h > w else 0
        src = "aspect-ratio fallback"
        print(f"    [OSD] Fallback: görüntü {w}×{h}px → {rotate_deg}° döndürülecek ({src})")

    if rotate_deg % 360 == 0:
        print(f"    [OSD] Döndürme gerekmedi (0°) — görüntü zaten dik")
        return image_bytes

    try:
        # OSD 'Rotate: N' = görüntüyü dik yapmak için saat yönünde N derece döndür.
        # PIL rotate pozitif açı = CCW; saat yönü için -N. expand=True kırpmayı önler.
        rotated = img.rotate(-rotate_deg, expand=True)
        buf = io.BytesIO()
        rotated.save(buf, "PNG")
        print(f"    [OSD] Görüntü {rotate_deg}° (saat yönü) döndürüldü → {rotated.width}×{rotated.height}px")
        return buf.getvalue()
    except Exception as e:
        _logger.warning(f"Görüntü döndürme başarısız: {e}")
        return image_bytes


# ----------------------------------------------------------------------
# Kırpma
# ----------------------------------------------------------------------

def _adaptive_zoom(long_side_pt: float) -> float:
    """Kırpılan bölgenin uzun kenarını hedef piksele getirecek render ölçeği.

    Farklı çözünürlükteki/küçültülmüş tabloları normalize eder (hücre başına yeterli
    piksel). Geçersiz girişte sabit VLM_TABLE_ZOOM'a düşer.
    """
    if long_side_pt <= 0:
        return settings.VLM_TABLE_ZOOM
    z = settings.VLM_TABLE_TARGET_LONG_PX / long_side_pt
    return max(settings.VLM_TABLE_MIN_ZOOM, min(z, settings.VLM_TABLE_MAX_ZOOM))


def crop_table_image(
    pdf_path: str,
    page_no: int,
    bbox_dict: Dict[str, Any],
    zoom: float | None = None,
    autorotate: bool = True,
    pad_frac: float | None = None,
) -> bytes | None:
    """Tablo bbox'ını PDF sayfasından kırpar; (opsiyonel dik çevirme sonrası) PNG byte döner.

    Args:
        pdf_path: Orijinal PDF dosya yolu.
        page_no: 1-tabanlı sayfa numarası.
        bbox_dict: Sınır kutusu (l, t, r, b, coord_origin).
        zoom: Render ölçeği. None → kırpılan bölgeye göre adaptif hesaplanır.
        autorotate: True ise kırpıntı Tesseract OSD ile dik çevrilir.
        pad_frac: bbox'ı sayfa boyutunun bu kesiri kadar dışa genişletir (None →
            settings.VLM_TABLE_BBOX_PAD_FRAC). Cetvel başlığı/alt notu kırpıntıya katar.

    Returns:
        Kırpılmış (ve gerekirse dik çevrilmiş) tablonun PNG byte'ları; başarısızsa None.
    """
    if pad_frac is None:
        pad_frac = settings.VLM_TABLE_BBOX_PAD_FRAC

    if not os.path.exists(pdf_path):
        _logger.error(f"Kırpma için PDF bulunamadı: {pdf_path}")
        return None

    try:
        doc = fitz.open(pdf_path)
        if page_no < 1 or page_no > len(doc):
            _logger.error(f"Sayfa {page_no} aralık dışı (toplam {len(doc)}): {pdf_path}")
            return None

        page = doc[page_no - 1]
        page_width = page.rect.width
        page_height = page.rect.height

        l = bbox_dict.get("l", 0.0)
        t = bbox_dict.get("t", 0.0)
        r = bbox_dict.get("r", 0.0)
        b = bbox_dict.get("b", 0.0)
        origin = str(bbox_dict.get("coord_origin", "")).upper()

        # BOTTOMLEFT koordinat sistemini PyMuPDF'in TOPLEFT'ine çevir
        if "BOTTOMLEFT" in origin:
            x0, y0, x1, y1 = l, page_height - t, r, page_height - b
        else:
            x0, y0, x1, y1 = l, t, r, b

        rect_x0, rect_y0 = min(x0, x1), min(y0, y1)
        rect_x1, rect_y1 = max(x0, x1), max(y0, y1)

        # Cetvel başlığı/alt notu yakalamak için bbox'ı dışa genişlet, sonra sayfaya clamp et.
        pad_x = pad_frac * page_width
        pad_y = pad_frac * page_height
        rect_x0 = max(0.0, rect_x0 - pad_x)
        rect_y0 = max(0.0, rect_y0 - pad_y)
        rect_x1 = min(page_width, rect_x1 + pad_x)
        rect_y1 = min(page_height, rect_y1 + pad_y)

        if rect_x1 - rect_x0 < 2 or rect_y1 - rect_y0 < 2:
            _logger.warning(
                f"Sayfa {page_no} tablo bbox'ı çok küçük: "
                f"({rect_x0}, {rect_y0}, {rect_x1}, {rect_y1})"
            )
            return None

        if zoom is None:
            zoom = _adaptive_zoom(max(rect_x1 - rect_x0, rect_y1 - rect_y0))

        print(f"    [CROP] Sayfa {page_no} kırpılıyor: zoom={zoom:.1f}×, "
              f"bbox=({rect_x0:.0f},{rect_y0:.0f})→({rect_x1:.0f},{rect_y1:.0f}) pt")
        rect = fitz.Rect(rect_x0, rect_y0, rect_x1, rect_y1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
        img_bytes = pix.tobytes("png")
        print(f"    [CROP] Kırpıntı hazır: {pix.width}×{pix.height}px, {len(img_bytes)//1024} KB")
        doc.close()

        if autorotate:
            img_bytes = deskew_to_upright(img_bytes, autorotate=True)
        return img_bytes

    except Exception as e:
        _logger.error(f"Sayfa {page_no} tablo kırpma hatası ({pdf_path}): {e}", exc_info=True)
        return None


# ----------------------------------------------------------------------
# Satır-bantı bölme + birleştirme (dev tablolar için)
# ----------------------------------------------------------------------

def _split_into_bands(img: Image.Image, max_h: int, overlap: int) -> List[Image.Image]:
    """Dik görüntü ``max_h``'yi aşarsa yatay satır-bantlarına böler (küçük örtüşmeli)."""
    w, h = img.size
    if h <= max_h:
        return [img]
    bands: List[Image.Image] = []
    step = max(1, max_h - overlap)
    y = 0
    while y < h:
        y1 = min(y + max_h, h)
        bands.append(img.crop((0, y, w, y1)))
        if y1 >= h:
            break
        y += step
    return bands


def _is_degenerate(md: str | None) -> bool:
    """VLM çıktısı dejenere mi? (tekrar döngüsü / hücre patlaması)

    7b bazı dev/yoğun görüntülerde tekrar döngüsüne giriyor (ör. ``GELDERSİ |`` yüzlerce
    kez) ya da tek satırda aşırı sütun üretiyor. Böyle çıktılar bantlamaya düşülerek
    kurtarılır.
    """
    if not md:
        return False
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    if lines:
        pipes = sorted(l.count("|") for l in lines)
        med = pipes[len(pipes) // 2]
        if med > 0 and pipes[-1] > 3 * med and pipes[-1] > 24:
            return True
    # Aynı kelimenin aşırı tekrarı (döngü göstergesi)
    from collections import Counter
    toks = re.findall(r"\w+", md)
    if toks and Counter(toks).most_common(1)[0][1] > 40:
        return True
    return False


def _find_separator_index(lines: List[str]) -> int | None:
    """Markdown tablo ayraç satırının (|---|---|) indeksini bulur."""
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|") and "-" in s and set(s) <= set("|-: "):
            return i
    return None


def _stitch_markdown_bands(parts: List[str]) -> str | None:
    """Bant markdown'larını birleştirir: 1. bant tam, sonrakilerden başlık+ayraç atılır.

    Örtüşme bölgesi, bant sınırında A, B, A', B' gibi ardışık-olmayan çiftler üretir
    (A' doğrudan A'yı takip etmez; aralarında B vardır). Bunları yakalamak için son
    N satır içinde ilk-hücre eşleşmesi kullanan kayan pencere dedup uygulanır.
    Sütun sayısı hafif farklı satırlarda (OCR kayması) daha fazla sütun içeren seçilir.
    """
    if not parts:
        return None
    out_lines: List[str] = []
    for idx, md in enumerate(parts):
        lines = md.splitlines()
        if idx == 0:
            out_lines.extend(lines)
            continue
        sep_idx = _find_separator_index(lines)
        data_lines = lines[sep_idx + 1:] if sep_idx is not None else lines
        out_lines.extend(data_lines)

    from collections import deque

    # Kayan pencere: örtüşme başına tipik 2-3 satır; 8 pencere yeterli tampon verir.
    _WINDOW = 8
    # (first_cell → deduped listesindeki indeks) kayan haritası
    seen: deque = deque(maxlen=_WINDOW)  # (first_cell, deduped_idx) çiftleri

    deduped: List[str] = []
    for line in out_lines:
        stripped = line.strip()
        if not stripped:
            deduped.append(line)
            continue
        # Tam eşleşme (mevcut mantık; hızlı yol)
        if deduped and stripped == deduped[-1].strip():
            continue
        # Veri satırı mı? (ayraç | --- | değil)
        if stripped.startswith("|") and not set(stripped) <= set("|-: "):
            cells = stripped.lstrip("|").split("|")
            first_cell = cells[0].strip() if cells else ""
            if first_cell:
                # Penceredeki eşleşmeyi ara
                match_idx = next(
                    (di for fc, di in seen if fc == first_cell), None
                )
                if match_idx is not None:
                    # Daha fazla sütun içeren satırı tut
                    if stripped.count("|") > deduped[match_idx].strip().count("|"):
                        deduped[match_idx] = line
                        # seen'deki indeks yerinde kalır; satır güncellendi
                    continue  # zaten temsil ediliyor, mevcut satırı atla
                seen.append((first_cell, len(deduped)))
        deduped.append(line)
    return "\n".join(deduped).strip()


# ----------------------------------------------------------------------
# VLM çağrısı
# ----------------------------------------------------------------------

def clean_markdown_response(text: str) -> str:
    """Model çıktısındaki backtick / chat-template artifact'larını temizler.

    Prompt yalnızca '## ' başlık ve '|' tablo satırı döndürmesini ister.
    Diğer her şey (<|im_start|>, addCriterion, yabancı dil karakter blokları vb.)
    model artifact'ıdır ve atılır.
    """
    text = text.strip()
    if text.startswith("```"):
        newline_idx = text.find("\n")
        text = text[newline_idx:].strip() if newline_idx != -1 else text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("|") or s.startswith("## "):
            kept.append(line)
    return "\n".join(kept).strip()


def extract_table_with_vlm(
    image_bytes: bytes,
    ollama_url: str,
    model_name: str,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    timeout: int | None = None,
) -> str | None:
    """Tablo görüntüsünü yerel Ollama VLM'ine gönderir; markdown tablo döndürür.

    num_ctx/num_predict/timeout None ise settings.VLM_TABLE_* kullanılır. Varsayılan
    KvSize (8192) dev tablolarda taşıp HTTP 500 döndürdüğü için bağlam büyütülür.
    """
    if num_ctx is None:
        num_ctx = settings.VLM_TABLE_NUM_CTX
    if num_predict is None:
        num_predict = settings.VLM_TABLE_NUM_PREDICT
    if timeout is None:
        timeout = settings.VLM_TABLE_TIMEOUT

    try:
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        url = f"{ollama_url.rstrip('/')}/api/chat"

        prompt = (
            "You are given an image of a Turkish government budget table (possibly a horizontal band of one). "
            "If there is a title or caption text above or beside the table (e.g. a 'CETVEL' / law name / "
            "classification heading), output it FIRST as a markdown heading line starting with '## '. "
            "Then output the table as a precise markdown table: keep the column headers, align cells to the "
            "correct columns, preserve every numeric value exactly, and include all total/subtotal rows. "
            "Ignore page footers and page numbers (e.g. 'Türkiye Büyük Millet Meclisi', 'Sıra Sayısı', '– 12 –'). "
            "Fix only obvious OCR errors. Do not add explanations or ``` code fences. "
            "Respond with only the optional '## ' heading and the markdown table."
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt, "images": [img_base64]}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }

        _logger.info(f"VLM'e tablo gönderiliyor: model={model_name} url={url} num_ctx={num_ctx}")
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        raw_content = response.json().get("message", {}).get("content", "")
        return clean_markdown_response(raw_content)

    except Exception as e:
        _logger.error(f"VLM modeli sorgulama hatası ({model_name}): {e}", exc_info=True)
        return None


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _band_with_header(header: Image.Image | None, band: Image.Image) -> Image.Image:
    """2..N bantların üstüne başlık şeridini dikey ekler (sütun hizası referansı)."""
    if header is None:
        return band
    w = max(header.width, band.width)
    out = Image.new("RGB", (w, header.height + band.height), "white")
    out.paste(header.convert("RGB"), (0, 0))
    out.paste(band.convert("RGB"), (0, header.height))
    return out


def extract_table_markdown(
    image_bytes: bytes,
    ollama_url: str,
    model_name: str,
    start_time: float | None = None,
    force_band: bool = False,
) -> str | None:
    """Dik tablo görüntüsünü markdown'a çevirir.

    VARSAYILAN: tek çağrı (model tüm tabloyu görür → en tutarlı sütun hizası + caption).
    Ölçümde tek-çağrı bantlamadan belirgin daha iyiydi (page-1: 47/50 vs 24/51 tutarlı
    sütun). ``force_band`` (--band) True ise tablo satır-bantlarına bölünür (A/B testi /
    tek çağrıya sığmayan dev tablolar); 2..N bantların üstüne başlık şeridi eklenir.
    """
    if start_time is None:
        start_time = time.time()

    # Varsayılan yol: tek çağrı. Çıktı dejenere değilse onu döndür; dejenere ise
    # (tekrar döngüsü) bantlamaya düş.
    if not force_band:
        single = extract_table_with_vlm(image_bytes, ollama_url, model_name)
        if not _is_degenerate(single):
            return single
        print(f"  [VLM] [{time.time() - start_time:.2f}s] Tek-çağrı çıktısı bozuk "
              f"(tekrar döngüsü) — bantlamaya düşülüyor.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        _logger.warning(f"VLM: görüntü açılamadı: {e}")
        return extract_table_with_vlm(image_bytes, ollama_url, model_name)

    overlap = settings.VLM_TABLE_TILE_OVERLAP_PX
    max_h = settings.VLM_TABLE_TILE_MAX_HEIGHT_PX
    # Tek banda sığsa bile en az 2 banda böl (--band testinin amacı budur).
    if img.height <= max_h and img.height > 4:
        max_h = img.height // 2 + overlap

    bands = _split_into_bands(img, max_h, overlap)

    if len(bands) == 1:
        return extract_table_with_vlm(image_bytes, ollama_url, model_name)

    print(f"  [VLM] [{time.time() - start_time:.2f}s] Büyük tablo {len(bands)} banda bölündü (yükseklik {img.size[1]}px).")
    header_px = settings.VLM_TABLE_HEADER_STRIP_PX
    header = img.crop((0, 0, img.width, min(header_px, img.height))) if header_px > 0 else None

    # Tüm bant görüntülerini önceden hazırla (header strip döngü dışında hazır).
    band_images: List[bytes] = [
        _png_bytes(band if i == 0 else _band_with_header(header, band))
        for i, band in enumerate(bands)
    ]

    max_workers = min(len(band_images), settings.VLM_TABLE_MAX_WORKERS)
    print(f"  [VLM] [{time.time() - start_time:.2f}s] "
          f"{len(band_images)} bant {max_workers} iş parçacığıyla paralel gönderiliyor...")

    def _send_band(args: tuple) -> "str | None":
        idx, img_bytes = args
        print(f"  [VLM] [{time.time() - start_time:.2f}s] Bant {idx + 1}/{len(band_images)} gönderiliyor...")
        try:
            return extract_table_with_vlm(img_bytes, ollama_url, model_name)
        except Exception as exc:
            _logger.warning(f"Bant {idx + 1} VLM çağrısı başarısız: {exc}")
            return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_send_band, enumerate(band_images)))

    parts = [md for md in results if md]
    if not parts:
        return None
    return _stitch_markdown_bands(parts)


# ----------------------------------------------------------------------
# Atom orkestrasyonu
# ----------------------------------------------------------------------

def process_atoms_with_vlm(
    pdf_path: str,
    atoms: List[Dict[str, Any]],
    ollama_url: str,
    model_name: str,
    start_time: float | None = None,
    force_band: bool = False,
) -> List[Dict[str, Any]]:
    """Bozuk tablo atomlarını VLM ile yeniden okur; metinlerini günceller.

    Yalnızca ``table_is_low_quality`` true olan ve bbox+page'i bulunan atomlar
    işlenir. Başarılı atomlar ``extracted_by="vlm"`` ile işaretlenir (cache/skip).
    ``force_band`` (--band): küçük tablolar da zorla bantlanır (A/B testi).
    """
    if start_time is None:
        start_time = time.time()
    updated_atoms: List[Dict[str, Any]] = []
    table_count = 0

    for atom in atoms:
        atom_copy = dict(atom)
        bbox = atom_copy.get("bbox")
        page = atom_copy.get("page")

        if table_is_low_quality(atom_copy):
            if not (bbox and page):
                print(f"  [VLM] [{time.time() - start_time:.2f}s] [WARN] Bozuk tablo bulundu ama bbox/sayfa yok — atlanıyor.")
            else:
                table_count += 1
                print(f"  [VLM] [{time.time() - start_time:.2f}s] Bozuk tablo (#{table_count}, sayfa {page}). Görsel kırpılıyor + dik çevriliyor...")
                image_bytes = crop_table_image(
                    pdf_path, page, bbox,
                    zoom=None,  # adaptif zoom (çözünürlük normalizasyonu)
                    autorotate=settings.VLM_TABLE_AUTOROTATE,
                )
                if image_bytes:
                    print(f"  [VLM] [{time.time() - start_time:.2f}s] Ollama VLM ({model_name}) çağrılıyor...")
                    vlm_table = extract_table_markdown(
                        image_bytes, ollama_url, model_name,
                        start_time=start_time, force_band=force_band,
                    )
                    if vlm_table:
                        print(f"  [VLM] [{time.time() - start_time:.2f}s] Tablo çıkarıldı. Karakter sayısı: {len(vlm_table)}")
                        atom_copy["text"] = vlm_table
                        atom_copy["extracted_by"] = "vlm"
                    else:
                        print(f"  [VLM] [{time.time() - start_time:.2f}s] [WARN] VLM çıkarımı başarısız — orijinal OCR metni korunuyor.")
                else:
                    print(f"  [VLM] [{time.time() - start_time:.2f}s] [WARN] Tablo görseli kırpılamadı — orijinal OCR metni korunuyor.")

        updated_atoms.append(atom_copy)

    return updated_atoms
