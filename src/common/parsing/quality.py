"""Tier-1 OCR kalite sinyalleri.

Parse sonrası belge bazında otomatik kalite metrikleri hesaplar:

  (a) Sayfa başına atom yoğunluğu — beklenen sayfa sayısına (atomlarda görülen
      en büyük sayfa numarası) göre anormal düşükse "low_atom_density" bayrağı.
  (b) Karakter sayısı sapması — aynı document_type'ın koleksiyon ortalamasından
      >%30 sapma → "char_count_outlier" bayrağı. Belgeler farklı uzunlukta
      olabildiği için karşılaştırma sayfa başına karakter üzerinden normalize
      edilir; istatistikler parse_cache/quality_stats.json içinde birikir.
  (c) OCR güven skoru — Docling ConversionResult'tan erişilebiliyorsa ortalama
      güven < 0.85 → "low_ocr_confidence" bayrağı.

Türkçe'ye özgü ek sinyaller (ı/İ-i/I karışıklığı, ünlü uyumu ihlali) ayrı bir
fonksiyondadır (turkish_ocr_signals) ve bilgi amaçlıdır — tek başına bayrak
üretmez.

Eşikler src/config/settings.py içindedir (QUALITY_* sabitleri).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------

def compute_quality(
    atoms: List[Dict[str, Any]],
    full_text: str,
    document_type: Optional[str] = None,
    ocr_mean_confidence: Optional[float] = None,
    stats_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Belge bazında kalite metriklerini ve bayraklarını hesaplar.

    Args:
        atoms:               MarkdownConverter atom listesi (page/pages metadata'lı).
        full_text:           Belgenin tam markdown metni.
        document_type:       Karakter sapması karşılaştırması için tip (ör. "tutanak").
                             None ise (b) kontrolü atlanır.
        ocr_mean_confidence: Docling'den çıkarılmış ortalama OCR güveni.
                             None ise (c) kontrolü atlanır (eski cache, OCR kapalı vb.).
        stats_key:           İstatistik deposunda bu belgeyi temsil eden anahtar
                             (dosya hash'i) — yeniden ingest'te çift sayımı önler.
                             None ise (b) kontrolü atlanır.

    Returns:
        quality dict: metrikler + "flags" listesi + "ocr_flagged" bool.
    """
    flags: List[str] = []

    atom_count = len(atoms)
    char_count = len(full_text or "")
    page_numbers = {p for a in atoms for p in a.get("pages", []) if p is not None}
    page_count = max(page_numbers) if page_numbers else 0
    atoms_per_page = atom_count / page_count if page_count else None
    chars_per_page = char_count / page_count if page_count else None
    empty_page_count = page_count - len(page_numbers) if page_count else 0

    # (a) Atom yoğunluğu
    if atoms_per_page is not None and atoms_per_page < settings.QUALITY_MIN_ATOMS_PER_PAGE:
        flags.append("low_atom_density")

    # (b) Karakter sayısı sapması (document_type ortalamasına göre)
    char_mean_for_type: Optional[float] = None
    if document_type and stats_key and chars_per_page is not None:
        deviates, char_mean_for_type = _update_and_check_char_stats(
            document_type, stats_key, chars_per_page
        )
        if deviates:
            flags.append("char_count_outlier")

    # (c) OCR güven skoru
    if (
        ocr_mean_confidence is not None
        and ocr_mean_confidence < settings.QUALITY_MIN_OCR_CONFIDENCE
    ):
        flags.append("low_ocr_confidence")

    return {
        "atom_count": atom_count,
        "char_count": char_count,
        "page_count": page_count,
        "empty_page_count": empty_page_count,
        "atoms_per_page": round(atoms_per_page, 2) if atoms_per_page is not None else None,
        "chars_per_page": round(chars_per_page, 1) if chars_per_page is not None else None,
        "char_mean_for_type": (
            round(char_mean_for_type, 1) if char_mean_for_type is not None else None
        ),
        "ocr_mean_confidence": (
            round(ocr_mean_confidence, 4) if ocr_mean_confidence is not None else None
        ),
        "turkish_signals": turkish_ocr_signals(full_text or ""),
        "flags": flags,
        "ocr_flagged": bool(flags),
    }


# ---------------------------------------------------------------------------
# OCR güven skoru çıkarımı (Docling sürümleri arasında savunmacı)
# ---------------------------------------------------------------------------

def extract_ocr_confidence(conversion_result: Any) -> Optional[float]:
    """Docling ConversionResult'tan ortalama OCR güvenini çıkarmaya çalışır.

    İki yol denenir; hiçbiri yoksa None döner (bayrak üretilmez):
      1. result.confidence.ocr_score  — Docling >= 2.x ConfidenceReport
      2. result.pages[*].cells[*].confidence — EasyOCR hücre güvenleri
    """
    if conversion_result is None:
        return None

    report = getattr(conversion_result, "confidence", None)
    score = getattr(report, "ocr_score", None)
    if isinstance(score, (int, float)) and not math.isnan(score):
        return float(score)

    confidences: List[float] = []
    for page in getattr(conversion_result, "pages", None) or []:
        for cell in getattr(page, "cells", None) or []:
            conf = getattr(cell, "confidence", None)
            if isinstance(conf, (int, float)) and not math.isnan(conf):
                confidences.append(float(conf))
    if confidences:
        return sum(confidences) / len(confidences)
    return None


# ---------------------------------------------------------------------------
# document_type bazlı karakter istatistikleri (kalıcı, JSON dosyası)
# ---------------------------------------------------------------------------

def _load_char_stats() -> Dict[str, Dict[str, float]]:
    path = settings.QUALITY_STATS_FILE
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _update_and_check_char_stats(
    document_type: str, stats_key: str, chars_per_page: float
) -> Tuple[bool, Optional[float]]:
    """Belgeyi tip ortalamasıyla karşılaştırır, sonra istatistiklere ekler.

    Karşılaştırma yalnızca DİĞER belgelere göre yapılır (kendisi hariç) ve
    en az QUALITY_STATS_MIN_DOCS başka belge varsa anlamlı sayılır.
    stats_key (dosya hash'i) anahtar olduğu için yeniden ingest idempotenttir.

    Returns:
        (sapma_var_mi, tip_ortalamasi)
    """
    stats = _load_char_stats()
    docs = stats.setdefault(document_type, {})
    others = [v for k, v in docs.items() if k != stats_key and isinstance(v, (int, float))]

    deviates = False
    mean: Optional[float] = None
    if len(others) >= settings.QUALITY_STATS_MIN_DOCS:
        mean = sum(others) / len(others)
        if mean > 0:
            deviates = (
                abs(chars_per_page - mean) / mean > settings.QUALITY_CHAR_DEVIATION_RATIO
            )

    docs[stats_key] = chars_per_page
    try:
        path = settings.QUALITY_STATS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [WARN] Kalite istatistik dosyası yazılamadı: {e}")

    return deviates, mean


# ---------------------------------------------------------------------------
# Türkçe'ye özgü ek sinyaller (opsiyonel, bilgi amaçlı)
# ---------------------------------------------------------------------------

_FRONT_VOWELS = set("eiöü")
_BACK_VOWELS = set("aıou")
_ALL_VOWELS = _FRONT_VOWELS | _BACK_VOWELS

# Küçük harfli kelime içinde büyük I/İ ya da büyük harfli kelime içinde küçük ı/i:
# tipik EasyOCR ı/İ-i/I karışıklığı imzası ("kIsa", "BAġBAKAN" benzeri bozulmalar).
_I_CONFUSION_RE = re.compile(r"[a-zçğıöşü][Iİ][a-zçğıöşü]|[A-ZÇĞÖŞÜ][ıi][A-ZÇĞÖŞÜ]")

_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+")
_LOWER_TR_RE = re.compile(r"^[abcçdefgğhıijklmnoöprsştuüvyz]+$")


def turkish_ocr_signals(text: str) -> Dict[str, Any]:
    """Türkçe metinde OCR bozulmasına işaret eden basit sezgisel sayaçlar.

    Kaba sinyallerdir — alıntı kelimeler (loanword) ünlü uyumunu meşru olarak
    bozabildiği için tek başına bayrak üretmez; quality dict içinde bilgi
    amaçlı taşınır.

    Returns:
        {"word_count", "i_confusion_ratio", "vowel_harmony_violation_ratio"}
    """
    words = _WORD_RE.findall(text)
    if not words:
        return {
            "word_count": 0,
            "i_confusion_ratio": 0.0,
            "vowel_harmony_violation_ratio": 0.0,
        }

    confused = sum(1 for w in words if _I_CONFUSION_RE.search(w))

    harmony_eligible = 0
    harmony_violations = 0
    for w in words:
        lw = w.lower()
        if len(lw) < 4 or not _LOWER_TR_RE.match(lw):
            continue
        vowels = [c for c in lw if c in _ALL_VOWELS]
        if len(vowels) < 2:
            continue
        harmony_eligible += 1
        if any(
            (a in _FRONT_VOWELS) != (b in _FRONT_VOWELS)
            for a, b in zip(vowels, vowels[1:])
        ):
            harmony_violations += 1

    return {
        "word_count": len(words),
        "i_confusion_ratio": round(confused / len(words), 4),
        "vowel_harmony_violation_ratio": (
            round(harmony_violations / harmony_eligible, 4) if harmony_eligible else 0.0
        ),
    }
